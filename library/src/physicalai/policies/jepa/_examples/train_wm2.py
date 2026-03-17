# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Minimal world model training script for PushT with DINOv2 encoder and AdaLN predictor.

This script is a simplified version of train.py that removes unused code paths and
uses dataclasses for configuration instead of YAML files. It follows the original
training logic exactly for the specific configuration:
  - pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save.yaml

Key simplifications:
  - Custom dataset type only (no MixedDataset)
  - DINOv2 encoder only (no V-JEPA)
  - AdaLN predictor only
  - Transition model training only (no head training)
  - Sequential rollout only (no parallel rollout)
  - Frozen encoder only
  - No eval-only modes

Usage:
    python train_wm2.py
"""

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import imageio
import lpips as lpips_lib
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
from app.plan_common.datasets.utils import init_data
from app.vjepa_wm.utils import init_opt, init_video_model, load_checkpoint
from app.vjepa_wm.video_wm import VideoWM
from src.datasets.utils.utils import get_dataset_paths
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""

    # Dataset - use data_paths for direct paths, or datasets for cluster-resolved names
    datasets: List[str] = field(default_factory=lambda: ["PushT"])
    data_paths: Optional[List[str]] = field(
        default_factory=lambda: ["/home/kdijkstr/jepa/datasets/pusht_noise"]
    )
    seed: int = 234
    img_size: int = 224

    # DataLoader
    batch_size: int = 8
    num_workers: int = 16
    pin_mem: bool = True
    persistent_workers: bool = True

    # Custom dataset parameters
    split_ratio: float = 0.9
    frameskip: int = 5
    action_skip: int = 1
    state_skip: int = 1
    normalize_action: bool = True
    traj_subset: bool = True
    filter_first_episodes: Optional[int] = None
    filter_tasks: Optional[List[str]] = None
    num_hist: int = 3
    num_pred: int = 1
    with_reward: bool = False


@dataclass
class DataAugConfig:
    """Data augmentation configuration."""

    auto_augment: bool = False
    random_horizontal_flip: bool = False
    motion_shift: bool = False
    random_resize_aspect_ratio: Tuple[float, float] = (1.0, 1.0)
    random_resize_scale: Tuple[float, float] = (1.777, 1.777)
    reprob: float = 0.0
    normalize: Tuple[Tuple[float, ...], Tuple[float, ...]] = (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Shared
    grid_size: int = 16
    tubelet_size_enc: int = 1
    use_activation_checkpointing: bool = False
    action_conditioning: str = "token"
    proprio_encoding: str = "feature"
    num_frames_pred: int = 4

    # Visual encoder (DINOv2)
    enc_type: str = "dino"
    enc_version: str = "dinov2_vits14"
    embed_dim: int = 384

    # Action encoder
    action_tokens: int = 1
    action_emb_dim: int = 0
    act_mlp: bool = False
    action_encoder_inpred: bool = True

    # Proprio encoder
    proprio_tokens: int = 0
    proprio_emb_dim: int = 16
    prop_mlp: bool = False
    proprio_encoder_inpred: bool = False

    # Predictor (AdaLN)
    tubelet_size: int = 1
    pred_num_heads: int = 16
    pred_depth: int = 6
    pred_embed_dim: int = 384
    pred_use_extrinsics: bool = False
    pred_type: str = "AdaLN"
    act_pred_projector: bool = False
    uniform_power: bool = True
    use_SiLU: bool = False
    use_rope: bool = True

    # WM encoding
    batchify_video: bool = True
    dup_image: bool = False
    normalize_reps: bool = False

    # Rollout
    rollout_steps: int = 2
    train_rollout_prefixes: str = "random"
    rollout_stop_gradient: bool = True
    ctxt_window_train_rollout: int = 3

    # Attention
    local_window_time: int = 3
    local_window_h: int = -1
    local_window_w: int = -1


@dataclass
class LossConfig:
    """Loss configuration."""

    cos_loss_weight: float = 0.0
    l1_loss_weight: float = 0.0
    l2_loss_weight: float = 1.0
    smooth_l1_loss_weight: float = 0.0


@dataclass
class OptimizationConfig:
    """Optimization configuration."""

    iterations_per_epoch: Optional[int] = None
    ipe_scale: float = 1.0
    clip_grad: float = 1.0
    use_radamw: bool = False
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-7
    final_weight_decay: float = 1e-6
    num_epochs: int = 4
    warmup: int = 0
    start_lr: float = 5e-4
    ref_lr: float = 5e-4
    final_lr: float = 5e-4


@dataclass
class EvalConfig:
    """Light evaluation configuration."""

    do_data_traj_rollout_eval: bool = True
    data_traj_eval_rollout_steps: int = 6
    data_traj_decode_gt: bool = True
    data_traj_eval_ctxt_window: int = 3
    light_eval_freq: int = 300


@dataclass
class LoggingConfig:
    """Logging and wandb configuration."""

    write_tag: str = "jepa"
    use_wandb: bool = False
    debug: bool = False
    project: str = "vjepa_wm"
    disable_wandb_media: bool = True
    log_media_locally: bool = True


@dataclass
class TrainConfig:
    """Complete training configuration."""

    # Output
    folder: str = "./output/train_wm2"
    checkpoint_folder: Optional[str] = None

    # Meta
    seed: int = 234
    dtype: str = "bfloat16"
    freeze_encoder: bool = True
    load_checkpoint: bool = True
    load_opt_scale_epoch: bool = True
    eval_freq: int = 1
    save_every_freq: int = 1

    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    data_aug: DataAugConfig = field(default_factory=DataAugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# =============================================================================
# Constants
# =============================================================================

_GLOBAL_SEED = 0
LOG_FREQ = 10
CHECKPOINT_FREQ = 1
SAVE_FREQ = 100  # Save checkpoint every N iterations within an epoch

logger = get_logger(__name__)


# =============================================================================
# Helper Classes
# =============================================================================


class Trainer:
    """Handles logging to wandb and local files."""

    def __init__(
        self, cfg: LoggingConfig, folder: str, rank: int, ipe: int, debug: bool = False
    ):
        self.cfg = cfg
        self.folder = folder
        self.rank = rank
        self.ipe = ipe
        self.debug = debug

        self.local_log_dir = None
        if cfg.log_media_locally and rank == 0:
            self.local_log_dir = os.path.join(folder, "local_logs")
            os.makedirs(self.local_log_dir, exist_ok=True)

        if cfg.use_wandb and rank == 0:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize wandb with resume support."""
        project_name = (
            self.cfg.project if not self.debug else f"{self.cfg.project}_debug"
        )
        wandb_run_id_file = os.path.join(self.folder, "wandb_run_id.txt")

        if os.path.exists(wandb_run_id_file):
            with open(wandb_run_id_file, "r") as f:
                wandb_run_id = f.read().strip()
            wandb.init(
                project=project_name, id=wandb_run_id, resume="allow", dir=self.folder
            )
            logger.info(f"Resuming Wandb run {wandb_run_id}")
        else:
            wandb.init(project=project_name, dir=self.folder)
            with open(wandb_run_id_file, "w") as f:
                f.write(wandb.run.id)

        wandb.run.name = os.path.basename(self.folder)

    def log(
        self,
        epoch: int,
        itr: int,
        losses: dict,
        total_stats: dict,
        eval_losses: Optional[dict] = None,
        eval_total_stats: Optional[dict] = None,
        image_stats: Optional[dict] = None,
    ):
        """Log metrics to wandb."""
        log_dict = {"epoch": epoch + 1, "itr": itr}

        # Add losses
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            log_dict[key] = value

        # Add stats
        for key, value in total_stats.items():
            log_dict[key] = value

        # Add eval losses
        if eval_losses is not None:
            for key, value in eval_losses.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().item()
                log_dict[key] = value

        # Add eval stats
        if eval_total_stats is not None:
            for key, value in eval_total_stats.items():
                log_dict[key] = value

        # Handle media
        if image_stats:
            if self.cfg.log_media_locally and self.rank == 0:
                self._log_media_local(image_stats, epoch, itr)
            if not self.cfg.disable_wandb_media:
                log_dict.update(image_stats)

        # Console logging
        if "loss" in log_dict and itr % LOG_FREQ == 0:
            logger.info(f"[{epoch + 1}, {itr:5d}] loss: {log_dict['loss']:.3f}")

        # Wandb logging
        if self.cfg.use_wandb and self.rank == 0:
            wandb.log(log_dict)

    def _log_media_local(self, image_stats: dict, epoch: int, itr: int):
        """Save media locally."""
        step = epoch * self.ipe + itr
        for key, value in image_stats.items():
            subfolder = os.path.join(self.local_log_dir, "/".join(key.split("/")))
            os.makedirs(subfolder, exist_ok=True)

            if isinstance(value, wandb.Video):
                filename = f"{step}.gif"
                frames = value._prepare_video(value.data)
                duration = 1.0 / 10
                imageio.mimsave(
                    os.path.join(subfolder, filename), frames, duration=duration, loop=0
                )
            elif isinstance(value, wandb.Image):
                filename = f"{step}.pdf"
                value.image.save(os.path.join(subfolder, filename))


# =============================================================================
# Data Loading
# =============================================================================


def create_data_loaders(cfg: TrainConfig, transform, world_size: int, rank: int):
    """Create train and validation data loaders."""
    # Use direct paths if provided, otherwise resolve from dataset names
    if cfg.data.data_paths is not None:
        dataset_paths = cfg.data.data_paths
    else:
        dataset_paths = get_dataset_paths(cfg.data.datasets)

    data_kwargs = {
        "data_paths": dataset_paths,
        "val_data_paths": None,
        "transform": transform,
        "world_size": world_size,
        "rank": rank,
        "dataset_type": "custom",
        "img_size": cfg.data.img_size,
        "seed": cfg.data.seed,
        # Loader config
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_mem": cfg.data.pin_mem,
        "persistent_workers": cfg.data.persistent_workers,
        # Custom config
        "split_ratio": cfg.data.split_ratio,
        "frameskip": cfg.data.frameskip,
        "action_skip": cfg.data.action_skip,
        "state_skip": cfg.data.state_skip,
        "normalize_action": cfg.data.normalize_action,
        "traj_subset": cfg.data.traj_subset,
        "filter_first_episodes": cfg.data.filter_first_episodes,
        "filter_tasks": cfg.data.filter_tasks,
        "num_hist": cfg.data.num_hist,
        "num_pred": cfg.data.num_pred,
        "with_reward": cfg.data.with_reward,
    }

    (
        dataset,
        val_dataset,
        traj_dataset,
        val_traj_dataset,
        unsupervised_loader,
        val_unsupervised_loader,
        unsupervised_sampler,
        viz_val_data_loader,
    ) = init_data(**data_kwargs)

    return (
        traj_dataset,
        val_traj_dataset,
        unsupervised_loader,
        val_unsupervised_loader,
        unsupervised_sampler,
    )


# =============================================================================
# Model Creation
# =============================================================================


def create_model(cfg: TrainConfig, traj_dataset, device: torch.device):
    """Create encoder, predictor, and action/proprio encoders."""
    # Compute action and proprio dimensions
    use_action = cfg.model.action_tokens > 0 or cfg.model.action_emb_dim > 0
    use_proprio = cfg.model.proprio_tokens > 0 or cfg.model.proprio_emb_dim > 0

    actions_per_vid_feat = (
        cfg.model.tubelet_size_enc * cfg.data.frameskip // cfg.data.action_skip
    )
    model_action_dim = (
        traj_dataset.action_dim * actions_per_vid_feat if use_action else None
    )

    proprio_multiplier = (
        cfg.model.tubelet_size_enc * cfg.data.frameskip // cfg.data.state_skip
    )
    model_proprio_dim = (
        traj_dataset.proprio_dim * cfg.model.tubelet_size_enc // cfg.data.state_skip
        if use_proprio
        else None
    )

    # Model kwargs
    model_kwargs = {
        "device": device,
        "img_size": cfg.data.img_size,
        "action_dim": model_action_dim,
        "proprio_dim": model_proprio_dim,
        "use_proprio": use_proprio,
        "use_action": use_action,
        # Model architecture
        "grid_size": cfg.model.grid_size,
        "tubelet_size_enc": cfg.model.tubelet_size_enc,
        "use_activation_checkpointing": cfg.model.use_activation_checkpointing,
        "action_conditioning": cfg.model.action_conditioning,
        "proprio_encoding": cfg.model.proprio_encoding,
        "num_frames_pred": cfg.model.num_frames_pred,
        # Visual encoder
        "enc_type": cfg.model.enc_type,
        "enc_version": cfg.model.enc_version,
        "embed_dim": cfg.model.embed_dim,
        # Action encoder
        "action_tokens": cfg.model.action_tokens,
        "action_emb_dim": cfg.model.action_emb_dim,
        "act_mlp": cfg.model.act_mlp,
        "action_encoder_inpred": cfg.model.action_encoder_inpred,
        # Proprio encoder
        "proprio_tokens": cfg.model.proprio_tokens,
        "proprio_emb_dim": cfg.model.proprio_emb_dim,
        "prop_mlp": cfg.model.prop_mlp,
        "proprio_encoder_inpred": cfg.model.proprio_encoder_inpred,
        # Predictor
        "tubelet_size": cfg.model.tubelet_size,
        "pred_num_heads": cfg.model.pred_num_heads,
        "pred_depth": cfg.model.pred_depth,
        "pred_embed_dim": cfg.model.pred_embed_dim,
        "pred_use_extrinsics": cfg.model.pred_use_extrinsics,
        "pred_type": cfg.model.pred_type,
        "act_pred_projector": cfg.model.act_pred_projector,
        "uniform_power": cfg.model.uniform_power,
        "use_SiLU": cfg.model.use_SiLU,
        "use_rope": cfg.model.use_rope,
        # Attention config
        "cfgs_attn_pattern": {
            "local_window_time": cfg.model.local_window_time,
            "local_window_h": cfg.model.local_window_h,
            "local_window_w": cfg.model.local_window_w,
        },
    }

    predictor, encoder, action_encoder, proprio_encoder = init_video_model(
        **model_kwargs
    )

    return (
        predictor,
        encoder,
        action_encoder,
        proprio_encoder,
        model_action_dim,
        model_proprio_dim,
        use_action,
        use_proprio,
    )


def create_world_model(
    cfg: TrainConfig,
    encoder,
    predictor,
    action_encoder,
    proprio_encoder,
    model_action_dim: int,
    model_proprio_dim: int,
    use_action: bool,
    use_proprio: bool,
    optimizer,
    scaler,
    device: torch.device,
    mixed_precision: bool,
):
    """Create the VideoWM world model."""
    wm_kwargs = {
        "device": device,
        # Model components
        "encoder": encoder,
        "predictor": predictor,
        "action_encoder": action_encoder,
        "proprio_encoder": proprio_encoder,
        # Dimensions
        "action_dim": model_action_dim,
        "proprio_dim": model_proprio_dim,
        "use_proprio": use_proprio,
        "use_action": use_action,
        # Architecture
        "action_tokens": cfg.model.action_tokens,
        "proprio_tokens": cfg.model.proprio_tokens,
        "grid_size": cfg.model.grid_size,
        "tubelet_size_enc": cfg.model.tubelet_size_enc,
        "action_conditioning": cfg.model.action_conditioning,
        "proprio_encoding": cfg.model.proprio_encoding,
        "enc_type": cfg.model.enc_type,
        "pred_type": cfg.model.pred_type,
        "action_encoder_inpred": cfg.model.action_encoder_inpred,
        "proprio_encoder_inpred": cfg.model.proprio_encoder_inpred,
        # WM encoding
        "batchify_video": cfg.model.batchify_video,
        "dup_image": cfg.model.dup_image,
        "normalize_reps": cfg.model.normalize_reps,
        # Data
        "action_skip": cfg.data.action_skip,
        "frameskip": cfg.data.frameskip,
        "img_size": cfg.data.img_size,
        # Optimization
        "scaler": scaler,
        "optimizer": optimizer,
        "clip_grad": cfg.optimization.clip_grad,
        "mixed_precision": mixed_precision,
        "use_radamw": cfg.optimization.use_radamw,
        # Loss config
        "cfgs_loss": {
            "l2_loss_weight": cfg.loss.l2_loss_weight,
            "l1_loss_weight": cfg.loss.l1_loss_weight,
            "cos_loss_weight": cfg.loss.cos_loss_weight,
            "smooth_l1_loss_weight": cfg.loss.smooth_l1_loss_weight,
        },
        # No heads
        "heads": {},
    }

    return VideoWM(**wm_kwargs)


# =============================================================================
# Checkpoint Handling
# =============================================================================


def save_checkpoint(
    world_model: VideoWM,
    optimizer,
    scaler,
    epoch: int,
    path: str,
    rank: int,
    cfg: TrainConfig,
    iteration: int = 0,
):
    """Save training checkpoint."""
    if rank != 0:
        return

    save_dict = {
        "predictor": world_model.predictor.state_dict()
        if world_model.predictor is not None
        else None,
        "opt": optimizer.state_dict() if optimizer is not None else None,
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "iteration": iteration,
    }

    if world_model.action_encoder is not None and not cfg.model.action_encoder_inpred:
        save_dict["action_encoder"] = world_model.action_encoder.state_dict()

    if world_model.proprio_encoder is not None and not cfg.model.proprio_encoder_inpred:
        save_dict["proprio_encoder"] = world_model.proprio_encoder.state_dict()

    try:
        torch.save(save_dict, path)
        logger.info(f"Saved checkpoint to {path}")
    except Exception as e:
        logger.info(f"Encountered exception when saving checkpoint: {e}")


# =============================================================================
# Batch Loading
# =============================================================================


class BatchLoader:
    """Helper class to load batches from data loaders with automatic refresh."""

    def __init__(
        self,
        train_loader,
        val_loader,
        unsupervised_loader,
        val_unsupervised_loader,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.train_iter = iter(train_loader)
        self.val_iter = iter(val_loader) if val_loader is not None else None
        self.unsupervised_loader = unsupervised_loader
        self.val_unsupervised_loader = val_unsupervised_loader
        self.device = device
        self.dtype = dtype

    def get_batch(self, train: bool = True):
        """Get a batch of data."""
        try:
            if train:
                obs, action, state, reward = next(self.train_iter)
            else:
                obs, action, state, reward = next(self.val_iter)
        except StopIteration:
            logger.info("Exhausted data loader. Refreshing...")
            if train:
                self.train_iter = iter(self.unsupervised_loader)
                obs, action, state, reward = next(self.train_iter)
            else:
                self.val_iter = iter(self.val_unsupervised_loader)
                obs, action, state, reward = next(self.val_iter)

        for k in obs.keys():
            obs[k] = obs[k].to(self.device, dtype=self.dtype, non_blocking=True)
        action = action.to(self.device, dtype=self.dtype, non_blocking=True)
        state = state.to(self.device, dtype=self.dtype, non_blocking=True)
        reward = reward.to(self.device, dtype=self.dtype, non_blocking=True)

        return obs, action, state, reward


# =============================================================================
# Training Step
# =============================================================================


def train_step(
    world_model: VideoWM,
    obs: dict,
    action: torch.Tensor,
    state: torch.Tensor,
    scheduler,
    wd_scheduler,
    cfg: TrainConfig,
    dtype: torch.dtype,
    mixed_precision: bool,
) -> Tuple[float, dict, dict]:
    """Execute a single training step."""
    # Update learning rates
    rates = {
        "info/transition_model/lr": scheduler.step(),
        "info/transition_model/wd": wd_scheduler.step(),
    }

    total_stats = {}
    total_transition_loss = 0.0

    # Log action stats
    if action is not None:
        total_stats.update(
            {
                "act_mean": action.mean(),
                "act_std": action.std(),
                "act_min": action.min(),
                "act_max": action.max(),
            }
        )

    # Forward pass with mixed precision
    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
        video_features, proprio_features, action_features = world_model.encode(
            obs, action
        )

        # Predict one step in the future using teacher forcing
        pred_video_features, pred_action_features, pred_proprio_features = (
            world_model.forward_pred(
                video_features,
                action_features,
                proprio_features,
            )
        )

        predictor_losses = world_model.compute_loss(
            pred_video_features,
            pred_proprio_features,
            video_features,
            proprio_features,
            shift=1,
        )

    # Compute loss (normalized by rollout steps)
    predictor_loss = predictor_losses.get("loss", 0.0) / (cfg.model.rollout_steps + 1)
    total_transition_loss += predictor_loss

    # Track per-step losses
    train_rollout_result = {}
    stats = defaultdict(list)
    for k, val in predictor_losses.items():
        if isinstance(val, torch.Tensor):
            val = val.detach().clone()
        else:
            val = torch.tensor(val)
        stats[k].append(val.unsqueeze(0))

    stats = {k: torch.stack(v).mean(0) for k, v in stats.items()}
    for k, v in stats.items():
        for j in range(len(v)):
            train_rollout_result[f"train_rollout/{k}/{j + 1}"] = v[j].item()

    # Sequential rollout training
    if cfg.model.rollout_steps > 1:
        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
            if cfg.model.train_rollout_prefixes == "random":
                prefixes = torch.randint(
                    video_features.shape[1] - cfg.model.rollout_steps, size=(1,)
                )
            elif cfg.model.train_rollout_prefixes == "first":
                prefixes = [0]
            elif cfg.model.train_rollout_prefixes == "all":
                prefixes = list(
                    range(video_features.shape[1] - cfg.model.rollout_steps)
                )

            rollout_stats = defaultdict(list)
            for t in prefixes:
                rollout_losses, total_rollout_loss, _, _ = world_model.rollout(
                    video_features=video_features,
                    pred_video_features=pred_video_features,
                    proprio_features=proprio_features,
                    pred_proprio_features=pred_proprio_features,
                    action_features=action_features,
                    action_noise=0.0,
                    loss_weight=1.0 / len(prefixes),
                    rollout_steps=cfg.model.rollout_steps - 1,
                    rollout_stop_gradient=cfg.model.rollout_stop_gradient,
                    ctxt_window=cfg.model.ctxt_window_train_rollout,
                    mode="sequential",
                    t=t,
                )
                total_transition_loss += total_rollout_loss
                for k in rollout_losses:
                    rollout_stats[k].append(rollout_losses[k])

            rollout_stats = {
                k: torch.stack(v).mean(0) for k, v in rollout_stats.items()
            }
            for k, v in rollout_stats.items():
                for j in range(len(v)):
                    train_rollout_result[f"train_rollout/{k}/{j + 2}"] = v[j].item()

    total_stats.update(train_rollout_result)

    # Build losses dict
    losses = {
        "predictor_loss": total_transition_loss.item()
        if isinstance(total_transition_loss, torch.Tensor)
        else total_transition_loss,
        "head_loss": 0.0,
    }
    losses["loss"] = losses["predictor_loss"]

    # Backward pass and optimization
    world_model.backward(total_transition_loss)
    grad_stats, optim_stats = world_model.optimization_step()

    # Log gradient and optimizer stats
    if grad_stats is not None:
        total_stats["optim/transition_model/grad_norm"] = grad_stats.global_norm
    if optim_stats is not None:
        total_stats["optim/transition_model/first_moment"] = optim_stats.get(
            "exp_avg"
        ).avg
        total_stats["optim/transition_model/second_moment"] = optim_stats.get(
            "exp_avg_sq"
        ).avg

    total_stats.update(rates)

    return losses["loss"], losses, total_stats


# =============================================================================
# Validation Step
# =============================================================================


@torch.no_grad()
def validation_step(
    world_model: VideoWM,
    obs: dict,
    action: torch.Tensor,
    state: torch.Tensor,
    lpips_model,
    inverse_transform,
    cfg: TrainConfig,
    dtype: torch.dtype,
    mixed_precision: bool,
) -> Tuple[dict, dict, dict]:
    """Execute a validation step with rollout evaluation."""
    world_model.eval()

    eval_total_stats = {}
    image_stats = {}

    # Encode
    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
        video_features, proprio_features, action_features = world_model.encode(
            obs, action
        )
        pred_video_features, _, pred_proprio_features = world_model.forward_pred(
            video_features,
            action_features,
            proprio_features,
        )

    # Validation rollout
    if cfg.eval.do_data_traj_rollout_eval:
        eval_rollout_result, eval_image_samples = _run_val_rollout(
            world_model=world_model,
            video_features=video_features,
            action_features=action_features,
            proprio_features=proprio_features,
            pred_video_features=pred_video_features,
            pred_proprio_features=pred_proprio_features,
            gt_obs=obs,
            gt_state=state,
            lpips_model=lpips_model,
            cfg=cfg,
            dtype=dtype,
            mixed_precision=mixed_precision,
            prefix="data_traj",
        )
        eval_total_stats.update(eval_rollout_result)

        # Create visualization images
        if eval_image_samples is not None:
            t = eval_image_samples.shape[1]
            b = min(4, obs["visual"].shape[0])

            rgb_v = inverse_transform(
                obs["visual"][:, :: cfg.model.tubelet_size_enc].cpu()
            )
            rgb_v = (255.0 * rgb_v).clip(0.0, 255.0).to(torch.uint8)
            rgb_v = rearrange(rgb_v, "b t (v c) h w -> b t v h w c", c=3)

            rgb = torch.stack([eval_image_samples, rgb_v[:, -t:]], dim=2)[:b]
            rgb = rearrange(rgb, "b t e v h w c -> (b v e h) (t w) c").cpu().numpy()

            animation = torch.stack([rgb_v[:, -t:], eval_image_samples], dim=2)[:b]
            animation = (
                rearrange(animation, "b t e v h w c -> t c (b v h) (e w)").cpu().numpy()
            )

            image_stats.update(
                {
                    "data_traj/image_rollouts": wandb.Image(rgb),
                    "data_traj/image_animated_rollout": wandb.Video(animation, fps=6),
                }
            )

    world_model.train()
    return {}, eval_total_stats, image_stats


def _run_val_rollout(
    world_model: VideoWM,
    video_features: torch.Tensor,
    action_features: torch.Tensor,
    proprio_features: torch.Tensor,
    pred_video_features: torch.Tensor,
    pred_proprio_features: torch.Tensor,
    gt_obs: dict,
    gt_state: torch.Tensor,
    lpips_model,
    cfg: TrainConfig,
    dtype: torch.dtype,
    mixed_precision: bool,
    prefix: str = "val_rollout",
) -> Tuple[dict, Optional[torch.Tensor]]:
    """Run validation rollout and compute metrics."""
    # For sequential mode with pred_video_features, the suffix starts at t+2
    # so we need rollout_steps <= T - t - 2. For t=0, this means T - 2.
    T = video_features.shape[1]
    rollout_steps = min(cfg.eval.data_traj_eval_rollout_steps, T - 2)

    # Skip if we don't have enough timesteps for a rollout
    if rollout_steps <= 0:
        logger.warning(
            f"Skipping validation rollout: not enough timesteps "
            f"(T={T}, need at least 3)"
        )
        return {}, None

    ctxt_window = cfg.eval.data_traj_eval_ctxt_window
    val_rollout_result = {}

    # Only use t=0 to avoid suffix length issues with multiple prefixes
    # (suffix at t has length T - t - 2, which decreases with t)
    t = 0

    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
        rollout_losses, _, last_vid_feats, last_prop_feats = world_model.rollout(
            video_features=video_features,
            pred_video_features=pred_video_features,
            proprio_features=proprio_features,
            pred_proprio_features=pred_proprio_features,
            action_features=action_features,
            action_noise=0.0,
            rollout_steps=rollout_steps,
            rollout_stop_gradient=True,
            debug=True,
            ctxt_window=ctxt_window,
            mode="sequential",
            t=t,
        )

        # Aggregate rollout losses
        for k, v in rollout_losses.items():
            if isinstance(v, torch.Tensor):
                for j in range(len(v)):
                    val_rollout_result[f"{prefix}/val_rollout/{k}/{j + 1}"] = v[
                        j
                    ].item()

        # Decode images if we have an image head (not in this minimal version)
        image_samples = None

    return val_rollout_result, image_samples


# =============================================================================
# Main Training Loop
# =============================================================================


def train(cfg: TrainConfig):
    """Main training function."""
    # Setup seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # Initialize distributed training
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # Set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # Setup paths
    folder = cfg.folder
    checkpoint_folder = cfg.checkpoint_folder or folder
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Determine dtype
    if cfg.dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif cfg.dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    logger.info(f"Using dtype: {cfg.dtype}")

    # Setup paths for checkpoints
    pref_tag = f"{cfg.logging.write_tag}-" if cfg.logging.write_tag else ""
    latest_file = f"{pref_tag}latest.pth.tar"
    latest_path = os.path.join(checkpoint_folder, latest_file)
    train_log_file = os.path.join(folder, f"log_r{rank}.csv")

    # Create transforms
    transform = make_transforms(
        img_size=cfg.data.img_size,
        auto_augment=cfg.data_aug.auto_augment,
        random_horizontal_flip=cfg.data_aug.random_horizontal_flip,
        motion_shift=cfg.data_aug.motion_shift,
        random_resize_aspect_ratio=cfg.data_aug.random_resize_aspect_ratio,
        random_resize_scale=cfg.data_aug.random_resize_scale,
        reprob=cfg.data_aug.reprob,
        normalize=cfg.data_aug.normalize,
    )
    inverse_transform = make_inverse_transforms(
        img_size=cfg.data.img_size,
        normalize=cfg.data_aug.normalize,
    )

    # Create data loaders
    logger.info("Creating data loaders...")
    (
        traj_dataset,
        val_traj_dataset,
        unsupervised_loader,
        val_unsupervised_loader,
        unsupervised_sampler,
    ) = create_data_loaders(cfg, transform, world_size, rank)

    # Compute iterations per epoch
    ipe = cfg.optimization.iterations_per_epoch
    if ipe is None:
        ipe = len(unsupervised_loader)
    logger.info(
        f"Iterations per epoch: {ipe} (dataset size: {len(unsupervised_loader)})"
    )

    # Create model
    logger.info("Creating model...")
    (
        predictor,
        encoder,
        action_encoder,
        proprio_encoder,
        model_action_dim,
        model_proprio_dim,
        use_action,
        use_proprio,
    ) = create_model(cfg, traj_dataset, device)

    # Create optimizer and schedulers
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        encoder=encoder,
        freeze_encoder=cfg.freeze_encoder,
        iterations_per_epoch=ipe,
        ipe_scale=cfg.optimization.ipe_scale,
        clip_grad=cfg.optimization.clip_grad,
        use_radamw=cfg.optimization.use_radamw,
        betas=cfg.optimization.betas,
        eps=cfg.optimization.eps,
        weight_decay=cfg.optimization.weight_decay,
        final_weight_decay=cfg.optimization.final_weight_decay,
        num_epochs=cfg.optimization.num_epochs,
        warmup=cfg.optimization.warmup,
        start_lr=cfg.optimization.start_lr,
        ref_lr=cfg.optimization.ref_lr,
        final_lr=cfg.optimization.final_lr,
        mixed_precision=mixed_precision,
    )

    # Load checkpoint if exists
    start_epoch = 0
    start_iteration = 0
    if cfg.load_checkpoint and os.path.exists(latest_path):
        logger.info(f"Loading checkpoint from {latest_path}")
        (
            predictor,
            action_encoder,
            proprio_encoder,
            _,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=latest_path,
            predictor=predictor,
            action_encoder=action_encoder,
            proprio_encoder=proprio_encoder,
            heads={},
            opt=optimizer,
            scaler=scaler,
            load_opt_scale_epoch=cfg.load_opt_scale_epoch,
            load_heads=False,
            train_heads=False,
            train_predictor=True,
        )

        # Load iteration from checkpoint (for resuming mid-epoch)
        checkpoint = torch.load(latest_path, map_location="cpu")
        start_iteration = checkpoint.get("iteration", 0)
        del checkpoint
        logger.info(f"Resuming from epoch {start_epoch}, iteration {start_iteration}")

        # Resume schedulers
        if (
            cfg.load_opt_scale_epoch
            and scheduler is not None
            and wd_scheduler is not None
        ):
            for _ in range(start_epoch * ipe + start_iteration):
                scheduler.step()
                wd_scheduler.step()

    # Wrap models in DDP (only if distributed training is initialized)
    use_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    if use_ddp:
        if action_encoder is not None:
            action_encoder = DDP(
                action_encoder, static_graph=False, find_unused_parameters=False
            )
        if proprio_encoder is not None:
            proprio_encoder = DDP(
                proprio_encoder, static_graph=False, find_unused_parameters=False
            )
        predictor = DDP(predictor, static_graph=False, find_unused_parameters=False)

    # Create world model
    world_model = create_world_model(
        cfg=cfg,
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        model_action_dim=model_action_dim,
        model_proprio_dim=model_proprio_dim,
        use_action=use_action,
        use_proprio=use_proprio,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        mixed_precision=mixed_precision,
    )

    # Initialize LPIPS for evaluation
    lpips_model = lpips_lib.LPIPS(net="vgg").eval().to(device)

    # Create trainer for logging
    trainer = Trainer(cfg.logging, folder, rank, ipe)

    # Create CSV logger
    train_csv_logger = None
    train_csv_logger_columns = []

    # Initialize batch loader
    batch_loader = BatchLoader(
        train_loader=unsupervised_loader,
        val_loader=val_unsupervised_loader,
        unsupervised_loader=unsupervised_loader,
        val_unsupervised_loader=val_unsupervised_loader,
        device=device,
        dtype=dtype,
    )

    # =============================================================================
    # Training Loop
    # =============================================================================

    logger.info("Starting training...")

    # Save initial checkpoint if starting fresh
    if start_epoch == 0 and start_iteration == 0 and rank == 0:
        logger.info("Saving initial checkpoint...")
        save_checkpoint(
            world_model,
            optimizer,
            scaler,
            0,
            latest_path,
            rank,
            cfg,
            iteration=0,
        )

    for epoch in range(start_epoch, cfg.optimization.num_epochs):
        logger.info("\n" + "=" * 50)
        logger.info(f"Epoch {epoch + 1}/{cfg.optimization.num_epochs}")
        logger.info("=" * 50)

        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        wall_time_meter = AverageMeter()

        # Determine starting iteration (for resuming mid-epoch)
        iter_start = start_iteration if epoch == start_epoch else 0

        # Skip batches if resuming mid-epoch
        if iter_start > 0:
            logger.info(
                f"Skipping {iter_start} iterations to resume from checkpoint..."
            )
            for _ in range(iter_start):
                batch_loader.get_batch(train=True)
            logger.info(f"Resuming training from iteration {iter_start}")

        for itr in range(iter_start, ipe):
            itr_start_time = time.time()

            # Get batch and train
            obs, action, state, reward = batch_loader.get_batch(train=True)

            (loss, losses, total_stats), gpu_etime_ms = gpu_timer(
                lambda: train_step(
                    world_model=world_model,
                    obs=obs,
                    action=action,
                    state=state,
                    scheduler=scheduler,
                    wd_scheduler=wd_scheduler,
                    cfg=cfg,
                    dtype=dtype,
                    mixed_precision=mixed_precision,
                )
            )

            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            loss_meter.update(loss)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)

            # Initialize CSV logger once we have the columns
            if train_csv_logger is None:
                all_keys = sorted(set(list(losses.keys()) + list(total_stats.keys())))
                train_csv_logger_columns = [
                    "epoch",
                    "itr",
                    "loss",
                    "gpu-time(ms)",
                    "iter-time(ms)",
                ] + all_keys
                new_columns = [("%.5f", key) for key in all_keys]
                train_csv_logger = CSVLogger(
                    train_log_file,
                    ("%d", "epoch"),
                    ("%d", "itr"),
                    ("%.5f", "loss"),
                    ("%.2f", "gpu-time(ms)"),
                    ("%.2f", "iter-time(ms)"),
                    *new_columns,
                )

            # Light evaluation
            eval_losses, eval_total_stats, image_stats = {}, {}, {}
            if (
                itr % cfg.eval.light_eval_freq == cfg.eval.light_eval_freq - 1
                and val_unsupervised_loader is not None
            ):
                obs_val, action_val, state_val, reward_val = batch_loader.get_batch(
                    train=False
                )
                eval_losses, eval_total_stats, image_stats = validation_step(
                    world_model=world_model,
                    obs=obs_val,
                    action=action_val,
                    state=state_val,
                    lpips_model=lpips_model,
                    inverse_transform=inverse_transform,
                    cfg=cfg,
                    dtype=dtype,
                    mixed_precision=mixed_precision,
                )

            # Logging
            trainer.log(
                epoch,
                itr,
                losses,
                total_stats,
                eval_losses,
                eval_total_stats,
                image_stats,
            )

            # Log to CSV
            log_values = [epoch + 1, itr, loss, gpu_etime_ms, iter_elapsed_time_ms]
            for key in train_csv_logger_columns[5:]:
                if key in losses:
                    value = losses[key]
                    log_values.append(
                        value.item() if isinstance(value, torch.Tensor) else value
                    )
                elif key in total_stats:
                    value = total_stats[key]
                    log_values.append(
                        value.item() if isinstance(value, torch.Tensor) else value
                    )
                else:
                    log_values.append(0.0)
            train_csv_logger.log(*log_values)

            # Console logging
            if itr % LOG_FREQ == 0:
                logger.info(
                    f"[{epoch + 1}, {itr:5d}] "
                    f"[mem: {torch.cuda.max_memory_allocated() / 1024.0**2:.2e}] "
                    f"[gpu: {gpu_time_meter.avg:.1f} ms] "
                    f"[wall: {wall_time_meter.avg:.1f} ms]"
                )

            # Periodic checkpoint saving within epoch
            if itr > 0 and itr % SAVE_FREQ == 0 and rank == 0:
                logger.info(f"Saving checkpoint at epoch {epoch + 1}, iteration {itr}")
                save_checkpoint(
                    world_model,
                    optimizer,
                    scaler,
                    epoch,
                    latest_path,
                    rank,
                    cfg,
                    iteration=itr,
                )

            assert not np.isnan(loss), "loss is nan"

        logger.info(f"Avg. loss: {loss_meter.avg:.3f}")

        # Save checkpoint at end of epoch
        if epoch % CHECKPOINT_FREQ == 0 or epoch == cfg.optimization.num_epochs - 1:
            if rank == 0:
                save_checkpoint(
                    world_model,
                    optimizer,
                    scaler,
                    epoch + 1,
                    latest_path,
                    rank,
                    cfg,
                    iteration=0,  # Reset iteration count for new epoch
                )

                if cfg.save_every_freq > 0 and epoch % cfg.save_every_freq == 0:
                    save_every_file = f"{pref_tag}e{epoch}.pth.tar"
                    save_every_path = os.path.join(checkpoint_folder, save_every_file)
                    save_checkpoint(
                        world_model,
                        optimizer,
                        scaler,
                        epoch + 1,
                        save_every_path,
                        rank,
                        cfg,
                        iteration=0,
                    )

    logger.info("Training completed!")


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Main entry point with default configuration."""
    # Create default config matching the YAML file
    cfg = TrainConfig(
        folder=os.environ.get("JEPAWM_LOGS", "./logs") + "/train_wm2",
        seed=234,
        dtype="bfloat16",
        freeze_encoder=True,
        load_checkpoint=True,
        load_opt_scale_epoch=True,
        eval_freq=1,
        save_every_freq=1,
        data=DataConfig(
            datasets=["PushT"],
            seed=234,
            img_size=224,
            batch_size=8,
            num_workers=16,
            frameskip=5,
            action_skip=1,
            state_skip=1,
            num_hist=3,
            num_pred=1,
        ),
        data_aug=DataAugConfig(
            random_resize_scale=(1.777, 1.777),
            normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ),
        model=ModelConfig(
            grid_size=16,
            tubelet_size_enc=1,
            enc_type="dino",
            enc_version="dinov2_vits14",
            embed_dim=384,
            action_tokens=1,
            action_encoder_inpred=True,
            proprio_tokens=0,
            proprio_emb_dim=16,
            proprio_encoder_inpred=False,
            pred_depth=6,
            pred_embed_dim=384,
            pred_type="AdaLN",
            use_rope=True,
            rollout_steps=2,
            ctxt_window_train_rollout=3,
            local_window_time=3,
        ),
        loss=LossConfig(l2_loss_weight=1.0),
        optimization=OptimizationConfig(
            num_epochs=50,
            start_lr=5e-4,
            ref_lr=5e-4,
            final_lr=5e-4,
        ),
        eval=EvalConfig(
            do_data_traj_rollout_eval=True,
            data_traj_eval_rollout_steps=6,
            data_traj_eval_ctxt_window=3,
            light_eval_freq=300,
        ),
        logging=LoggingConfig(
            use_wandb=False,
            log_media_locally=True,
        ),
    )

    train(cfg)


if __name__ == "__main__":
    main()
