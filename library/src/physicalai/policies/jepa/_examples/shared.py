#!/usr/bin/env python3
"""
Shared components for JEPA-WMS training and inference.

This module contains common utilities used by both inference.py and train_wm.py:
- Configuration (ModelConfig base class)
- Dataset loading (load_pusht_datasets)
- Model loading (load_model_hub, load_model_pretrained)
- Model creation (create_encoder, create_predictor, create_proprio_encoder, create_video_wm)
- Environment setup (PushTEnvWithEval, create_pusht_env)
- Preprocessor creation (create_preprocessor)
- Checkpoint saving (save_checkpoint)

All components are designed to match the original codebase exactly for reproducibility.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Optional


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ModelConfig:
    """
    Base configuration for JEPA world model.

    Contains shared settings for model architecture, data processing, and runtime.
    Inherited by TrainingConfig and EvalConfig for their specific needs.
    """

    # Paths
    data_path: str = ""

    # Model architecture
    img_size: int = 224
    enc_version: str = "dinov2_vits14"
    pred_depth: int = 6
    pred_embed_dim: int = 384
    pred_num_heads: int = 16

    # Proprioception
    use_proprio: bool = True
    proprio_emb_dim: int = 16
    proprio_dim: int = 4  # Push-T with velocity: (x, y, vx, vy)

    # Action
    action_dim: int = 2  # Push-T: (dx, dy)

    # Temporal settings
    ctxt_window: int = 2  # Context window size for inference (unroll sliding window)
    frameskip: int = 5  # Skip between frames (temporal downsampling)
    action_skip: int = 1  # Actions between predictions

    # Device
    device: str = "cuda:0"


# ============================================================================
# Dataset Loading Functions
# ============================================================================


def load_pusht_datasets(
    data_path: str,
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    img_size: int = 224,
    frameskip: int = 5,
    action_skip: int = 1,
    num_hist: int = 3,
    num_pred: int = 1,
    split_ratio: float = 0.9,
    seed: int = 42,
    return_train: bool = True,
    return_traj_dset: bool = False,
):
    """
    Load PushT datasets for training and/or inference.

    This is the unified dataset loading function used by both train_wm.py and inference.py.

    Args:
        data_path: Path to PushT dataset
        normalize_mean: Image normalization mean (default: ImageNet)
        normalize_std: Image normalization std (default: ImageNet)
        img_size: Image size
        frameskip: Number of environment steps between frames
        action_skip: Number of actions to skip
        num_hist: Number of history frames
        num_pred: Number of prediction frames
        split_ratio: Train/val split ratio
        seed: Random seed
        return_train: If True, return both train and val datasets; if False, return only val
        return_traj_dset: If True, return raw trajectory dataset instead of sliced dataset
                         (used by inference for sampling full trajectories)

    Returns:
        If return_train=True:
            train_dataset: Training dataset (TrajSlicerDataset or PushTDataset if return_traj_dset)
            val_dataset: Validation dataset (TrajSlicerDataset or PushTDataset if return_traj_dset)
            preprocessor: Data preprocessor
        If return_train=False:
            val_dataset: Validation dataset (TrajSlicerDataset or PushTDataset if return_traj_dset)
            preprocessor: Data preprocessor
    """
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import (
        make_inverse_transforms,
        make_transforms,
    )
    from app.plan_common.datasets.pusht_dset import load_pusht_slice_train_val

    normalize = [list(normalize_mean), list(normalize_std)]
    transform = make_transforms(
        img_size=img_size,
        normalize=normalize,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    inverse_transform = make_inverse_transforms(
        img_size=img_size,
        normalize=normalize,
    )

    # Load datasets using the codebase utility
    datasets, traj_dsets = load_pusht_slice_train_val(
        transform=transform,
        n_rollout=None,  # Use all rollouts
        data_path=data_path,
        normalize_action=True,
        split_ratio=split_ratio,
        num_hist=num_hist,
        num_pred=num_pred,
        num_frames_val=num_hist + num_pred,
        frameskip=frameskip,
        action_skip=action_skip,
        with_velocity=True,
        random_seed=seed,
        process_actions="concat",
    )

    # Create preprocessor using dataset statistics
    # Use the underlying trajectory dataset for statistics
    traj_dset = traj_dsets["train"]
    preprocessor = Preprocessor(
        action_mean=traj_dset.action_mean,
        action_std=traj_dset.action_std,
        state_mean=traj_dset.state_mean,
        state_std=traj_dset.state_std,
        proprio_mean=traj_dset.proprio_mean,
        proprio_std=traj_dset.proprio_std,
        transform=transform,
        inverse_transform=inverse_transform,
    )

    # Select which datasets to return
    if return_traj_dset:
        # Return raw trajectory datasets (for inference - sampling full trajectories)
        train_dataset = traj_dsets["train"]
        val_dataset = traj_dsets["valid"]
    else:
        # Return sliced datasets (for training)
        train_dataset = datasets["train"]
        val_dataset = datasets["valid"]

    if return_train:
        return train_dataset, val_dataset, preprocessor
    else:
        return val_dataset, preprocessor


# ============================================================================
# Normalization Constants
# ============================================================================

# ImageNet normalization (used by hub model)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Simple normalization (used by train_wm)
SIMPLE_MEAN = (0.5, 0.5, 0.5)
SIMPLE_STD = (0.5, 0.5, 0.5)


# ============================================================================
# Model Component Creation
# ============================================================================


def create_encoder(
    enc_version: str = "dinov2_vits14",
    device: str = "cuda:0",
    freeze: bool = True,
):
    """
    Create DINO encoder.

    Args:
        enc_version: DINO model version
        device: Device to load on
        freeze: Whether to freeze encoder parameters

    Returns:
        encoder: DinoEncoder instance
    """
    from app.plan_common.models.dino import DinoEncoder

    encoder = DinoEncoder(name=enc_version, feature_key="x_norm_patchtokens").to(device)
    if freeze:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
    return encoder


def create_predictor(
    img_size: int = 224,
    num_frames: int = 3,
    embed_dim: int = 384,
    pred_embed_dim: int = 384,
    pred_depth: int = 6,
    pred_num_heads: int = 16,
    internal_action_dim: int = 10,
    proprio_dim: int = 4,
    use_proprio: bool = True,
    proprio_emb_dim: int = 16,
    device: str = "cuda:0",
):
    """
    Create AdaLN predictor.

    Args:
        img_size: Image size
        num_frames: Number of frames (context + prediction)
        embed_dim: Encoder embedding dimension
        pred_embed_dim: Predictor embedding dimension
        pred_depth: Number of transformer layers
        pred_num_heads: Number of attention heads
        internal_action_dim: Action dimension (action_dim * frameskip)
        proprio_dim: Proprioceptive input dimension (raw, not embedded)
        use_proprio: Whether to use proprioceptive input
        proprio_emb_dim: Proprioceptive embedding dimension
        device: Device

    Returns:
        predictor: vit_predictor_AdaLN instance
    """
    from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN

    predictor = vit_predictor_AdaLN(
        img_size=img_size,
        patch_size=14,
        num_frames=num_frames,
        tubelet_size=1,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=pred_num_heads,
        action_dim=internal_action_dim,
        proprio_dim=proprio_emb_dim if use_proprio else 0,  # Use embedded dim
        use_proprio=use_proprio,
        action_encoder_inpred=True,
        proprio_encoder_inpred=False,  # Proprio encoder OUTSIDE predictor
        proprio_encoding="feature" if use_proprio else None,
        proprio_emb_dim=proprio_emb_dim if use_proprio else 0,
        use_rope=True,
        init_scale_factor_adaln=10,
    ).to(device)

    return predictor


def create_proprio_encoder(
    proprio_dim: int = 4,
    proprio_emb_dim: int = 16,
    num_frames: int = 3,
    device: str = "cuda:0",
):
    """
    Create proprioceptive encoder.

    Args:
        proprio_dim: Input proprioceptive dimension
        proprio_emb_dim: Output embedding dimension
        num_frames: Number of frames
        device: Device

    Returns:
        proprio_encoder: ProprioceptiveEmbedding instance
    """
    from app.plan_common.models.prop_embedding import ProprioceptiveEmbedding

    proprio_encoder = ProprioceptiveEmbedding(
        in_chans=proprio_dim,
        embed_dim=proprio_emb_dim,
        tokens_per_step=1,
        tubelet_size=1,
        num_frames=num_frames,
    ).to(device)

    return proprio_encoder


def create_video_wm(
    encoder,
    predictor,
    proprio_encoder=None,
    img_size: int = 224,
    internal_action_dim: int = 10,
    proprio_dim: int = 4,
    use_proprio: bool = True,
    frameskip: int = 5,
    device: str = "cuda:0",
    optimizer=None,
    clip_grad: float = None,
    cfgs_loss: dict = None,
):
    """
    Create VideoWM instance.

    Args:
        encoder: DINO encoder
        predictor: AdaLN predictor
        proprio_encoder: Proprioceptive encoder (optional)
        img_size: Image size
        internal_action_dim: Action dimension (action_dim * frameskip)
        proprio_dim: Proprioceptive dimension
        use_proprio: Whether to use proprioceptive input
        frameskip: Frameskip value
        device: Device
        optimizer: Optimizer (for training)
        clip_grad: Gradient clipping value
        cfgs_loss: Loss configuration dict

    Returns:
        video_wm: VideoWM instance
    """
    from app.vjepa_wm.video_wm import VideoWM

    grid_size = img_size // 14

    if cfgs_loss is None:
        cfgs_loss = {
            "l2_loss_weight": 1.0,
            "l1_loss_weight": 0.0,
            "cos_loss_weight": 0.0,
            "smooth_l1_loss_weight": 0.0,
            "proprio_loss": use_proprio,
        }

    video_wm = VideoWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=None,
        proprio_encoder=proprio_encoder,
        enc_type="dino",
        pred_type="AdaLN",
        grid_size=grid_size,
        tubelet_size_enc=1,
        img_size=img_size,
        action_dim=internal_action_dim,
        proprio_dim=proprio_dim,
        use_action=True,
        use_proprio=use_proprio,
        action_skip=1,
        frameskip=frameskip,
        action_conditioning="token",
        proprio_encoding="feature" if use_proprio else None,
        action_encoder_inpred=True,
        proprio_encoder_inpred=False,
        batchify_video=True,
        normalize_reps=False,
        device=torch.device(device) if isinstance(device, str) else device,
        optimizer=optimizer,
        scaler=None,
        clip_grad=clip_grad,
        mixed_precision=False,
        cfgs_loss=cfgs_loss,
        heads=[],
    )

    return video_wm


def create_preprocessor(
    img_size: int = 224,
    normalize_mean: Tuple[float, ...] = IMAGENET_MEAN,
    normalize_std: Tuple[float, ...] = IMAGENET_STD,
):
    """
    Create preprocessor with specified normalization.

    Args:
        img_size: Image size
        normalize_mean: Normalization mean
        normalize_std: Normalization std

    Returns:
        preprocessor: Preprocessor instance
    """
    from app.plan_common.datasets.pusht_dset import (
        ACTION_MEAN,
        ACTION_STD,
        STATE_MEAN,
        STATE_STD,
        PROPRIO_MEAN,
        PROPRIO_STD,
    )
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import (
        make_transforms,
        make_inverse_transforms,
    )

    normalize = [list(normalize_mean), list(normalize_std)]
    transform = make_transforms(
        img_size=img_size,
        normalize=normalize,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    inverse_transform = make_inverse_transforms(img_size=img_size, normalize=normalize)

    preprocessor = Preprocessor(
        action_mean=ACTION_MEAN,
        action_std=ACTION_STD,
        state_mean=STATE_MEAN,
        state_std=STATE_STD,
        proprio_mean=PROPRIO_MEAN,
        proprio_std=PROPRIO_STD,
        transform=transform,
        inverse_transform=inverse_transform,
    )

    return preprocessor


# ============================================================================
# Model Loading Functions
# ============================================================================


def load_model_hub(model_name: str = "jepa_wm_pusht", device: str = "cuda:0"):
    """
    Load model via PyTorch Hub.

    Args:
        model_name: Model name (e.g., "jepa_wm_pusht")
        device: Device to load on

    Returns:
        model: EncPredWM model
        preprocessor: Preprocessor instance
    """
    import hubconf

    model_fn = getattr(hubconf, model_name)
    model, preprocessor = model_fn(pretrained=True, device=device)
    return model, preprocessor


def load_model_pretrained(
    checkpoint_path: str,
    device: str = "cuda:0",
    img_size: int = 224,
    pred_depth: int = 6,
    pred_embed_dim: int = 384,
    pred_num_heads: int = 16,
    use_proprio: bool = True,
    proprio_emb_dim: int = 16,
    action_dim: int = 2,
    proprio_dim: int = 4,
    frameskip: int = 5,
    ctxt_window: int = 2,
    normalize_mean: Tuple[float, ...] = IMAGENET_MEAN,
    normalize_std: Tuple[float, ...] = IMAGENET_STD,
):
    """
    Load model from pretrained .pth.tar checkpoint (no embedded config).

    Args:
        checkpoint_path: Path to checkpoint
        device: Device
        Other args: Model architecture parameters

    Returns:
        model: EncPredWM model
        preprocessor: Preprocessor instance
    """
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded pretrained checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    internal_action_dim = action_dim * frameskip
    num_frames = ctxt_window + 1  # Context + 1 prediction

    # Create components using shared functions
    encoder = create_encoder(enc_version="dinov2_vits14", device=device, freeze=True)

    predictor = create_predictor(
        img_size=img_size,
        num_frames=num_frames,
        embed_dim=encoder.emb_dim,
        pred_embed_dim=pred_embed_dim,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        internal_action_dim=internal_action_dim,
        proprio_dim=proprio_dim,
        use_proprio=use_proprio,
        proprio_emb_dim=proprio_emb_dim,
        device=device,
    )

    # Load weights (handle module. prefix from DDP)
    state_dict = {
        k.replace("module.", ""): v for k, v in checkpoint["predictor"].items()
    }
    predictor.load_state_dict(state_dict)
    predictor.eval()

    proprio_encoder = None
    if use_proprio and "proprio_encoder" in checkpoint:
        proprio_encoder = create_proprio_encoder(
            proprio_dim=proprio_dim,
            proprio_emb_dim=proprio_emb_dim,
            num_frames=num_frames,
            device=device,
        )
        state_dict = {
            k.replace("module.", ""): v
            for k, v in checkpoint["proprio_encoder"].items()
        }
        proprio_encoder.load_state_dict(state_dict)
        proprio_encoder.eval()

    video_wm = create_video_wm(
        encoder=encoder,
        predictor=predictor,
        proprio_encoder=proprio_encoder,
        img_size=img_size,
        internal_action_dim=internal_action_dim,
        proprio_dim=proprio_dim,
        use_proprio=use_proprio,
        frameskip=frameskip,
        device=device,
    )
    video_wm.eval()

    # Create preprocessor with specified normalization
    preprocessor = create_preprocessor(
        img_size=img_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    # Wrap in EncPredWM
    model = EncPredWM(
        model=video_wm,
        action_dim=internal_action_dim,
        preprocessor=preprocessor,
        ctxt_window=ctxt_window,
        proprio_mode="predict_proprio",
    )
    model.eval()

    return model, preprocessor


# ============================================================================
# Environment Setup
# ============================================================================


class PushTEnvWithEval:
    """
    Thin wrapper around PushTEnv that adds eval_state() method.
    Preserves the original PushTEnv interface (unlike PushTWrapper which changes it).
    """

    def __init__(self, env):
        # Use object.__setattr__ to avoid triggering our __setattr__
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "action_space", env.action_space)
        object.__setattr__(self, "observation_space", env.observation_space)

    def __getattr__(self, name):
        # Forward all other attributes to the underlying env
        return getattr(self._env, name)

    def __setattr__(self, name, value):
        # Forward attribute writes to the underlying env (except our own attributes)
        if name in ("_env", "action_space", "observation_space"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._env, name, value)

    def eval_state(self, goal_state, cur_state):
        """
        Evaluate if the goal is reached.
        State format: [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        Returns dict with 'success' bool and 'state_dist' float.
        """
        # Position difference threshold: 20
        pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
        # Angle difference threshold: pi/9
        angle_diff = np.abs(goal_state[4] - cur_state[4])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        # success = pos_diff < 20 and angle_diff < np.pi / 9
        success = False
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            "success": success,
            "state_dist": state_dist,
        }

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def step(self, action):
        return self._env.step(action)

    def seed(self, seed):
        return self._env.seed(seed)

    def close(self):
        return self._env.close()

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)


def create_pusht_env(
    img_size: int = 224, with_velocity: bool = True, with_target: bool = True
):
    """Create PushT environment with eval_state() support."""
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

    base_env = PushTEnv(
        render_size=img_size,
        with_velocity=with_velocity,
        with_target=with_target,
        relative=True,
        action_scale=100,
    )
    return PushTEnvWithEval(base_env)


# ============================================================================
# Training Utilities
# ============================================================================


def create_world_model_for_training(
    img_size: int = 224,
    enc_version: str = "dinov2_vits14",
    pred_depth: int = 6,
    pred_embed_dim: int = 384,
    pred_num_heads: int = 16,
    use_proprio: bool = True,
    proprio_emb_dim: int = 16,
    action_dim: int = 2,
    proprio_dim: int = 4,
    frameskip: int = 5,
    num_hist: int = 3,
    num_pred: int = 1,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-7,
    clip_grad: float = 1.0,
    l2_loss_weight: float = 1.0,
    l1_loss_weight: float = 0.0,
    cos_loss_weight: float = 0.0,
    device: str = "cuda:0",
):
    """
    Create world model configured for training.

    This function creates all model components with trainable parameters
    and an optimizer.

    Args:
        Various architecture and training hyperparameters

    Returns:
        world_model: VideoWM instance ready for training
    """
    internal_action_dim = action_dim * frameskip
    num_frames = num_hist + num_pred

    # Create encoder (frozen)
    encoder = create_encoder(enc_version=enc_version, device=device, freeze=True)

    # Create predictor (trainable)
    predictor = create_predictor(
        img_size=img_size,
        num_frames=num_frames,
        embed_dim=encoder.emb_dim,
        pred_embed_dim=pred_embed_dim,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        internal_action_dim=internal_action_dim,
        proprio_dim=proprio_dim,
        use_proprio=use_proprio,
        proprio_emb_dim=proprio_emb_dim,
        device=device,
    )

    # Create proprio encoder if needed (trainable)
    proprio_encoder = None
    if use_proprio:
        proprio_encoder = create_proprio_encoder(
            proprio_dim=proprio_dim,
            proprio_emb_dim=proprio_emb_dim,
            num_frames=num_frames,
            device=device,
        )

    # Create optimizer for trainable parameters
    trainable_params = list(predictor.parameters())
    if proprio_encoder is not None:
        trainable_params += list(proprio_encoder.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Loss configuration
    cfgs_loss = {
        "l2_loss_weight": l2_loss_weight,
        "l1_loss_weight": l1_loss_weight,
        "cos_loss_weight": cos_loss_weight,
        "smooth_l1_loss_weight": 0.0,
        "proprio_loss": use_proprio,
    }

    # Create VideoWM
    video_wm = create_video_wm(
        encoder=encoder,
        predictor=predictor,
        proprio_encoder=proprio_encoder,
        img_size=img_size,
        internal_action_dim=internal_action_dim,
        proprio_dim=proprio_dim,
        use_proprio=use_proprio,
        frameskip=frameskip,
        device=device,
        optimizer=optimizer,
        clip_grad=clip_grad,
        cfgs_loss=cfgs_loss,
    )

    return video_wm


def save_checkpoint(
    world_model,
    epoch: int,
    checkpoint_dir: str,
    filename: str = None,
):
    """
    Save inference checkpoint in .pth.tar format.

    Args:
        world_model: VideoWM instance
        epoch: Current epoch
        checkpoint_dir: Directory to save checkpoint
        filename: Optional filename (defaults to epoch_XXX.pth.tar)
    """
    from pathlib import Path

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"epoch_{epoch:03d}.pth.tar"

    checkpoint = {
        "epoch": epoch,
        "predictor": world_model.predictor.state_dict(),
    }

    # Save proprio_encoder if it exists
    if world_model.proprio_encoder is not None:
        checkpoint["proprio_encoder"] = world_model.proprio_encoder.state_dict()

    path = checkpoint_dir / filename
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

    # Also save as latest
    latest_path = checkpoint_dir / "latest.pth.tar"
    torch.save(checkpoint, latest_path)
