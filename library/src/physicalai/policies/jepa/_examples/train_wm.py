#!/usr/bin/env python3
"""
Minimal training script for JEPA World Model on Push-T environment.

This script demonstrates how to:
1. Load the Push-T dataset (or generate online)
2. Initialize the world model (encoder, predictor, action/proprio encoders)
3. Run the training loop with proper loss computation
4. Save checkpoints in .pth.tar format (compatible with inference.py --checkpoint)
5. Validate and log progress

The world model learns to predict future latent states given:
- Current visual observation (encoded by frozen DINO encoder)
- Current proprioceptive state
- Action sequence

Loss is computed as L2 distance between predicted and target latent features.

Usage:
    # Basic training (uses online data generation by default)
    python train_wm.py

    # With custom settings
    python train_wm.py --epochs 20 --batch_size 16 --lr 1e-4

    # Load from disk dataset instead of online generation
    python train_wm.py --no_online --data_path /path/to/pusht_dataset
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# Import shared components
from shared import (
    ModelConfig,
    create_world_model_for_training,
    save_checkpoint,
    create_pusht_env,
    load_pusht_datasets,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# Import from the codebase
from app.plan_common.datasets.pusht_dset import (
    PushTDataset,
    ACTION_MEAN,
    ACTION_STD,
    STATE_MEAN,
    STATE_STD,
    PROPRIO_MEAN,
    PROPRIO_STD,
)
from app.plan_common.datasets.traj_dset import TrajSlicerDataset
from app.plan_common.datasets.transforms import make_transforms
from app.vjepa_wm.video_wm import VideoWM
from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TrainingConfig(ModelConfig):
    """Training configuration with sensible defaults for Push-T."""

    # Data (additional training-specific settings)
    num_hist: int = 3  # Number of context frames
    num_pred: int = 1  # Number of frames to predict
    split_ratio: float = 0.9  # Train/val split

    # Model (additional training-specific settings)
    enc_type: str = "dino"
    pred_type: str = "AdaLN"  # "AdaLN" or "dino_wm"
    action_emb_dim: int = 10  # Action embedding dimension

    # Training
    batch_size: int = 8
    num_epochs: int = 1000
    iters_per_epoch: Optional[int] = (
        None  # If set, limit batches per epoch (None = use all)
    )
    learning_rate: float = 5e-4
    weight_decay: float = 1e-7
    warmup_epochs: int = 2
    clip_grad: float = 1.0

    # Loss weights
    l2_loss_weight: float = 1.0
    l1_loss_weight: float = 0.0
    cos_loss_weight: float = 0.0

    # Rollout training (multi-step prediction)
    rollout_steps: int = 1  # 1 = single-step, >1 = multi-step rollout training
    rollout_stop_gradient: bool = True

    # Logging and checkpointing
    checkpoint_dir: str = "checkpoints"
    log_freq: int = 10  # Log every N batches
    val_freq: int = 1  # Validate every N epochs
    save_freq: int = 5  # Save checkpoint every N epochs

    # Online data generation
    use_online_data: bool = (
        True  # If True, generate data from environment instead of loading from disk
    )


# ============================================================================
# Online Data Generation from Environment
# ============================================================================


class PushTOnlineDataset(IterableDataset):
    """
    Online dataset that generates samples by running the PushT environment.

    Instead of loading pre-recorded trajectories from disk, this dataset
    creates random trajectories on-the-fly by:
    1. Resetting the environment to a random state
    2. Taking random actions for num_frames * frameskip steps
    3. Collecting observations, actions, and states

    This is useful for:
    - Training without needing pre-recorded data
    - Data augmentation through random exploration
    - Testing the world model on diverse scenarios

    Args:
        num_frames: Number of frames per sample (after frameskip)
        frameskip: Number of environment steps between frames
        action_skip: Number of actions to skip (for action aggregation)
        img_size: Target image size for observations
        normalize_action: Whether to normalize actions using dataset statistics
        with_velocity: Include velocity in proprioceptive state
        samples_per_epoch: Number of samples to generate per epoch
        seed: Random seed for reproducibility (None for random)
        transform: Image transform to apply
        process_actions: How to process actions ("concat" or "sum")
    """

    def __init__(
        self,
        num_frames: int = 4,
        frameskip: int = 5,
        action_skip: int = 1,
        img_size: int = 224,
        normalize_action: bool = True,
        with_velocity: bool = True,
        samples_per_epoch: int = 1000,
        seed: Optional[int] = None,
        transform=None,
        process_actions: str = "concat",
    ):
        super().__init__()
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.action_skip = action_skip
        self.img_size = img_size
        self.normalize_action = normalize_action
        self.with_velocity = with_velocity
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.transform = transform
        self.process_actions = process_actions

        # Action/state dimensions (Push-T specific)
        self.action_dim = 2  # (dx, dy)
        self.proprio_dim = 4 if with_velocity else 2
        self.state_dim = 7 if with_velocity else 5

        # Normalization statistics
        if normalize_action:
            self.action_mean = ACTION_MEAN
            self.action_std = ACTION_STD
            self.proprio_mean = PROPRIO_MEAN[: self.proprio_dim]
            self.proprio_std = PROPRIO_STD[: self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        # Compute effective action dimension after processing
        if self.frameskip >= self.action_skip:
            self.effective_action_dim = self.action_dim * (
                self.frameskip // self.action_skip
            )
        else:
            self.effective_action_dim = self.action_dim

    def _create_env(self):
        """Create a new PushT environment instance."""
        return PushTEnv(
            render_size=self.img_size,
            with_velocity=self.with_velocity,
            with_target=True,
            relative=True,
            action_scale=100,
        )

    def _generate_sample(self, env, rng):
        """
        Generate a single training sample by running the environment.

        Args:
            env: PushT environment instance
            rng: numpy random state

        Returns:
            obs: dict with 'visual' [T, C, H, W] and 'proprio' [T, proprio_dim]
            actions: [T, action_dim] tensor of actions
            states: [T, state_dim] tensor of states
            rewards: [T] tensor of rewards
        """
        # Total steps needed: num_frames * frameskip
        total_steps = self.num_frames * self.frameskip

        # Reset environment with random seed
        env.seed(rng.randint(0, 100000))
        obs, state = env.reset()

        # Storage for trajectory
        visuals = []
        proprios = []
        actions = []
        states = []
        rewards = []

        # Collect initial observation
        visuals.append(obs["visual"])
        proprios.append(obs["proprio"])
        states.append(state)

        # Run environment for total_steps
        for step in range(total_steps):
            # Generate random action (normalized range roughly -1 to 1)
            action = rng.randn(2).astype(np.float32) * 0.5

            # Step environment
            obs, reward, done, info = env.step(action)

            # Store data
            visuals.append(obs["visual"])
            proprios.append(obs["proprio"])
            actions.append(action)
            states.append(info["state"])
            rewards.append(reward)

            if done:
                # If done early, pad with zeros or reset
                # For simplicity, we'll just break and handle short sequences
                break

        # Handle case where we didn't get enough steps
        while len(actions) < total_steps:
            actions.append(np.zeros(2, dtype=np.float32))
            visuals.append(visuals[-1])
            proprios.append(proprios[-1])
            states.append(states[-1])
            rewards.append(0.0)

        # Convert to tensors and subsample by frameskip
        # Visual: [total_steps+1] -> sample every frameskip -> [num_frames]
        visual_indices = list(range(0, total_steps + 1, self.frameskip))[
            : self.num_frames
        ]
        visual_tensor = torch.stack(
            [
                torch.from_numpy(visuals[i]).permute(2, 0, 1).float() / 255.0
                for i in visual_indices
            ]
        )  # [T, C, H, W]

        # Apply transform if provided
        if self.transform is not None:
            visual_tensor = self.transform(visual_tensor)

        proprio_tensor = torch.stack(
            [torch.from_numpy(proprios[i]).float() for i in visual_indices]
        )  # [T, proprio_dim]

        state_tensor = torch.stack(
            [torch.from_numpy(np.array(states[i])).float() for i in visual_indices]
        )  # [T, state_dim]

        reward_tensor = torch.tensor(
            [
                rewards[i] if i < len(rewards) else 0.0
                for i in range(0, total_steps, self.frameskip)
            ][: self.num_frames],
            dtype=torch.float32,
        )  # [T]

        # Actions: subsample by action_skip and aggregate
        action_list = [torch.from_numpy(a).float() for a in actions]
        action_tensor = torch.stack(action_list)  # [total_steps, action_dim]

        # Apply action_skip subsampling
        action_tensor = action_tensor[
            :: self.action_skip
        ]  # [total_steps // action_skip, action_dim]

        # Normalize actions
        action_tensor = (action_tensor - self.action_mean) / self.action_std

        # Normalize proprio
        proprio_tensor = (proprio_tensor - self.proprio_mean) / self.proprio_std

        # Reshape actions to match TrajSlicerDataset behavior
        if self.frameskip >= self.action_skip:
            # Concatenate actions within each frame period
            # [total_steps // action_skip, action_dim] -> [num_frames, frameskip // action_skip * action_dim]
            actions_per_frame = self.frameskip // self.action_skip
            n_action_frames = action_tensor.shape[0] // actions_per_frame
            if n_action_frames >= self.num_frames:
                action_tensor = action_tensor[: self.num_frames * actions_per_frame]
                action_tensor = rearrange(
                    action_tensor, "(n f) d -> n (f d)", n=self.num_frames
                )  # [num_frames, frameskip // action_skip * action_dim]
            else:
                # Pad if needed
                pad_size = self.num_frames * actions_per_frame - action_tensor.shape[0]
                action_tensor = torch.cat(
                    [action_tensor, torch.zeros(pad_size, self.action_dim)]
                )
                action_tensor = rearrange(
                    action_tensor, "(n f) d -> n (f d)", n=self.num_frames
                )

        obs = {
            "visual": visual_tensor,
            "proprio": proprio_tensor,
        }

        return obs, action_tensor, state_tensor, reward_tensor

    def __iter__(self):
        """Iterate and yield samples."""
        # Create random state
        if self.seed is not None:
            rng = np.random.RandomState(self.seed)
        else:
            rng = np.random.RandomState()

        # Create environment
        env = self._create_env()

        try:
            for i in range(self.samples_per_epoch):
                yield self._generate_sample(env, rng)
        finally:
            env.close()

    def __len__(self):
        """Return the number of samples per epoch."""
        return self.samples_per_epoch


# ============================================================================
# Data Loading
# ============================================================================


def create_online_dataloaders(config: TrainingConfig):
    """
    Create train and validation dataloaders that generate data online from the PushT environment.

    Instead of loading pre-recorded data from disk, this creates IterableDatasets
    that generate random trajectories by running the environment.

    Args:
        config: Training configuration

    Returns:
        train_loader: DataLoader for training (generates random trajectories)
        val_loader: DataLoader for validation (generates random trajectories with fixed seed)
        dataset_info: Dict with action/proprio dimensions and normalization stats
    """
    # Create image transforms
    transform = make_transforms(
        img_size=config.img_size,
        normalize=[
            list(IMAGENET_MEAN),
            list(IMAGENET_STD),
        ],  # ImageNet normalization (matches pretrained models)
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )

    num_frames = config.num_hist + config.num_pred

    # Determine samples per epoch from iters_per_epoch (default 1000 if not set)
    samples_per_epoch = (config.iters_per_epoch or 1000) * config.batch_size

    # Create online datasets
    # Training: random seed for variety
    train_dataset = PushTOnlineDataset(
        num_frames=num_frames,
        frameskip=config.frameskip,
        action_skip=config.action_skip,
        img_size=config.img_size,
        normalize_action=True,
        with_velocity=True,
        samples_per_epoch=samples_per_epoch,
        seed=None,  # Random seed for training
        transform=transform,
    )

    # Validation: fixed seed for reproducibility
    val_dataset = PushTOnlineDataset(
        num_frames=num_frames,
        frameskip=config.frameskip,
        action_skip=config.action_skip,
        img_size=config.img_size,
        normalize_action=True,
        with_velocity=True,
        samples_per_epoch=max(
            100, samples_per_epoch // 10
        ),  # 10% of train for validation
        seed=42,  # Fixed seed for reproducible validation
        transform=transform,
    )

    print(f"Online train dataset: {len(train_dataset)} samples per epoch")
    print(f"Online val dataset: {len(val_dataset)} samples per epoch")

    # Create data loaders (single-process, no workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # Dataset info for model initialization
    dataset_info = {
        "action_dim": config.action_dim,
        "proprio_dim": config.proprio_dim,
        "action_mean": ACTION_MEAN,
        "action_std": ACTION_STD,
        "proprio_mean": PROPRIO_MEAN,
        "proprio_std": PROPRIO_STD,
    }

    return train_loader, val_loader, dataset_info


def create_dataloaders(config: TrainingConfig):
    """
    Create train and validation dataloaders for Push-T.

    The data pipeline:
    1. Load PushTDataset (full trajectories)
    2. Wrap with TrajSlicerDataset to create fixed-length slices
    3. Apply image transforms (resize, normalize)
    4. Create DataLoaders with proper batching

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        dataset_info: Dict with action/proprio dimensions and normalization stats
    """
    # Load datasets using the shared utility
    train_dataset, val_dataset, preprocessor = load_pusht_datasets(
        data_path=config.data_path,
        normalize_mean=IMAGENET_MEAN,
        normalize_std=IMAGENET_STD,
        img_size=config.img_size,
        frameskip=config.frameskip,
        action_skip=config.action_skip,
        num_hist=config.num_hist,
        num_pred=config.num_pred,
        split_ratio=config.split_ratio,
        return_train=True,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # Dataset info for model initialization
    dataset_info = {
        "action_dim": config.action_dim,
        "proprio_dim": config.proprio_dim,
        "action_mean": ACTION_MEAN,
        "action_std": ACTION_STD,
        "proprio_mean": PROPRIO_MEAN,
        "proprio_std": PROPRIO_STD,
    }

    return train_loader, val_loader, dataset_info


# ============================================================================
# Model Initialization
# ============================================================================


def create_world_model(config: TrainingConfig, dataset_info: dict):
    """
    Initialize the world model components using shared module.

    Returns:
        world_model: VideoWM instance ready for training
    """
    print("Initializing world model using shared components...")

    world_model = create_world_model_for_training(
        img_size=config.img_size,
        enc_version=config.enc_version,
        pred_depth=config.pred_depth,
        pred_embed_dim=config.pred_embed_dim,
        pred_num_heads=config.pred_num_heads,
        use_proprio=config.use_proprio,
        proprio_emb_dim=config.proprio_emb_dim,
        action_dim=config.action_dim,
        proprio_dim=config.proprio_dim,
        frameskip=config.frameskip,
        num_hist=config.num_hist,
        num_pred=config.num_pred,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        clip_grad=config.clip_grad,
        l2_loss_weight=config.l2_loss_weight,
        l1_loss_weight=config.l1_loss_weight,
        cos_loss_weight=config.cos_loss_weight,
        device=config.device,
    )

    # Print model info
    pred_params = sum(
        p.numel() for p in world_model.predictor.parameters() if p.requires_grad
    )
    print(
        f"Predictor: {type(world_model.predictor).__name__} ({pred_params:,} trainable params)"
    )

    if world_model.proprio_encoder is not None:
        prop_params = sum(
            p.numel()
            for p in world_model.proprio_encoder.parameters()
            if p.requires_grad
        )
        print(
            f"Proprio encoder: {type(world_model.proprio_encoder).__name__} ({prop_params:,} trainable params)"
        )

    return world_model


# ============================================================================
# Training Step
# ============================================================================


def train_step(world_model: VideoWM, batch, config: TrainingConfig):
    """
    Perform a single training step.

    The training process:
    1. Encode visual observations with frozen encoder
    2. Encode actions (inside predictor for AdaLN)
    3. Forward pass through predictor to get predicted features
    4. Compute L2 loss between predictions and targets
    5. Backward pass with gradient clipping

    Args:
        world_model: VideoWM instance
        batch: Tuple of (obs, action, state, reward)
        config: Training configuration

    Returns:
        loss_dict: Dictionary of loss values
    """
    device = world_model.device

    # Unpack batch
    # TrajSlicerDataset returns (obs, act, state, reward)
    # obs: dict with 'visual' [B, T, C, H, W] and 'proprio' [B, T, proprio_dim]
    # action: [B, T, action_dim]
    obs, action, state, reward = batch

    # Move to device
    visual = obs["visual"].to(device)  # [B, T, C, H, W]
    proprio = (
        obs["proprio"].to(device) if config.use_proprio else None
    )  # [B, T, proprio_dim]
    action = action.to(device)  # [B, T, action_dim]

    B, T, C, H, W = visual.shape

    # Reshape actions: [B, T, action_dim] -> [B, T, internal_action_dim]
    # The model expects actions concatenated over frameskip
    # For Push-T: action_dim=2, frameskip=5 -> internal_action_dim=10
    internal_action_dim = config.action_dim * config.frameskip
    if action.shape[-1] != internal_action_dim:
        # Repeat last action dimension to match expected shape
        # This is a simplification - in practice, actions should be properly aggregated
        action = action.repeat(1, 1, config.frameskip)

    # Zero gradients
    world_model.optimizer.zero_grad()

    # Forward pass
    # Encode observations
    # video_features: [B, T, V, H, W, D] where V=1 (single view), H=W=grid_size, D=embed_dim
    video_features, proprio_features, action_features = world_model.encode(
        obs={"visual": visual, "proprio": proprio},
        a=action,
    )

    # Forward prediction
    # pred_video_features: [B, T, V, H, W, D]
    pred_video_features, pred_action_features, pred_proprio_features = (
        world_model.forward_pred(
            video_features=video_features,
            action_features=action_features,
            proprio_features=proprio_features,
        )
    )

    # Compute loss (shift=1 means we compare pred[t] with target[t+1])
    loss_dict = world_model.compute_loss(
        pred_video_features=pred_video_features,
        pred_proprio_features=pred_proprio_features,
        video_features=video_features,
        proprio_features=proprio_features,
        shift=1,
    )

    loss = loss_dict["loss"]

    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(world_model.predictor.parameters(), config.clip_grad)
    world_model.optimizer.step()

    # Convert losses to Python floats
    return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}


# ============================================================================
# Validation
# ============================================================================


@torch.no_grad()
def validate(world_model: VideoWM, val_loader: DataLoader, config: TrainingConfig):
    """
    Run validation and compute average losses.

    Args:
        world_model: VideoWM instance
        val_loader: Validation data loader
        config: Training configuration

    Returns:
        avg_losses: Dictionary of average loss values
    """
    device = world_model.device
    world_model.predictor.eval()

    total_losses = {}
    num_batches = 0

    for batch in val_loader:
        obs, action, state, reward = batch

        visual = obs["visual"].to(device)
        proprio = obs["proprio"].to(device) if config.use_proprio else None
        action = action.to(device)

        # Reshape actions if needed
        internal_action_dim = config.action_dim * config.frameskip
        if action.shape[-1] != internal_action_dim:
            action = action.repeat(1, 1, config.frameskip)

        video_features, proprio_features, action_features = world_model.encode(
            obs={"visual": visual, "proprio": proprio},
            a=action,
        )

        pred_video_features, pred_action_features, pred_proprio_features = (
            world_model.forward_pred(
                video_features=video_features,
                action_features=action_features,
                proprio_features=proprio_features,
            )
        )

        loss_dict = world_model.compute_loss(
            pred_video_features=pred_video_features,
            pred_proprio_features=pred_proprio_features,
            video_features=video_features,
            proprio_features=proprio_features,
            shift=1,
        )

        # Accumulate losses
        for k, v in loss_dict.items():
            val = v.item() if torch.is_tensor(v) else v
            total_losses[k] = total_losses.get(k, 0.0) + val

        num_batches += 1

    world_model.predictor.train()

    # Compute averages
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


# ============================================================================
# Checkpoint Management (using shared module with local wrappers)
# ============================================================================


def save_checkpoint_local(
    world_model: VideoWM,
    epoch: int,
    config: TrainingConfig,
    filename: str = None,
):
    """Save training checkpoint and config using shared module."""
    import yaml
    from pathlib import Path
    from dataclasses import asdict

    save_checkpoint(
        world_model=world_model,
        epoch=epoch,
        checkpoint_dir=config.checkpoint_dir,
        filename=filename,
    )

    # Save config as YAML (only once, on first save)
    config_path = Path(config.checkpoint_dir) / "config.yaml"
    if not config_path.exists():
        with open(config_path, "w") as f:
            yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
        print(f"Saved config to {config_path}")


# ============================================================================
# Learning Rate Scheduler
# ============================================================================


def create_scheduler(optimizer, config: TrainingConfig, steps_per_epoch: int):
    """Create learning rate scheduler with warmup."""
    # Ensure at least 1 step per epoch to avoid division by zero
    steps_per_epoch = max(1, steps_per_epoch)
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        elif total_steps > warmup_steps:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
        else:
            # No decay phase, just return 1.0
            return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


# ============================================================================
# Main Training Loop
# ============================================================================


def train(config: TrainingConfig):
    """
    Main training function.

    Training loop:
    1. Create dataloaders
    2. Initialize model
    3. Optionally resume from checkpoint
    4. For each epoch:
       - Train on all batches
       - Validate periodically
       - Save checkpoints
       - Log progress
    """
    print("=" * 60)
    print("JEPA World Model Training - Push-T")
    print("=" * 60)
    print(f"Config: {config}")
    print()

    # Set device
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Create data loaders
    print("\nLoading data...")
    if config.use_online_data:
        print("Using ONLINE data generation from PushT environment")
        train_loader, val_loader, dataset_info = create_online_dataloaders(config)
    else:
        print("Using DISK-based data from pre-recorded trajectories")
        train_loader, val_loader, dataset_info = create_dataloaders(config)

    # Determine effective steps per epoch (for scheduler)
    steps_per_epoch = config.iters_per_epoch or len(train_loader)
    print(f"Steps per epoch: {steps_per_epoch}")

    # Create model
    print("\nInitializing model...")
    world_model = create_world_model(config, dataset_info)

    # Create scheduler
    scheduler = create_scheduler(world_model.optimizer, config, steps_per_epoch)

    # Training loop
    print("\nStarting training...")
    print("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(config.num_epochs):
        world_model.predictor.train()

        epoch_losses = {}
        num_batches = 0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            total=steps_per_epoch,
        )

        for batch_idx, batch in enumerate(pbar):
            # Check if we've reached the iteration limit
            if (
                config.iters_per_epoch is not None
                and batch_idx >= config.iters_per_epoch
            ):
                break

            # Training step
            loss_dict = train_step(world_model, batch, config)
            num_batches += 1

            # Update scheduler
            scheduler.step()

            # Accumulate losses
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            # Update progress bar
            if batch_idx % config.log_freq == 0:
                pbar.set_postfix(
                    loss=f"{loss_dict['loss']:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        # Compute epoch averages (use actual batch count, not len(train_loader))
        avg_epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        print(f"\nEpoch {epoch + 1} - Train Loss: {avg_epoch_losses['loss']:.4f}")

        # Validation (always uses all batches for deterministic results)
        if (epoch + 1) % config.val_freq == 0:
            print("Running validation...")
            val_losses = validate(world_model, val_loader, config)
            print(f"Epoch {epoch + 1} - Val Loss: {val_losses['loss']:.4f}")

            if val_losses["loss"] < best_val_loss:
                best_val_loss = val_losses["loss"]
                best_val_epoch = epoch + 1
                print(
                    f"  New best! Saving best.pth.tar (epoch {best_val_epoch}, loss {best_val_loss:.4f})"
                )
                save_checkpoint_local(
                    world_model,
                    epoch + 1,
                    config,
                    "best.pth.tar",
                )

        # Save checkpoint
        if (epoch + 1) % config.save_freq == 0:
            save_checkpoint_local(
                world_model,
                epoch + 1,
                config,
            )

    # Final save
    save_checkpoint_local(
        world_model,
        config.num_epochs,
        config,
        "final.pth.tar",
    )
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_val_epoch}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train JEPA World Model on Push-T",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/kdijkstr/jepa/datasets/pusht_noise",
        help="Path to Push-T dataset",
    )
    parser.add_argument("--img_size", type=int, default=224, help="Image size")

    # Training
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument(
        "--iters_per_epoch",
        type=int,
        default=None,
        help="Max iterations (batches) per epoch. Default: use all batches",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    # Model
    parser.add_argument("--pred_depth", type=int, default=6, help="Predictor depth")
    parser.add_argument(
        "--no_proprio",
        action="store_true",
        help="Disable proprioceptive input (enabled by default)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory",
    )

    # Device
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    # Online data generation (enabled by default)
    parser.add_argument(
        "--no_online",
        action="store_true",
        help="Disable online data generation, load from disk instead (online is default)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create config from args
    config = TrainingConfig(
        data_path=args.data_path,
        img_size=args.img_size,
        num_epochs=args.epochs,
        iters_per_epoch=args.iters_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        pred_depth=args.pred_depth,
        use_proprio=not args.no_proprio,  # Proprio enabled by default
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        use_online_data=not args.no_online,  # Online data enabled by default
    )

    train(config)


if __name__ == "__main__":
    main()
