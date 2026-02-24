# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for JEPA world model.

This module provides dataclass configurations for the JEPA (Joint Embedding
Predictive Architecture) world model used for visual prediction and planning.

Example (API):
    >>> from physicalai.policies.jepa import ModelConfig, TrainingConfig
    >>> config = TrainingConfig(
    ...     img_size=224,
    ...     batch_size=16,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass

from physicalai.config import Config


@dataclass
class JEPAConfig(Config):
    """Base configuration for JEPA world model.

    Attributes:
        img_size: Input image size (square).
        enc_version: Vision encoder version (e.g., "dinov2_vits14").
        pred_depth: Number of transformer layers in the predictor.
        pred_embed_dim: Embedding dimension for the predictor.
        pred_num_heads: Number of attention heads in the predictor.
        use_proprio: Whether to use proprioceptive state information.
        proprio_emb_dim: Embedding dimension for proprioceptive features.
        proprio_dim: Dimension of proprioceptive input (e.g., 4 for x, y, vx, vy).
        action_dim: Dimension of action space (e.g., 2 for dx, dy).
        ctxt_window: Context window size for inference (sliding window).
        frameskip: Number of frames to skip (temporal downsampling).
        action_skip: Number of actions between predictions.
    """

    # Model architecture
    img_size: int = 224
    enc_version: str = "dinov2_vits14"
    pred_depth: int = 6
    pred_embed_dim: int = 384
    pred_num_heads: int = 16

    # Proprioception
    use_proprio: bool = True
    proprio_emb_dim: int = 16
    proprio_dim: int = 4

    # Action
    action_dim: int = 2

    # Temporal settings
    ctxt_window: int = 2
    frameskip: int = 5
    action_skip: int = 1


@dataclass
class JEPATrainingConfig(JEPAConfig):
    """Training configuration for JEPA world model.

    Extends ModelConfig with training-specific parameters including
    optimization, loss weighting

    Attributes:
        num_hist: Number of context frames for training.
        num_pred: Number of frames to predict.
        enc_type: Encoder type (e.g., "dino").
        pred_type: Predictor type ("AdaLN" or "dino_wm").
        action_emb_dim: Action embedding dimension.
        batch_size: Training batch size.
        num_epochs: Total number of training epochs.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        warmup_epochs: Number of warmup epochs for learning rate scheduler.
        clip_grad: Gradient clipping threshold.
        l2_loss_weight: Weight for L2 loss component.
        l1_loss_weight: Weight for L1 loss component.
        cos_loss_weight: Weight for cosine similarity loss component.
        rollout_steps: Number of rollout steps (1 = single-step, >1 = multi-step).
        rollout_stop_gradient: Whether to stop gradients through rollout steps.
        checkpoint_dir: Directory to save checkpoints.
        log_freq: Log every N batches.
        val_freq: Validate every N epochs.
        save_freq: Save checkpoint every N epochs.
    """

    # Data
    num_hist: int = 3
    num_pred: int = 1

    # Model
    enc_type: str = "dino"
    pred_type: str = "AdaLN"
    action_emb_dim: int = 10

    # # Training
    # batch_size: int = 8
    # num_epochs: int = 1000
    #
    # learning_rate: float = 5e-4
    # weight_decay: float = 1e-7
    # warmup_epochs: int = 2
    # clip_grad: float = 1.0

    # Loss weights
    l2_loss_weight: float = 1.0
    l1_loss_weight: float = 0.0
    cos_loss_weight: float = 0.0

    # Rollout training (multi-step prediction)
    rollout_steps: int = 1
    rollout_stop_gradient: bool = True

@dataclass
class JEPAInferenceConfig(JEPAConfig):
    """Evaluation configuration for JEPA world model.

    Extends ModelConfig with evaluation-specific parameters including checkpoint
    loading, planner settings, and environment configuration.

    Attributes:
        seed: Random seed for reproducibility.
        goal_horizon: Goal horizon (number of high-level steps for goal).
        planner_name: Planner algorithm name (e.g., "cem").
        iterations: Number of planner optimization iterations.
        num_samples: Number of action samples per iteration.
        num_elites: Number of elite samples to keep.
        horizon: Planner lookahead horizon.
        var_scale: Variance scale for sampling.
        num_act_stepped: Number of actions to step in environment.
        objective_type: Objective function type (e.g., "L2").
        alpha: Objective weighting factor.
        with_target: Whether to include target in environment.
        with_velocity: Whether to include velocity in state.
        max_steps_multiplier: Multiplier for max episode steps
            (max_steps = goal_horizon * frameskip * max_steps_multiplier).
    """

    # Meta
    seed: int = 1

    # Data
    goal_horizon: int = 6

    # Planner
    planner_name: str = "cem"
    iterations: int = 30
    num_samples: int = 300
    num_elites: int = 10
    horizon: int = 6
    var_scale: float = 1.0
    num_act_stepped: int = 6
    objective_type: str = "L2"
    alpha: float = 0.1

    # Environment
    with_target: bool = True
    with_velocity: bool = True
    max_steps_multiplier: int = 10  # max_steps = goal_horizon * frameskip * max_steps_multiplier
