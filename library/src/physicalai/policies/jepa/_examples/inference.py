#!/usr/bin/env python3
"""
Minimal script for JEPA-WMs inference and planning on PushT.

This script provides a standalone evaluation that matches the original
evals/simu_env_planning pipeline exactly, including:
- Identical transforms and normalization (ImageNet stats)
- Same dataset sampling with reproducible seeds
- Same planner configuration and execution

Two model loading options are supported:
A) PyTorch Hub (downloads pretrained weights automatically)
B) Pretrained .pth.tar checkpoint

Usage:
    python inference.py --checkpoint /path/to/checkpoint.pth.tar --episode 0
    python inference.py --hub jepa_wm_pusht --episode 0
"""

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Optional, Tuple, List

import numpy as np
import torch
from einops import rearrange
from tensordict import TensorDict
from tqdm import tqdm

# Import shared components
from shared import (
    ModelConfig,
    load_pusht_datasets,
    load_model_hub,
    load_model_pretrained,
    PushTEnvWithEval,
    create_pusht_env,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SIMPLE_MEAN,
    SIMPLE_STD,
)


# ============================================================================
# Configuration (matches original config file)
# ============================================================================


@dataclass
class EvalConfig(ModelConfig):
    """Evaluation configuration matching the original YAML config."""

    # Paths (additional eval-specific)
    checkpoint: str = ""

    # Meta
    seed: int = 1
    episode: int = 0  # Episode index to evaluate

    # Data (additional eval-specific)
    goal_horizon: int = 6  # Goal horizon (number of high-level steps for goal)

    # Planner (matches original config exactly)
    planner_name: str = "cem"
    iterations: int = 30
    num_samples: int = 300
    num_elites: int = 10
    horizon: int = 6  # Planner lookahead horizon
    var_scale: float = 1.0
    num_act_stepped: int = 6
    objective_type: str = "L2"
    alpha: float = 0.1  # For objective weighting

    # Environment
    with_target: bool = True
    with_velocity: bool = True
    max_steps_multiplier: int = (
        10  # max_steps = goal_horizon * frameskip * max_steps_multiplier
    )

    # Normalization (ImageNet stats - matches original config)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Output
    output_dir: str = "output"
    save_video: bool = True


# ============================================================================
# Model Loading Functions (config-based loading for original eval.py compat)
# ============================================================================


# ============================================================================
# Dataset Sampling (matches original exactly)
# ============================================================================


def sample_trajectory_from_dataset(
    dset,
    traj_len: int,
    generator: torch.Generator,
) -> Tuple[dict, np.ndarray, torch.Tensor, dict]:
    """
    Sample a trajectory segment from the dataset.

    Matches the original sample_traj_segment_from_dset() in plan_evaluator.py.

    Args:
        dset: Trajectory dataset
        traj_len: Required trajectory length
        generator: Random generator for reproducibility

    Returns:
        obs: Dict with 'visual' and 'proprio' tensors
        states: State array [traj_len, state_dim]
        actions: Action tensor [traj_len-1, action_dim]
        env_info: Dict with environment info (e.g., 'shape' for PushT)
    """
    # Find valid trajectory (long enough)
    max_offset = -1
    while max_offset < 0:
        traj_id = torch.randint(
            low=0, high=len(dset), size=(1,), generator=generator
        ).item()
        obs, act, state, reward, e_info = dset[traj_id]
        max_offset = obs["visual"].shape[0] - traj_len

    state = state.numpy()
    offset = torch.randint(
        low=0, high=max_offset + 1, size=(1,), generator=generator
    ).item()

    print(
        f"  Sampled trajectory: traj_id={traj_id}, offset={offset}, traj_len={traj_len}"
    )

    obs = {key: arr[offset : offset + traj_len] for key, arr in obs.items()}
    state = state[offset : offset + traj_len]
    act = act[offset : offset + traj_len - 1]  # Actions between frames

    return obs, state, act, e_info


# ============================================================================
# Planning (matches original GC_Agent)
# ============================================================================


def create_planner(model, cfg: EvalConfig, generator: torch.Generator):
    """
    Create CEM planner matching original GC_Agent setup.

    Args:
        model: EncPredWM model
        cfg: Evaluation config
        generator: GPU random generator for reproducibility

    Returns:
        planner: CEMPlanner instance
    """
    from evals.simu_env_planning.planning.planning.planner import CEMPlanner

    planner = CEMPlanner(
        unroll=model.unroll,
        action_dim=model.action_dim,
        iterations=cfg.iterations,
        num_samples=cfg.num_samples,
        num_elites=cfg.num_elites,
        horizon=cfg.horizon,
        var_scale=cfg.var_scale,
        num_act_stepped=cfg.num_act_stepped,
        local_generator=generator,
        decode_unroll=model.decode_unroll,
        decode_each_iteration=False,
    )
    return planner


def set_goal(model, planner, goal_obs: TensorDict, cfg: EvalConfig):
    """
    Set planning goal (matches GC_Agent.set_goal()).

    Args:
        model: EncPredWM model
        planner: CEMPlanner
        goal_obs: Goal observation TensorDict
        cfg: Config
    """
    from evals.simu_env_planning.planning.planning.objectives import (
        ReprTargetDistMPCObjective,
        ReprTargetDistL1MPCObjective,
        ReprTargetCosMPCObjective,
    )
    from omegaconf import OmegaConf

    # Encode goal
    goal_state = goal_obs.unsqueeze(0).to(model.device)
    goal_enc_full = model.encode(goal_state, act=False).detach()

    # Take only the last timestep for the target (the objective expects a single goal state)
    # goal_enc_full["visual"] has shape [B, T, V, H, W, D]
    # goal_enc_full["proprio"] has shape [B, T, proprio_tokens, D]
    goal_enc = TensorDict(
        {
            "visual": goal_enc_full["visual"][:, -1:],  # [B, 1, V, H, W, D]
            "proprio": goal_enc_full["proprio"][:, -1:],  # [B, 1, proprio_tokens, D]
        }
    )

    # Create objective
    obj_cfg = OmegaConf.create(
        {
            "planner": {
                "planning_objective": {
                    "objective_type": cfg.objective_type,
                    "alpha": cfg.alpha,
                }
            }
        }
    )

    if cfg.objective_type == "L2":
        objective = ReprTargetDistMPCObjective(
            obj_cfg, target_enc=goal_enc, alpha=cfg.alpha
        )
    elif cfg.objective_type == "L1":
        objective = ReprTargetDistL1MPCObjective(
            obj_cfg, target_enc=goal_enc, alpha=cfg.alpha
        )
    elif cfg.objective_type == "repr_sim":
        objective = ReprTargetCosMPCObjective(
            obj_cfg, target_enc=goal_enc, alpha=cfg.alpha
        )
    else:
        raise ValueError(f"Unknown objective: {cfg.objective_type}")

    planner.set_objective(objective)
    return goal_enc


def plan_step(model, planner, obs: TensorDict, steps_left: int) -> torch.Tensor:
    """
    Execute one planning step (matches GC_Agent.act()).

    Args:
        model: EncPredWM model
        planner: CEMPlanner
        obs: Current observation TensorDict
        steps_left: Steps remaining in episode

    Returns:
        actions: Planned actions [num_act_stepped, action_dim]
    """
    obs_batch = obs.to(model.device, non_blocking=True).unsqueeze(0)
    z = model.encode(obs_batch, act=True)
    planning_result = planner.plan(z, steps_left=steps_left)
    return planning_result.actions.cpu(), planning_result


# ============================================================================
# Evaluation Loop (matches original plan_evaluator.py)
# ============================================================================


def make_tensordict(obs: dict, info: dict) -> TensorDict:
    """Convert observation dict to TensorDict (single frame, no time dim)."""
    return TensorDict(
        {
            "visual": torch.as_tensor(obs["visual"], dtype=torch.uint8),
            "proprio": torch.as_tensor(obs["proprio"], dtype=torch.float32),
        }
    )


class ObservationBuffer:
    """Buffer to maintain temporal context for observations."""

    def __init__(self, num_frames: int, num_proprios: int):
        self.num_frames = num_frames
        self.num_proprios = num_proprios
        self._frames: List[torch.Tensor] = []
        self._proprios: List[torch.Tensor] = []

    def reset(self):
        """Clear the buffer."""
        self._frames = []
        self._proprios = []

    def add(self, obs: dict, info: dict):
        """Add observation to buffer."""
        visual = obs["visual"]
        # Convert HWC to CHW format if needed
        if visual.ndim == 3 and visual.shape[-1] == 3:
            visual = np.transpose(visual, (2, 0, 1))  # HWC -> CHW
        frame = torch.as_tensor(visual, dtype=torch.uint8)
        proprio = torch.as_tensor(obs["proprio"], dtype=torch.float32)

        self._frames.append(frame)
        self._proprios.append(proprio)

        # Keep only the most recent frames/proprios
        if len(self._frames) > self.num_frames:
            self._frames = self._frames[-self.num_frames :]
        if len(self._proprios) > self.num_proprios:
            self._proprios = self._proprios[-self.num_proprios :]

    def get(self) -> TensorDict:
        """Get buffered observation with time dimension [T, C, H, W] and [T, D]."""
        # Pad if not enough frames yet
        while len(self._frames) < self.num_frames:
            self._frames.insert(0, self._frames[0].clone())
        while len(self._proprios) < self.num_proprios:
            self._proprios.insert(0, self._proprios[0].clone())

        return TensorDict(
            {
                "visual": torch.stack(self._frames[-self.num_frames :]),  # [T, C, H, W]
                "proprio": torch.stack(self._proprios[-self.num_proprios :]),  # [T, D]
            }
        )


def run_evaluation(
    model,
    preprocessor,
    dset,
    cfg: EvalConfig,
    generator: torch.Generator,
    gpu_generator: torch.Generator,
):
    """
    Run evaluation episode matching original pipeline.

    Args:
        model: EncPredWM model
        preprocessor: Preprocessor
        dset: Trajectory dataset
        cfg: Evaluation config
        generator: CPU random generator
        gpu_generator: GPU random generator

    Returns:
        results: Dict with metrics and frames
    """
    device = torch.device(cfg.device)

    # Compute episode seed (matches original)
    local_seed = cfg.seed + cfg.episode * cfg.horizon * 1000
    ep_seed = (local_seed * local_seed + cfg.episode * local_seed) % (2**32 - 2)
    print(f"Episode {cfg.episode}: ep_seed={ep_seed}, local_seed={local_seed}")

    # Seed generators
    generator.manual_seed(local_seed)
    gpu_generator.manual_seed(local_seed)

    # Sample trajectory from dataset
    traj_len = cfg.frameskip * cfg.goal_horizon + 1
    print(f"Sampling trajectory of length {traj_len} from dataset...")
    obs_traj, states_traj, actions_traj, env_info = sample_trajectory_from_dataset(
        dset, traj_len, generator
    )

    # Create environment
    env = create_pusht_env(cfg.img_size, cfg.with_velocity, cfg.with_target)

    # Set environment shape to match the dataset trajectory
    if "shape" in env_info:
        env.shape = env_info["shape"]
        print(f"  Set environment shape to: {env_info['shape']}")

    # Get init and goal states
    init_state = states_traj[0]
    goal_state = states_traj[-1]

    # Replay expert actions to get ground truth trajectory
    print("Replaying expert actions in environment...")
    exec_actions = preprocessor.denormalize_actions(actions_traj)

    env.seed(ep_seed)
    obs, state = env.reset()
    env.reset_to_state = init_state
    obs, state = env.reset()

    # Verify initial state was set correctly
    # Note: PushTEnv.reset() returns (observation, state), not (observation, info)
    actual_init_state = state
    print(f"  Requested init_state: {init_state}")
    print(f"  Actual init_state:    {actual_init_state}")
    if not np.allclose(actual_init_state, init_state, atol=1e-2):
        print(f"  WARNING: Initial state mismatch!")

    # Create observation buffer for temporal context
    # Use ctxt_window for both frames and proprios (matching original pipeline)
    obs_buffer = ObservationBuffer(
        num_frames=cfg.ctxt_window, num_proprios=cfg.ctxt_window
    )

    expert_frames = [obs["visual"].copy()]
    # Warm up buffer with initial observation (repeated)
    # Create info dict for ObservationBuffer (it expects info with 'proprio')
    info = {"state": state, "proprio": obs["proprio"]}
    for _ in range(cfg.ctxt_window):
        obs_buffer.add(obs, info)

    for i, action in enumerate(exec_actions):
        obs, reward, done, info = env.step(action.numpy())
        # Record at frameskip intervals (matching agent recording)
        if (i + 1) % cfg.frameskip == 0:
            expert_frames.append(obs["visual"].copy())
        obs_buffer.add(obs, info)
        if done:
            break

    # Get goal observation (buffered observation at end of expert trajectory)
    goal_obs = obs_buffer.get()

    # Verify expert replay correctness: final state should match goal_state from dataset
    replay_final_state = info["state"]
    print(f"  Number of actions executed: {len(exec_actions)}")
    print(f"  Number of expert_frames: {len(expert_frames)}")
    print(f"  states_traj shape: {states_traj.shape}")
    print(f"  actions_traj shape: {actions_traj.shape}")
    if not np.allclose(replay_final_state, goal_state, atol=1e-3):
        print(f"WARNING: Expert replay mismatch!")
        print(f"  Expected goal_state (states_traj[-1]): {goal_state}")
        print(f"  Replay final_state:  {replay_final_state}")
        print(f"  Difference: {np.abs(replay_final_state - goal_state)}")
        # Also check intermediate states
        print(f"  states_traj[0]: {states_traj[0]}")
        print(f"  states_traj[1]: {states_traj[1]}")
        print(f"  states_traj[-2]: {states_traj[-2]}")

    print(f"  Init state: agent=({init_state[0]:.1f}, {init_state[1]:.1f})")
    print(f"  Goal state: agent=({goal_state[0]:.1f}, {goal_state[1]:.1f})")

    # Reset environment to initial state for agent execution
    env.reset_to_state = init_state
    obs, info = env.reset()

    # Reset buffer and warm up with initial observation
    obs_buffer.reset()
    for _ in range(cfg.ctxt_window):
        obs_buffer.add(obs, info)

    # Create planner and set goal
    planner = create_planner(model, cfg, gpu_generator)
    goal_enc = set_goal(model, planner, goal_obs, cfg)

    # Run agent
    print("Running agent planning loop...")
    agent_frames = [obs["visual"].copy()]

    max_steps = (
        cfg.goal_horizon * cfg.frameskip * cfg.max_steps_multiplier
    )  # Max environment steps
    step = 0
    done = False

    pbar = tqdm(total=max_steps, desc="Agent execution")

    while not done and step < max_steps:  # skip loop
        # Compute steps left for horizon adjustment
        steps_left = max((max_steps - step) // cfg.frameskip, 1)

        # Plan using buffered observation
        current_obs = obs_buffer.get()
        actions, planning_result = plan_step(model, planner, current_obs, steps_left)

        # actions shape: [num_act_stepped, action_dim] where action_dim = frameskip * actual_action_dim
        # But num_act_stepped may be truncated if steps_left < horizon
        # Reshape to [N, frameskip, actual_action_dim] then flatten to [N * frameskip, actual_action_dim]
        num_stepped = actions.shape[0]
        all_actions = actions.view(num_stepped, cfg.frameskip, cfg.action_dim)
        all_actions = all_actions.reshape(-1, cfg.action_dim)

        # Denormalize
        all_actions_denorm = preprocessor.denormalize_actions(all_actions)

        # Execute actions
        for i, action in enumerate(all_actions_denorm):
            obs, reward, done, info = env.step(action.numpy())
            obs_buffer.add(obs, info)
            step += 1

            # Record at frameskip intervals
            if (i + 1) % cfg.frameskip == 0:
                agent_frames.append(obs["visual"].copy())

            pbar.update(1)
            pbar.set_postfix({"loss": f"{planning_result.losses[-1].item():.4f}"})

            if done:
                break

        # Check success
        eval_result = env.eval_state(goal_state, info["state"])
        if eval_result["success"]:
            print(f"\n  Success at step {step}!")
            break

    pbar.close()

    # Compute final metrics
    final_state = info["state"]
    eval_result = env.eval_state(goal_state, final_state)

    print(f"\nResults:")
    print(f"  Success: {eval_result['success']}")
    print(f"  State distance: {eval_result['state_dist']:.4f}")
    print(f"  Steps taken: {step}")

    env.close()

    return {
        "success": eval_result["success"],
        "state_dist": eval_result["state_dist"],
        "steps": step,
        "agent_frames": agent_frames,
        "expert_frames": expert_frames,
        "init_state": init_state,
        "goal_state": goal_state,
    }


# ============================================================================
# Video Saving
# ============================================================================


def save_video(frames: List[np.ndarray], path: str, fps: int = 10):
    """Save frames as video/GIF."""
    import imageio

    frames_uint8 = []
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (
                (frame * 255).astype(np.uint8)
                if frame.max() <= 1.0
                else frame.astype(np.uint8)
            )
        frames_uint8.append(frame)

    imageio.mimsave(path, frames_uint8, fps=fps, loop=0)
    print(f"Saved video to {path}")


def save_comparison_video(
    agent_frames: List[np.ndarray],
    expert_frames: List[np.ndarray],
    path: str,
    fps: int = 10,
):
    """Save side-by-side comparison video."""
    import imageio

    # Pad to same length
    max_len = max(len(agent_frames), len(expert_frames))
    while len(agent_frames) < max_len:
        agent_frames.append(agent_frames[-1])
    while len(expert_frames) < max_len:
        expert_frames.append(expert_frames[-1])

    combined = []
    for agent, expert in zip(agent_frames, expert_frames):
        # Add labels
        agent_labeled = agent.copy()
        expert_labeled = expert.copy()
        # Green border for expert
        expert_labeled[:3, :, :] = [0, 255, 0]
        expert_labeled[-3:, :, :] = [0, 255, 0]
        expert_labeled[:, :3, :] = [0, 255, 0]
        expert_labeled[:, -3:, :] = [0, 255, 0]
        # Blue border for agent
        agent_labeled[:3, :, :] = [0, 0, 255]
        agent_labeled[-3:, :, :] = [0, 0, 255]
        agent_labeled[:, :3, :] = [0, 0, 255]
        agent_labeled[:, -3:, :] = [0, 0, 255]

        combined.append(np.concatenate([agent_labeled, expert_labeled], axis=1))

    imageio.mimsave(path, combined, fps=fps, loop=0)
    print(f"Saved comparison video to {path}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="JEPA-WM PushT Evaluation")

    # Model loading options (mutually exclusive)
    load_group = parser.add_mutually_exclusive_group(required=True)
    load_group.add_argument(
        "--hub", type=str, help="Load from hub (e.g., jepa_wm_pusht)"
    )
    load_group.add_argument(
        "--checkpoint", type=str, help="Load from .pth.tar checkpoint"
    )

    # Evaluation options
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )

    # Planner options (can override defaults)
    parser.add_argument("--iterations", type=int, default=10, help="CEM iterations")
    parser.add_argument("--num_samples", type=int, default=64, help="CEM samples")
    parser.add_argument("--num_elites", type=int, default=5, help="CEM elites")
    parser.add_argument(
        "--max_steps_multiplier",
        type=int,
        default=10,
        help="Max steps = goal_horizon * frameskip * max_steps_multiplier",
    )
    parser.add_argument(
        "--goal_horizon",
        type=int,
        default=6,
        help="Goal horizon (number of high-level steps to goal)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=6,
        help="Planner lookahead horizon",
    )

    # Dataset path (required for --hub, --checkpoint modes)
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to PushT dataset (required for --hub/--checkpoint)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model based on selected option
    print("Loading model...")
    dset = None
    cfg = EvalConfig(
        seed=args.seed,
        episode=args.episode,
        device=args.device,
        output_dir=args.output_dir,
        iterations=args.iterations,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        max_steps_multiplier=args.max_steps_multiplier,
        goal_horizon=args.goal_horizon,
        horizon=args.horizon,
    )

    if args.hub:
        # Require dataset_path for hub mode
        if not args.dataset_path:
            raise ValueError("--dataset_path is required when using --hub")
        model, preprocessor = load_model_hub(args.hub, args.device)
        # Load dataset with ImageNet normalization (only need val set for inference)
        # Use return_traj_dset=True to get full trajectories (not sliced windows)
        dset, _ = load_pusht_datasets(
            args.dataset_path,
            normalize_mean=IMAGENET_MEAN,
            normalize_std=IMAGENET_STD,
            return_train=False,
            return_traj_dset=True,
        )

    elif args.checkpoint:
        # Require dataset_path for checkpoint mode
        if not args.dataset_path:
            raise ValueError("--dataset_path is required when using --checkpoint")
        model, preprocessor = load_model_pretrained(
            args.checkpoint,
            args.device,
            ctxt_window=cfg.ctxt_window,
            normalize_mean=cfg.normalize_mean,
            normalize_std=cfg.normalize_std,
        )
        # Load dataset with configured normalization (only need val set for inference)
        # Use return_traj_dset=True to get full trajectories (not sliced windows)
        dset, _ = load_pusht_datasets(
            args.dataset_path,
            normalize_mean=cfg.normalize_mean,
            normalize_std=cfg.normalize_std,
            return_train=False,
            return_traj_dset=True,
        )

    print(
        f"Model loaded. Action dim: {model.action_dim}, Context window: {model.ctxt_window}"
    )

    # =========================================================================
    # WORKAROUND: Patch the model's unroll method to fix tensor dimension mismatch
    # The original unroll has a bug where act_feats has fewer timesteps than vid_feats
    # at the start of the loop, causing a mismatch in the AdaLN modulation.
    # =========================================================================
    def patched_unroll(self, z_ctxt, act_suffix=None, debug=False):
        """Fixed unroll that pads action features to match video features."""
        T, B, A = act_suffix.shape
        if isinstance(z_ctxt, TensorDict) or isinstance(z_ctxt, dict):
            vid_feats_prefix = z_ctxt["visual"].expand(
                act_suffix.shape[1], *z_ctxt["visual"].shape[1:]
            )
            prop_feats_prefix = z_ctxt["proprio"].expand(
                act_suffix.shape[1], *z_ctxt["proprio"].shape[1:]
            )
        elif isinstance(z_ctxt, torch.Tensor):
            vid_feats_prefix = z_ctxt.expand(act_suffix.shape[1], *z_ctxt.shape[1:])
            prop_feats_prefix = None
        vid_feats = vid_feats_prefix
        prop_feats = prop_feats_prefix

        act_suffix = rearrange(act_suffix, "t b ... -> b t ...")
        act_feats_suffix = self.model.encode_act(act_suffix)

        # FIX: Initialize act_feats with zeros to match ctxt_window - 1 timesteps
        # This ensures act_feats[:, -ctxt_window:] always has ctxt_window timesteps
        num_padding = self.ctxt_window - 1
        if num_padding > 0:
            # Create zero padding with the same shape as one action feature timestep
            padding_shape = list(act_feats_suffix[:, :1].shape)
            padding_shape[1] = num_padding
            act_feats = torch.zeros(
                padding_shape,
                device=act_feats_suffix.device,
                dtype=act_feats_suffix.dtype,
            )
        else:
            act_feats = None

        for h in range(T):
            new_act_feats = act_feats_suffix[:, h : h + 1]
            if act_feats is None:
                act_feats = new_act_feats
            else:
                act_feats = torch.cat([act_feats, new_act_feats], dim=1)

            pred_video_features, _, pred_proprio_features = self.model.forward_pred(
                vid_feats[:, -self.ctxt_window :],
                act_feats[:, -self.ctxt_window :],
                prop_feats[:, -self.ctxt_window :] if prop_feats is not None else None,
                debug=debug,
            )
            next_vid_feat = pred_video_features[:, -1:]

            if prop_feats is not None:
                if self.proprio_mode == "compute_new_pose":
                    from app.plan_common.datasets.droid_dset import compute_new_pose

                    next_prop_feat = compute_new_pose(
                        prop_feats[:, -1:], act_suffix[:, h : h + 1]
                    )
                elif self.proprio_mode == "predict_proprio":
                    next_prop_feat = pred_proprio_features[:, -1:]
                else:
                    raise ValueError(f"Invalid mode: {self.proprio_mode}")

            vid_feats = torch.cat([vid_feats, next_vid_feat], dim=1)
            if prop_feats is not None:
                prop_feats = torch.cat([prop_feats, next_prop_feat], dim=1)

        if isinstance(z_ctxt, TensorDict):
            vid_feats = rearrange(vid_feats, "b t ... -> t b ...")
            prop_feats = rearrange(prop_feats, "b t ... -> t b ...")
            return TensorDict({"visual": vid_feats, "proprio": prop_feats})
        elif isinstance(z_ctxt, torch.Tensor):
            vid_feats = rearrange(vid_feats, "b t ... -> t b ...")
            return vid_feats

    # Apply the patch
    import types

    model.unroll = types.MethodType(patched_unroll, model)
    print("Applied unroll patch to fix tensor dimension mismatch")
    # =========================================================================

    # Create random generators
    generator = torch.Generator(device="cpu")
    gpu_generator = torch.Generator(device=args.device)

    # Run evaluation
    print(f"\nRunning evaluation episode {args.episode}...")
    start_time = time()

    results = run_evaluation(
        model=model,
        preprocessor=preprocessor,
        dset=dset,
        cfg=cfg,
        generator=generator,
        gpu_generator=gpu_generator,
    )

    elapsed = time() - start_time
    print(f"\nEvaluation completed in {elapsed:.1f}s")

    # Save videos
    if cfg.output_dir:
        agent_path = os.path.join(cfg.output_dir, f"ep{cfg.episode}_agent.gif")
        expert_path = os.path.join(cfg.output_dir, f"ep{cfg.episode}_expert.gif")
        comparison_path = os.path.join(
            cfg.output_dir, f"ep{cfg.episode}_comparison.gif"
        )

        save_video(results["agent_frames"], agent_path)
        save_video(results["expert_frames"], expert_path)
        save_comparison_video(
            results["agent_frames"], results["expert_frames"], comparison_path
        )

    return results


if __name__ == "__main__":
    main()
