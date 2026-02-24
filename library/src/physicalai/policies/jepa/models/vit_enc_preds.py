# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import numpy as np
from einops import rearrange

def _lazy_import_tensordict() -> tuple:
    """Lazy import tensordict to reduce initial load time.

    Returns:
        Tuple containing (TensorDict).

    Raises:
        ImportError: If tensordict is not installed.
    """
    try:
        from tensordict import TensorDict  # noqa: PLC0415
    except ImportError as e:
        msg = "jepa requires tensordict library.\n\nInstall with:\n    uv pip install tensordict"
        raise ImportError(msg) from e
    else:
        return (TensorDict,)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def compute_new_pose(pose, action):
    """
    :param pose: [B, T=1, 7]
    :param action: [B, T=1, 7]
    :returns: [B, T=1, 7]
    """

    device, dtype = pose.device, pose.dtype
    pose = pose[:, 0].cpu().numpy()
    action = action[:, 0].cpu().numpy()
    # -- compute delta xyz
    new_xyz = pose[:, :3] + action[:, :3]
    # -- compute delta theta
    thetas = pose[:, 3:6]
    delta_thetas = action[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    delta_matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in delta_thetas]
    angle_diff = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    new_angle = np.stack([d for d in angle_diff], axis=0)  # [B, 7]
    # -- compute delta gripper
    new_closedness = pose[:, -1:] + action[:, -1:]
    new_closedness = np.clip(new_closedness, 0, 1)
    # -- new pose
    new_pose = np.concatenate([new_xyz, new_angle, new_closedness], axis=-1)
    return torch.from_numpy(new_pose).to(device).to(dtype)[:, None]

class EncPredWM(nn.Module):
    """Wrapper around VideoWM for encoding, prediction unrolling, and decoding.
    Provides interfaces for encoding raw observations into latent space, unrolling
    predictions conditioned on actions, and decoding predictions back to visual/state space.
    """

    def __init__(
        self,
        model,
        action_dim,
        preprocessor,
        ctxt_window=2,
        proprio_mode="predict_proprio",
    ):
        """Args:
        proprio_mode (str): Mode for proprio handling. Options:
        - "predict_proprio": Use predictor to predict proprio features (default)
        - "compute_new_pose": Use compute_new_pose() to compute proprio from actions
        """
        super().__init__()
        self.model = model
        self.heads = model.heads
        self.device = self.model.device
        self.action_dim = action_dim
        self.tubelet_size_enc = self.model.tubelet_size_enc
        self.encode_proprio = self.model.encode_proprio
        self.encode_obs = self.model.encode_obs
        self.preprocessor = preprocessor
        self.grid_size = self.model.grid_size
        self.action_skip = self.model.action_skip
        self.normalize_reps = self.model.normalize_reps
        self.enc_type = self.model.enc_type
        # wrapper_kwargs
        self.ctxt_window = ctxt_window
        self.proprio_mode = proprio_mode

    def unroll(self, z_ctxt, act_suffix=None, debug=False):
        """Autoregressively predict latent features forward in time using actions.

        Starts from context features and iteratively predicts next timestep using
        action conditioning. Maintains a sliding window of ctxt_window frames for prediction.

        Args:
            z_ctxt (TensorDict or Tensor): Context latent features.
                If TensorDict: keys "visual" [B, tau, V, H, W, D] and "proprio" [B, tau, proprio_tokens, D].
                If Tensor: visual features only [B, tau, V, H, W, D].
            act_suffix (Tensor): Action sequence [T, B, A] where A matches predictor's expected action dim.
            debug (bool): Enable debug mode in forward_pred.

        Returns:
            TensorDict or Tensor: Predicted latent features [T+tau, B, V, H, W, D].
                Returns same type as z_ctxt input.
        """
        (TensorDict,) = _lazy_import_tensordict()

        T, B, A = act_suffix.shape
        if isinstance(z_ctxt, TensorDict) or isinstance(z_ctxt, dict):
            vid_feats_prefix = z_ctxt["visual"].expand(act_suffix.shape[1], *z_ctxt["visual"].shape[1:])
            prop_feats_prefix = z_ctxt["proprio"].expand(act_suffix.shape[1], *z_ctxt["proprio"].shape[1:])
        elif isinstance(z_ctxt, torch.Tensor):
            vid_feats_prefix = z_ctxt.expand(act_suffix.shape[1], *z_ctxt.shape[1:])
            prop_feats_prefix = None
        vid_feats = vid_feats_prefix
        prop_feats = prop_feats_prefix
        # CAUSE OF THE BUG: act_suffix = rearrange(act_suffix, "t b (act_suffix tube) -> b (t tube) act_suffix", tube=self.tubelet_size_enc)
        act_suffix = rearrange(act_suffix, "t b ... -> b t ...")
        act_feats_suffix = self.model.encode_act(act_suffix)  # (b t a) or (b t 1 d) if action_encoder_inpred=False
        for h in range(T):
            new_act_feats = act_feats_suffix[:, h : h + 1]
            if h == 0:
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
            # self.normalize_reps already done in self.model.forward_pred()
            if prop_feats is not None:
                if self.proprio_mode == "compute_new_pose":
                    # Use compute_new_pose to compute proprio from actions
                    # act_suffix has raw actions in shape [B, T, A]
                    next_prop_feat = compute_new_pose(prop_feats[:, -1:], act_suffix[:, h : h + 1])
                elif self.proprio_mode == "predict_proprio":
                    # Use predictor to predict proprio
                    next_prop_feat = pred_proprio_features[:, -1:]
                else:
                    raise ValueError(
                        f"Invalid mode: {self.proprio_mode}. Must be 'predict_proprio' or 'compute_new_pose'"
                    )
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

    @torch.no_grad()
    def encode(self, obs, act=True):
        """Encode raw simulator observations into latent representations.

        Handles preprocessing (normalization, transforms) and encoding in a single pass
        to minimize CPU-GPU transfers. Supports both visual-only and multimodal inputs.

        Args:
            obs (TensorDict, dict, or Tensor): Raw observations from simulator.
                If dict/TensorDict: keys "visual" [B, T, C, H, W] and "proprio" [B, T, P].
                If Tensor: visual observations only [B, T, C, H, W].
            act (bool): Unused legacy parameter.

        Returns:
            TensorDict or Tensor: Latent features [B, T, V, H, W, D].
                Returns TensorDict with "visual" and "proprio" keys if input is dict, else Tensor.
        """
        (TensorDict,) = _lazy_import_tensordict()

        if isinstance(obs, TensorDict) or isinstance(obs, dict):
            visual = obs["visual"]
            trans_proprio = self.preprocessor.normalize_proprios(obs["proprio"].cpu()).to(
                self.model.device, dtype=torch.float32, non_blocking=True
            )
            proprio_emb = self.encode_proprio(trans_proprio)
        elif isinstance(obs, torch.Tensor):
            visual = obs
        else:
            raise ValueError("Input must be a dictionary with key 'visual' or a tensor")
        b, t, c, h, w = visual.shape  # b t c h w
        visual = visual.to(self.model.device, non_blocking=True, dtype=torch.float32)
        trans_visual = visual / 255.0  # instead of calling preprocessor.preprocess_obs_visual()
        # same transform as train time transform part of dataloader
        trans_visual = self.preprocessor.transform(trans_visual)
        if self.model.batchify_video:
            trans_visual = rearrange(trans_visual, "b t ... -> (b t) ...")
        if self.model.dup_image:
            if not self.model.batchify_video:
                trans_visual = trans_visual.repeat_interleave(2, dim=1)  # b t c h w -> b 2*t c h w
            else:
                trans_visual = trans_visual.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # b c h w -> b c 2 h w
                # vjepa expects (b c t h w), so no rearrange needed below
        else:
            # if we feed t=1 to a model expecting at least t=2, need to duplicate
            # self.tubelet_size_enc==1 for self.enc_type == "dino"
            if self.enc_type == "vjepa":
                trans_visual = trans_visual.repeat(1, self.tubelet_size_enc, 1, 1, 1)  # b 1 c h w -> b t c h w
        if self.enc_type == "dino":
            visual_embs = self.model.encoder(trans_visual)
            visual_embs = rearrange(
                visual_embs, "(b t) (h w) d -> b t 1 h w d", b=b, h=self.grid_size, w=self.grid_size
            )
        elif self.enc_type == "vjepa":
            if not self.model.batchify_video:
                trans_visual = rearrange(trans_visual, "b t c h w -> b c t h w ")
            visual_embs = self.model.encoder(trans_visual)
            if self.model.batchify_video:
                visual_embs = rearrange(
                    visual_embs, "(b t) (h w) d -> b t 1 h w d", b=b, t=t, h=self.grid_size, w=self.grid_size
                )
            else:
                visual_embs = rearrange(visual_embs, "b (t h w) d -> b t 1 h w d", h=self.grid_size, w=self.grid_size)
        if self.normalize_reps:
            visual_embs = F.layer_norm(visual_embs, (visual_embs.size(-1),))
        if isinstance(obs, TensorDict) or isinstance(obs, dict):
            return TensorDict({"visual": visual_embs, "proprio": proprio_emb}, device=visual_embs.device)
        elif isinstance(obs, torch.Tensor):
            return visual_embs

    @torch.no_grad()
    def decode_unroll(self, predicted_encs, batch=False):
        """Decode predicted latent features back to visual observations.

        Uses the image_head decoder to reconstruct RGB frames from latent predictions.

        Args:
            predicted_encs (TensorDict, dict, or Tensor): Predicted latent features.
                If dict/TensorDict: key "visual" [T, B, V, H, W, D].
                If Tensor: visual features [T, B, V, H, W, D].
            batch (bool): If True, return batch dimension [B, T, H, W, 3], else [T, H, W, 3].

        Returns:
            ndarray: Decoded RGB frames as uint8 in [0, 255].
                Shape [B, T, H, W, 3] if batch=True, else [T, H, W, 3].
        """
        (TensorDict,) = _lazy_import_tensordict()

        if isinstance(predicted_encs, TensorDict) or isinstance(predicted_encs, dict):
            visual_feat_preds = predicted_encs["visual"]
            proprio_feat_preds = predicted_encs["proprio"]
        elif isinstance(predicted_encs, torch.Tensor):
            visual_feat_preds = predicted_encs
        if "image_head" in self.model.heads:
            visual_feat_preds = rearrange(visual_feat_preds, "t b v h w c -> b t v h w c ")
            eval_image_samples = self.model.heads["image_head"].decode(visual_feat_preds)
            if batch:
                pred_frames = eval_image_samples[:, :, 0].cpu().numpy()
            else:
                pred_frames = eval_image_samples[0, :, 0].cpu().numpy()
            return pred_frames
        # TODO: if proprio decoder heads, also return its decoding
