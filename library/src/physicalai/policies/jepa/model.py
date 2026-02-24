# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA model implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast, Optional

import torch
from torch import nn

from physicalai.policies.jepa.encoders import DinoEncoder, ProprioceptiveEmbedding
from physicalai.policies.jepa.models import vit_predictor_AdaLN
from physicalai.policies.jepa.models.planner import CEMPlanner
from physicalai.policies.jepa.models.vit_enc_preds import EncPredWM
from physicalai.policies.jepa.models.world_wm import VideoWM
from physicalai.policies.jepa.preprocessing.preprocessor import Preprocessor
from physicalai.policies.jepa.preprocessing.transforms import make_inverse_transforms, make_transforms

logger = logging.getLogger(__name__)

class JEPAModel(nn.Module):
    def __init__(self,
                 *,
                 img_size: int = 224,
                 enc_version: str = "dinov2_vits14",
                 pred_depth: int = 6,
                 pred_embed_dim: int = 384,
                 pred_num_heads: int = 16,
                 use_proprio: bool = True,
                 proprio_emb_dim: int = 16,
                 proprio_dim: int = 4,
                 action_dim: int = 2,
                 ctxt_window: int = 2,
                 frameskip: int = 5,
                 action_skip: int = 1,
                 pt_weights: Optional[Path | str] = None,
                 image_mean: list[float] = (0.485, 0.456, 0.406),
                 image_std: list[float] = (0.229, 0.224, 0.225),
                 action_mean: list[float] = (-0.0087, 0.0068),
                 action_std: list[float] = (0.2019, 0.2002),
                 state_mean: list[float] = (236.6155, 264.5674, 255.1307, 266.3721, 1.9584, -2.93032027, 2.54307914),
                 state_std: list[float] = (101.1202, 87.0112, 52.7054, 57.4971, 1.7556, 74.84556075, 74.14009094),
                 proprio_mean: list[float] = (236.6155, 264.5674, -2.93032027, 2.54307914),
                 proprio_std: list[float] = (101.1202, 87.0112, 74.84556075, 74.14009094),
                 num_hist: int = 3,
                 num_pred: int = 1,
                 enc_type: str = "dino",
                 pred_type: str = "AdaLN",
                 action_emb_dim: int = 10,
                 batch_size: int = 8,
                 num_epochs: int = 1000,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-7,
                 warmup_epochs: int = 2,
                 clip_grad: float = 1.0,
                 freeze_image_encoder: bool = True,
                 l2_loss_weight: float = 1.0,
                 l1_loss_weight: float = 0.0,
                 cos_loss_weight: float = 0.0,
                 rollout_steps: int = 1,
                 rollout_stop_gradient: bool = True,
                 goal_horizon: int = 6,
                 planner_name: str = "cem",
                 iterations: int = 30,
                 num_samples: int = 300,
                 num_elites: int = 10,
                 horizon: int = 6,
                 var_scale: float = 1.0,
                 num_act_stepped: int = 6,
                 objective_type: str = "L2",
                 alpha: float = 0.1,
                 with_target: bool = True,
                 with_velocity: bool = True,
                 max_steps_multiplier: int = 10,
            ):
        super().__init__()

        # The model predicts 'frameskip' actions
        internal_action_dim = action_dim * frameskip
        num_frames = num_hist + num_pred

        # Create image encoder
        self.encoder = DinoEncoder(name=enc_version, feature_key="x_norm_patchtokens")
        if freeze_image_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        # Create proprioceptive encoder
        self.proprio_encoder = ProprioceptiveEmbedding(
            in_chans=proprio_dim,
            embed_dim=proprio_emb_dim,
            tokens_per_step=1,
            tubelet_size=1,
            num_frames=num_frames,
        )

        # Create predictor
        self.predictor = vit_predictor_AdaLN(
            img_size=img_size,
            patch_size=14,
            num_frames=num_frames,
            tubelet_size=1,
            embed_dim=self.encoder.embed_dim,
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
        )

        # # Create optimizer for trainable parameters
        # trainable_params = list(predictor.parameters())
        # if proprio_encoder is not None:
        #     trainable_params += list(proprio_encoder.parameters())
        #
        # optimizer = torch.optim.AdamW(
        #     trainable_params,
        #     lr=learning_rate,
        #     weight_decay=weight_decay,
        #     betas=(0.9, 0.999),
        # )

        # Configuration
        cfgs_loss = {
            "l2_loss_weight": l2_loss_weight,
            "l1_loss_weight": l1_loss_weight,
            "cos_loss_weight": cos_loss_weight,
            "smooth_l1_loss_weight": 0.0,
            "proprio_loss": use_proprio,
        }
        grid_size = img_size // 14

        # Create world model
        self.video_wm = VideoWM(
            encoder=self.encoder,
            predictor=self.predictor,
            action_encoder=None,
            proprio_encoder=self.proprio_encoder,
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
            # device=torch.device(device) if isinstance(device, str) else device,
            # optimizer=optimizer,
            scaler=None,
            clip_grad=clip_grad,
            mixed_precision=False,
            cfgs_loss=cfgs_loss,
            heads=[],
        )

        # Preprocessing
        normalize = [list(image_mean), list(image_std)]
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
            action_mean=torch.tensor(action_mean),
            action_std=torch.tensor(action_std),
            state_mean=torch.tensor(state_mean),
            state_std=torch.tensor(state_std),
            proprio_mean=torch.tensor(proprio_mean),
            proprio_std=torch.tensor(proprio_std),
            transform=transform,
            inverse_transform=inverse_transform,
        )

        # Create inference wrapper (with rollout)
        self.model = EncPredWM(
            model=self.video_wm,
            action_dim=internal_action_dim,
            preprocessor=preprocessor,
            ctxt_window=ctxt_window,
            proprio_mode="predict_proprio",
        )

        # Create CEM planner for inference
        self.planner = CEMPlanner(
            unroll=self.model.unroll,
            action_dim=self.model.action_dim,
            iterations=iterations,
            num_samples=num_samples,
            num_elites=num_elites,
            horizon=horizon,
            var_scale=var_scale,
            num_act_stepped=num_act_stepped,
            local_generator=torch.Generator(),  # could use manual seed here
            decode_unroll=self.model.decode_unroll,
            decode_each_iteration=False,
        )
