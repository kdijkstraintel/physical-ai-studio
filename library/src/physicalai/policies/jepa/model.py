# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA model implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast, Optional

import torch
from physicalai.data.observation import ACTION
from torch import nn

from physicalai.policies.jepa.encoders import DinoEncoder, ProprioceptiveEmbedding
from physicalai.policies.jepa.preprocessing.preprocessor import Preprocessor
from physicalai.policies.jepa.preprocessing.transforms import make_inverse_transforms, make_transforms

from .models.AdaLN_vit import vit_predictor_AdaLN
from .models.planner import CEMPlanner
from .models.vit_enc_preds import EncPredWM
from .models.world_wm import VideoWM

logger = logging.getLogger(__name__)


class JEPAModel(nn.Module):
    def __init__(
        self,
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
        pt_weights: Optional[Path] = None,
        hf_weights: Optional[str] = None,
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
        num_epochs: int = 50,
        ite: Optional[int] = None,
        start_lr: float = 5e-4,
        ref_lr: float = 5e-4,
        final_lr: float = 5e-4,
        weight_decay: float = 1e-7,
        final_weight_decay: float = 1e-6,
        warmup_epochs: int = 0,
        freeze_image_encoder: bool = True,
        use_radamw: bool = False,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        ipe_scale: float = 1.0,
        clip_grad: float = 1.0,
        mixed_precision: bool = True,
        dtype: str = "bfloat16",
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

        # Create frozen image encoder
        encoder = DinoEncoder(name=enc_version, feature_key="x_norm_patchtokens")
        if freeze_image_encoder:
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.eval()

        # Create predictor
        predictor = vit_predictor_AdaLN(
            img_size=img_size,
            patch_size=14,
            num_frames=num_frames,
            tubelet_size=1,
            embed_dim=encoder.embed_dim,
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

        # Create proprioceptive encoder
        proprio_encoder = ProprioceptiveEmbedding(
            in_chans=proprio_dim,
            embed_dim=proprio_emb_dim,
            tokens_per_step=1,
            tubelet_size=1,
            num_frames=num_frames,
        )

        # Create optimizer for trainable parameters
        trainable_params = list(predictor.parameters())
        if proprio_encoder is not None:
            trainable_params += list(proprio_encoder.parameters())

        # Advanced Optimizer, scaler, and schedulers
        optimizer, scaler, scheduler, wd_scheduler = init_opt(
            predictor=predictor,
            action_encoder=None,  # Action encoder is inside predictor for AdaLN
            proprio_encoder=proprio_encoder,
            encoder=encoder,
            freeze_encoder=True,  # Encoder is always frozen for this training
            iterations_per_epoch=ite,
            start_lr=config.start_lr,
            ref_lr=config.ref_lr,
            final_lr=config.final_lr,
            warmup=config.warmup,
            num_epochs=config.num_epochs,
            use_radamw=config.use_radamw,
            weight_decay=config.weight_decay,
            final_weight_decay=config.final_weight_decay,
            mixed_precision=config.mixed_precision,
            ipe_scale=config.ipe_scale,
            betas=config.betas,
            eps=config.eps,
        )

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
            device=self.device,
            optimizer=optimizer,
            scaler=scaler,
            clip_grad=clip_grad,
            mixed_precision=mixed_precision,
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
            model=video_wm,
            action_dim=internal_action_dim,
            preprocessor=preprocessor,
            ctxt_window=ctxt_window,
            proprio_mode="predict_proprio",
        )

        # Load weigths from hf hub or from pth file.
        if pt_weights is not None:
            self._load_weights_pt(pt_weights)
        elif hf_weights is not None:
            self._load_weights_hf(hf_weights)

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
        self._observation_buffer: ObservationBuffer(ctxt_window)

        logger.info(f"JEPA Model with Action dim: {model.action_dim}, Context window: {model.ctxt_window}")

    def _load_weights_hf(self, hf_weights: str | None = None):
        import hubconf
        model_fn = hubconfg.hf_weights
        self.model, preprocessor = model_fn(pretrained=True, device=self.device)
        self.model.eval()
        logger.info(print(f"Loaded from hf hub {hf_weights}"))

    def _load_weights_pt(self, pt_weights: Path | None = None):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info(print(f"Loaded pretrained checkpoint from {checkpoint_path}"))
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

        # First load an empy model
        self._load_empty()

        # Replace predictor weights
        state_dict = {
            k.replace("module.", ""): v for k, v in checkpoint["predictor"].items()
        }
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()

        # Replace proprio weights
        if use_proprio and "proprio_encoder" in checkpoint:
            state_dict = {
                k.replace("module.", ""): v
                for k, v in checkpoint["proprio_encoder"].items()
            }
            self.proprio_encoder.load_state_dict(state_dict)
            self.proprio_encoder.eval()

        self.model.eval()


    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass for the JEPA model.

        During training, processes the input batch to compute the loss for action prediction.
        During inference, delegates to predict_action_chunk for action generation.

        Args:
            batch: Dictionary containing input tensors with keys:
                - STATE: Robot state tensor
                - ACTION: Ground truth action tensor (training only)
                - "tokenized_prompt": Language instruction tokens
                - "tokenized_prompt_mask": Attention mask for language tokens
                - EXTRA + ".actions_id_pad": Optional padding mask for actions
                - Image-related keys generated by JEPA's preprocessor

        Returns:
            If training: A tuple containing:
                - loss: Mean loss value as a tensor
                - loss_dict: Dictionary with intermediate loss values for debugging
            If inference: Output from predict_action_chunk (action predictions)
        """
        if self.training:
            raise NotImplementedError("JEPA does not support training")
            # batch = self._preprocess_batch(batch)
            # images, img_masks = batch[IMAGES], batch["image_masks"]
            # state = self._prepare_state(batch)
            # actions = self._prepare_action(batch)
            #
            # lang_tokens = batch["tokenized_prompt"]
            # lang_masks = batch["tokenized_prompt_mask"]
            # actions_is_pad = batch.get(EXTRA + ".actions_id_pad")
            # loss_dict = {}
            # losses = self._model.forward(images, img_masks, lang_tokens, lang_masks, state, actions)
            # loss_dict["losses_after_forward"] = losses.clone()
            #
            # if actions_is_pad is not None:
            #     in_episode_bound = ~actions_is_pad
            #     losses *= in_episode_bound.unsqueeze(-1)
            #     loss_dict["losses_after_in_ep_bound"] = losses.clone()
            #
            # # Remove padding
            # losses = losses[:, :, : self._max_action_dim]
            # loss_dict["losses_after_rm_padding"] = losses.clone()
            #
            # loss = losses.mean()
            # loss_dict["loss"] = loss.item()
            # return loss, loss_dict
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict a chunk of actions from input batch.

        This method processes the input batch, prepares images, state, and language tokens,
        then uses the model to sample actions. The resulting actions are unpadded to match
        the original action dimension and optionally encoded for Pi Aloha compatibility.

        Args:
            batch: A dictionary containing input tensors including images, state information,
                and tokenized prompts with their masks.

        Returns:
            torch.Tensor: A tensor of predicted actions with shape matching the original
                action dimensions from the dataset statistics.
        """
        return
        # self.model.preprocessor.denormalize_actions(batch[])
        #
        # processed_batch = self._preprocess_batch(batch)
        # images, img_masks = processed_batch[IMAGES], processed_batch["image_masks"]
        # state = self._prepare_state(processed_batch)
        # lang_tokens = processed_batch["tokenized_prompt"]
        # lang_masks = processed_batch["tokenized_prompt_mask"]
        #
        # actions = self._model.sample_actions(
        #     images,
        #     img_masks,
        #     lang_tokens,
        #     lang_masks,
        #     state,
        # )
        #
        # # Unpad actions
        # original_action_dim = int(self._dataset_stats[ACTION]["shape"][-1])
        # actions = actions[:, :, :original_action_dim]
        #
        # if self._adapt_to_pi_aloha:
        #     actions = self._pi_aloha_encode_actions(actions)
        #
        # return actions

class ObservationBuffer:
    """Buffer to maintain temporal context for observations."""

    def __init__(self, num_obs: int):
        self.num_obs = num_obs
        self._frames: List[torch.Tensor] = []
        self._proprios: List[torch.Tensor] = []

    def reset(self):
        """Clear the buffer."""
        self._frames = []
        self._proprios = []

    def add(self, frame: torch.Tensor, prorio: torch.Tensor):
        # Convert HWC to CHW format if needed
        if frame.ndim == 3 and frame.shape[-1] == 3:
            frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
        frame = frame.to(torch.uint8)
        proprio = proprio.to(torch.float32)

        self._frames.append(frame)
        self._proprios.append(proprio)

        # Keep only the most recent frames/proprios
        self._frames = self._frames[-min(self.num_obs, len(self._frames)) :]
        self._proprios = self._proprios[-min(self.num_obs, len(self._proprios)) :]

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get buffered observation with time dimension [T, C, H, W] and [T, D]."""
        # Pad if not enough frames yet
        while len(self._frames) < self.num_obs:
            self._frames.insert(0, self._frames[0].clone())
        while len(self._proprios) < self.num_obs:
            self._proprios.insert(0, self._proprios[0].clone())

        return torch.stack(self._frames[-self.num_obs :]), torch.stack(self._proprios[-self.num_obs :])

def init_opt(
    predictor,
    action_encoder,
    proprio_encoder=None,
    encoder=None,
    iterations_per_epoch=1000,
    start_lr=0.0,
    ref_lr=1e-3,
    warmup=2,
    num_epochs=90,
    freeze_encoder=True,
    use_radamw=False,
    weight_decay=1e-6,
    final_weight_decay=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    use_wsd_schedule=False,
    anneal_steps=None,
):
    """
    Initialize optimizer and learning rate scheduler.

    Args:
        use_wsd_schedule: If True, use WSDSchedule (Warmup-Stable-Decay) instead of WarmupCosineSchedule.
        anneal_steps: Number of steps for the decay phase in WSDSchedule. If None, defaults to
                      warmup_steps to be backwards compatible (approximately similar behavior).
                      Only used when use_wsd_schedule=True.
    """
    param_groups = []
    param_groups += [
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1) and p.requires_grad
            )
        },
    ]
    param_groups += [
        {
            "params": (
                p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1) and p.requires_grad
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]
    if action_encoder is not None:
        param_groups += [
            {
                "params": (
                    p
                    for n, p in action_encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1) and p.requires_grad
                )
            }
        ]
        param_groups += [
            {
                "params": (
                    p
                    for n, p in action_encoder.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1) and p.requires_grad
                ),
                "WD_exclude": True,
                "weight_decay": 0,
            },
        ]
    if proprio_encoder is not None:
        param_groups += [
            {
                "params": (
                    p
                    for n, p in proprio_encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1) and p.requires_grad
                )
            }
        ]
        param_groups += [
            {
                "params": (
                    p
                    for n, p in proprio_encoder.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1) and p.requires_grad
                ),
                "WD_exclude": True,
                "weight_decay": 0,
            },
        ]
    if encoder is not None and not freeze_encoder:
        param_groups += [
            {
                "params": (
                    p
                    for n, p in encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1) and p.requires_grad
                )
            }
        ]
        param_groups += [
            {
                "params": (
                    p
                    for n, p in encoder.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1) and p.requires_grad
                ),
                "WD_exclude": True,
                "weight_decay": 0,
            },
        ]

    if use_radamw:
        logger.info("Using Rescaled-AdamW")
        optimizer = RAdamW(param_groups, betas=betas, eps=eps)
    else:
        logger.info("Using AdamW")
        optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    warmup_steps = int(warmup * iterations_per_epoch)
    T_max = int(ipe_scale * num_epochs * iterations_per_epoch)
    if use_wsd_schedule:
        # Use WSDSchedule (Warmup-Stable-Decay)
        # Default anneal_steps to warmup_steps
        if anneal_steps is None:
            anneal_steps = warmup_steps
        logger.info(f"Using WSDSchedule with warmup_steps={warmup_steps}, anneal_steps={anneal_steps}, T_max={T_max}")
        scheduler = WSDSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            anneal_steps=anneal_steps,
            T_max=T_max,
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
        )
    else:
        # Use WarmupCosineSchedule (default for backwards compatibility)
        logger.info(f"Using WarmupCosineSchedule with warmup_steps={warmup_steps}, T_max={T_max}")
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=T_max,
        )

    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=weight_decay,
        final_wd=final_weight_decay,
        T_max=T_max,
    )
    scaler = torch.amp.GradScaler("cuda") if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
