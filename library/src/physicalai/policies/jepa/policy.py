# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA Policy - Lightning wrapper for training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

from physicalai.data.observation import ACTION
from physicalai.export.mixin_export import Export
from physicalai.policies.base import Policy
from physicalai.policies.jepa import JEPAConfig
from physicalai.train.utils import reformat_dataset_to_match_policy

from .model import JEPAModel

if TYPE_CHECKING:
    from physicalai.data import Observation
    from physicalai.gyms import Gym


class JEPA(Export, Policy):
    """JEPA Policy - FAIR's JEPA-WM policy

    Lightning wrapper for training and inference with JEPA model.

    Uses dual-path initialization:
    - **Lazy path**: `JEPA()` + `trainer.fit()` - model built in setup()
    - **Eager path**: `JEPa.load_from_checkpoint()` - model built immediately

    Args:
        n_obs_steps: Number of observation steps to use. Default: 1.
        chunk_size: Size of action chunks for prediction. Default: 50.
        n_action_steps: Number of action steps to execute. Default: 50.
        max_state_dim: Maximum state dimension (shorter vectors are padded). Default: 32.
        max_action_dim: Maximum action dimension (shorter vectors are padded). Default: 32.
        resize_imgs_with_padding: Target image resolution (height, width). Default: (512, 512).
        tokenizer_max_length: Maximum length for tokenizer. Default: 48.
        vlm_model_name: VLM backbone model name. Default: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct".
        load_vlm_weights: Whether to load pretrained VLM weights. Default: False.
        add_image_special_tokens: Whether to use special image tokens around image features. Default: False.
        attention_mode: Attention mode for the model. Default: "cross_attn".
        prefix_length: Prefix length for attention. Default: -1.
        pad_language_to: Padding strategy for language tokens. Default: "longest".
        num_expert_layers: Number of expert layers (-1 matches VLM layers). Default: -1.
        num_vlm_layers: Number of layers used in the VLM. Default: 16.
        self_attn_every_n_layers: Interleave self-attention layers frequency. Default: 2.
        expert_width_multiplier: Action expert hidden size ratio to VLM. Default: 0.75.
        min_period: Minimum period for sine-cosine positional encoding. Default: 4e-3.
        max_period: Maximum period for sine-cosine positional encoding. Default: 4.0.
        num_steps: Number of decoding steps. Default: 10.
        use_cache: Whether to use attention cache. Default: True.
        freeze_vision_encoder: Whether to freeze vision encoder during training. Default: True.
        train_expert_only: Whether to train only the expert layers. Default: True.
        train_state_proj: Whether to train state projection layers. Default: True.
        optimizer_lr: Learning rate for optimizer. Default: 1e-4.
        optimizer_betas: Beta parameters for AdamW optimizer. Default: (0.9, 0.95).
        optimizer_eps: Epsilon for optimizer numerical stability. Default: 1e-8.
        optimizer_weight_decay: Weight decay for optimizer. Default: 1e-10.
        optimizer_grad_clip_norm: Gradient clipping norm value. Default: 10.
        scheduler_warmup_steps: Number of warmup steps for scheduler. Default: 1_000.
        scheduler_decay_steps: Number of steps between learning rate decays. Default: 30_000.
        scheduler_decay_lr: Learning rate decay factor. Default: 2.5e-6.
        dataset_stats: Dataset normalization statistics for eager initialization. Default: None.

    Example:
        Training:

        >>> policy = SmolVLA(learning_rate=2.5e-5)
        >>> trainer = physicalai.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)

        Inference:

        >>> policy = SmolVLA.load_from_checkpoint("checkpoint.ckpt")
        >>> action = policy.select_action(obs)
    """

    def __init__(  # noqa: PLR0913
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
    ) -> None:
        """Initialize JEPA policy.

        Creates JEPAConfig from explicit args and saves it as hyperparameters.
        """
        super().__init__(n_action_steps=frameskip)

        # Create config from explicit args (policy-level config)
        self.config = JEPAConfig(
            img_size=img_size,
            enc_version=enc_version,
            pred_depth=pred_depth,
            pred_embed_dim=pred_embed_dim,
            pred_num_heads=pred_num_heads,
            use_proprio=use_proprio,
            proprio_emb_dim=proprio_emb_dim,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            ctxt_window=ctxt_window,
            frameskip=frameskip,
            action_skip=action_skip,
            pt_weights=pt_weights,
            hf_weights=hf_weights,
            image_mean=image_mean,
            image_std=image_std,
            action_mean=action_mean,
            action_std=action_std,
            state_mean=state_mean,
            state_std=state_std,
            proprio_mean=proprio_mean,
            proprio_std=proprio_std,
            num_hist=num_hist,
            num_pred=num_pred,
            enc_type=enc_type,
            pred_type=pred_type,
            action_emb_dim=action_emb_dim,
            batch_size=batch_size,
            num_epochs=num_epochs,
            ite=ite,
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            weight_decay=weight_decay,
            final_weight_decay=final_weight_decay,
            warmup_epochs=warmup_epochs,
            freeze_image_encoder=freeze_image_encoder,
            use_radamw=use_radamw,
            betas=betas,
            eps=eps,
            ipe_scale=ipe_scale,
            clip_grad=clip_grad,
            mixed_precision=mixed_precision,
            dtype=dtype,
            l2_loss_weight=l2_loss_weight,
            l1_loss_weight=l1_loss_weight,
            cos_loss_weight=cos_loss_weight,
            rollout_steps=rollout_steps,
            rollout_stop_gradient=rollout_stop_gradient,
            goal_horizon=goal_horizon,
            planner_name=planner_name,
            iterations=iterations,
            num_samples=num_samples,
            num_elites=num_elites,
            horizon=horizon,
            var_scale=var_scale,
            num_act_stepped=num_act_stepped,
            objective_type=objective_type,
            alpha=alpha,
            with_target=with_target,
            with_velocity=with_velocity,
            max_steps_multiplier=max_steps_multiplier,
        )

        # Save config as hyperparameters for checkpoint restoration
        self.save_hyperparameters(ignore=["config"])  # Save individual args, not config object
        # Also save config dict for compatibility
        self.hparams["config"] = self.config.to_dict()

        # Model will be built in setup() or immediately if env_action_dim provided
        self.model: JEPAModel | None = None

        # TODO: Implement pre/post processing
        # # Preprocessor/postprocessor set in setup() or _initialize_model()
        # self._preprocessor: SmolVLAPreprocessor | None = None
        # self._postprocessor: SmolVLAPostprocessor | None = None
        #
        # # Eager initialization if dataset_stats is provided
        # if dataset_stats is not None:
        #     self._initialize_model(dataset_stats)
        #
        # self._dataset_stats = dataset_stats

    def _initialize_model(
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]],
    ) -> None:
        """Initialize model and preprocessors.

        Called by both lazy (setup) and eager (checkpoint) paths.

        Args:
            env_action_dim: Environment action dimension.
            dataset_stats: Dataset normalization statistics.
        """
        # TODO: Implement dataset_stats to infer config parameters
        # from .preprocessor import make_smolvla_preprocessors  # noqa: PLC0415

        self.model = JEPAModel(
            img_size=self.config.img_size,
            enc_version=self.config.enc_version,
            pred_depth=self.config.pred_depth,
            pred_embed_dim=self.config.pred_embed_dim,
            pred_num_heads=self.config.pred_num_heads,
            use_proprio=self.config.use_proprio,
            proprio_emb_dim=self.config.proprio_emb_dim,
            proprio_dim=self.config.proprio_dim,
            action_dim=self.config.action_dim,
            ctxt_window=self.config.ctxt_window,
            frameskip=self.config.frameskip,
            action_skip=self.config.action_skip,
            pt_weights=self.config.pt_weights,
            hf_weights=self.config.hf_weights,
            image_mean=self.config.image_mean,
            image_std=self.config.image_std,
            action_mean=self.config.action_mean,
            action_std=self.config.action_std,
            state_mean=self.config.state_mean,
            state_std=self.config.state_std,
            proprio_mean=self.config.proprio_mean,
            proprio_std=self.config.proprio_std,
            num_hist=self.config.num_hist,
            num_pred=self.config.num_pred,
            enc_type=self.config.enc_type,
            pred_type=self.config.pred_type,
            action_emb_dim=self.config.action_emb_dim,
            batch_size=self.config.batch_size,
            num_epochs=self.config.num_epochs,
            ite=self.config.ite,
            start_lr=self.config.start_lr,
            ref_lr=self.config.ref_lr,
            final_lr=self.config.final_lr,
            weight_decay=self.config.weight_decay,
            final_weight_decay=self.config.final_weight_decay,
            warmup_epochs=self.config.warmup_epochs,
            freeze_image_encoder=self.config.freeze_image_encoder,
            use_radamw=self.config.use_radamw,
            betas=self.config.betas,
            eps=self.config.eps,
            ipe_scale=self.config.ipe_scale,
            clip_grad=self.config.clip_grad,
            mixed_precision=self.config.mixed_precision,
            dtype=self.config.dtype,
            l2_loss_weight=self.config.l2_loss_weight,
            l1_loss_weight=self.config.l1_loss_weight,
            cos_loss_weight=self.config.cos_loss_weight,
            rollout_steps=self.config.rollout_steps,
            rollout_stop_gradient=self.config.rollout_stop_gradient,
            goal_horizon=self.config.goal_horizon,
            planner_name=self.config.planner_name,
            iterations=self.config.iterations,
            num_samples=self.config.num_samples,
            num_elites=self.config.num_elites,
            horizon=self.config.horizon,
            var_scale=self.config.var_scale,
            num_act_stepped=self.config.num_act_stepped,
            objective_type=self.config.objective_type,
            alpha=self.config.alpha,
            with_target=self.config.with_target,
            with_velocity=self.config.with_velocity,
            max_steps_multiplier=self.config.max_steps_multiplier,
        )

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy initialization path).

        Called by Lightning before fit/validate/test/predict.

        Args:
            stage: Lightning stage (unused, required by Lightning API).

        Raises:
            TypeError: If train dataset is not a physicalai Dataset.
        """
        del stage  # Unused argument

        if self.model is not None:
            return

        from physicalai.data.dataset import Dataset  # noqa: PLC0415

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        stats_dict = train_dataset.stats

        # Save to hparams for checkpoint
        self.hparams["dataset_stats"] = stats_dict

        self._initialize_model(stats_dict)

        reformat_dataset_to_match_policy(self, datamodule)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass through the model.

        Processes the input batch and either trains the model or predicts actions
        depending on the current mode.

        Args:
            batch: An Observation object containing the input data for the model.

        Returns:
            If training: Returns the model output, either a tensor or a tuple
                containing a tensor and a dictionary of loss metrics.
            If not training: Returns the predicted action chunk as a tensor.

        Raises:
            ValueError: If the model is not initialized during training mode.
        """
        if self.training:
            if self.model is None or self._preprocessor is None:
                msg = "Model is not initialized"
                raise ValueError(msg)

            # TODO implement preprocessing
            # processed_batch = self._preprocessor(batch.to_dict())
            processed_batch = batch.to_dict()
            return self.model(processed_batch)
        return self.predict_action_chunk(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions from the given observation batch.

        Args:
            batch: An Observation object containing the input data for action prediction.

        Returns:
            torch.Tensor: The predicted action chunk after post-processing.

        Raises:
            ValueError: If the model has not been initialized.
        """
        # TODO: Implement pre/post processor
        # if self.model is None or self._preprocessor is None or self._postprocessor is None:
        if self.model is None:
            msg = "Model is not initialized"
            raise ValueError(msg)

        # processed_batch = self._preprocessor(batch.to(self.device).to_dict())
        # chunk = self.model.predict_action_chunk(processed_batch)
        # return self._postprocessor({ACTION: chunk})[ACTION]

        chunk = self.model.predict_action_chunk(batch.to(self.device).to_dict())
        return chunk[ACTION]

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Args:
            batch: Input batch.
            batch_idx: Batch index (unused, required by Lightning API).

        Returns:
            Loss tensor for backpropagation.
        """
        del batch_idx
        loss, loss_dict = self(batch)

        # Log metrics
        self.log("train/loss", loss_dict["loss"], prog_bar=True)

        return loss

    def validation_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:  # type: ignore[override]
        """Lightning validation step.

        Runs gym-based validation via rollout evaluation. The DataModule's val_dataloader
        returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Index of the batch (used as seed for reproducibility).

        Returns:
            Dictionary of metrics from the gym rollout evaluation.
        """
        return self.evaluate_gym(batch, batch_idx, stage="val")

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler.

        Returns:
            Optimizer configuration dict.
        """
        # Get trainable parameters
        params = [p for p in self.parameters() if p.requires_grad]

        # Create optimizer (use config values)
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.optimizer_lr,
            weight_decay=self.config.optimizer_weight_decay,
            betas=self.config.optimizer_betas,
        )

        warmup_steps = self.config.scheduler_warmup_steps
        drop_steps = self.config.scheduler_decay_steps
        decay_value = self.config.scheduler_decay_lr

        def lr_lambda(step: int) -> float:
            num_drops = step // drop_steps
            decay_factor = decay_value**num_drops
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return decay_factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Configure gradient clipping from policy config.

        This overrides Lightning's default gradient clipping to use
        the policy's grad_clip_norm config value.

        Args:
            optimizer: The optimizer being used.
            gradient_clip_val: Ignored (uses config value instead).
            gradient_clip_algorithm: Ignored (always uses 'norm').
        """
        # Use Trainer's value if set, otherwise fall back to policy config
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.config.optimizer_grad_clip_norm

        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )
