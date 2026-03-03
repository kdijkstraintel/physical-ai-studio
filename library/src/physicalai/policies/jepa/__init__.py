# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA Policy - HuggingFace's flow matching VLA model."""

from .config import JEPAConfig
from .model import JEPAModel
from .policy import JEPA

__all__ = ["JEPA", "JEPAModel", "JEPAConfig"]
