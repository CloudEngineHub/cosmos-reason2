# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vision processing utilities."""

import os
from pathlib import Path

import numpy as np
import pydantic
import torch
from PIL import Image

from cosmos_reason2_utils.text import (
    SamplingOverrides,
)


# Source: https://huggingface.co/nvidia/Cosmos-Reason2-2B/blob/main/video_preprocessor_config.json
_PATCH_SIZE = 16
_MERGE_SIZE = 2
_PATCH_FACTOR = _PATCH_SIZE * _MERGE_SIZE
PIXELS_PER_TOKEN = _PATCH_FACTOR**2
"""Number of pixels per visual token."""

MAX_MODEL_LEN = 16384
"""Maximum model context length."""
# 64 visual tokens = 65536 = 64 * 32**2
SHORTEST_EDGE = 65536
# 16777216 = 16384 * 32**2
LONGEST_EDGE = 16777216

class VisionSizeConfig(pydantic.BaseModel):
    """Config for vision size."""

    model_config = pydantic.ConfigDict(extra="allow", use_attribute_docstrings=True)
    
    shortest_edge: int | None = None
    """Min total pixels of the image/video."""
    longest_edge: int | None = None
    """Max total pixels of the image/video."""

class VisionConfig(pydantic.BaseModel):
    """Config for vision processing.

    Source: https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor

    Attributes are sorted by priority. Higher priority attributes override lower
    priority attributes.
    """

    model_config = pydantic.ConfigDict(extra="allow", use_attribute_docstrings=True)

    num_frames: int | None = None
    """Maximum number of frames to sample when do_sample_frames=True."""
    fps: float | None = None
    """FPS of the video."""
    do_sample_frames: bool = True
    """Whether to sample frames from the video before processing or to process the whole video."""
    size: VisionSizeConfig | None = None
    """Size of the image/video."""

def create_vision_kwargs(
    *,
    sampling_params: SamplingOverrides,
    vision_config: VisionConfig,
    max_model_len: int | None,
) -> dict:
    """Create vision processor kwargs.

    Args:
        sampling_params: Sampling parameters.
        vision_config: Vision config.
        max_model_len: Maximum model length.

    Returns:
        kwargs: Keyword arguments for transformers processor.
    """

    # Compute min pixels
    if vision_config.size.shortest_edge:
        shortest_edge = vision_config.size.shortest_edge
    else:
        shortest_edge = 0

    # Compute total pixels
    if not max_model_len:
        max_model_len = MAX_MODEL_LEN
    assert sampling_params.max_tokens
    if max_model_len < sampling_params.max_tokens:
        raise ValueError("Max model length must be greater than max tokens.")
    max_seq_len = max_model_len - sampling_params.max_tokens
    longest_edge = int(max_seq_len * PIXELS_PER_TOKEN * 0.9)
    if vision_config.size.longest_edge:
        if vision_config.size.longest_edge > longest_edge:
            raise ValueError(
                f"Longest edge {vision_config.size.longest_edge} exceeds limit {longest_edge}."
            )
        longest_edge = vision_config.size.longest_edge
    
    vision_kwargs = vision_config.model_dump(exclude_none=True)
    vision_kwargs["size"] = {
        "shortest_edge": shortest_edge,
        "longest_edge": longest_edge,
    }
    return vision_kwargs


def _tensor_to_pil_images(tensor: torch.Tensor) -> list[Image.Image]:
    """Convert a tensor to a list of PIL images.

    Args:
        tensor: Tensor with shape (C, H, W), (C, T, H, W) or (T, C, H, W)

    Returns:
        List of PIL images
    """
    # Check tensor shape and convert if needed
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.shape[0] == 3:  # (C, T, H, W)
        if tensor.shape[1] == 3:
            raise ValueError(f"Ambiguous shape: {tensor.shape}")
        # Convert to (T, C, H, W)
        tensor = tensor.permute(1, 0, 2, 3)

    # Convert to numpy array with shape (T, H, W, C)
    frames = tensor.permute(0, 2, 3, 1).cpu().numpy()

    # Ensure values are in the right range for PIL (0-255, uint8)
    if np.issubdtype(frames.dtype, np.floating):
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)

    return [Image.fromarray(frame) for frame in frames]


def save_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a tensor as images to a directory.

    Args:
        tensor: Tensor with shape (C, H, W) or (T, C, H, W)
        path: Directory to save the images
    """
    os.makedirs(path, exist_ok=True)
    images = _tensor_to_pil_images(tensor)
    for i, image in enumerate(images):
        image.save(f"{path}/{i}.png")
