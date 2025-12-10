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

"""OpenAI API utilities."""

import os
from typing import Any

from cosmos_reason2_utils.text import (
    SYSTEM_PROMPT,
    SamplingOverrides,
    create_mm_processor_kwargs,
)
from cosmos_reason2_utils.vision import VisionConfig

def create_openai_conversation(
    *,
    user_prompt: str = "",
    response: str = "",
    system_prompt: str = SYSTEM_PROMPT,
    images: list[Any] | None = None,
    videos: list[Any] | None = None,
) -> list[dict]:
    """Create chat conversation for OpenAI API.

    Specification: https://platform.openai.com/docs/api-reference/messages

    Args:
        system_prompt: System prompt.
        user_prompt: User prompt.
        response: Assistant response.
        images: List of images.
        videos: List of videos.

    Returns:
        conversation: Chat conversation.
    """
    user_content = []
    if images is not None:
        for image in images:
            user_content.append(
                {"type": "image_url", "image_url": {"url": _get_media_url(image)}}
            )
    if videos is not None:
        for video in videos:
            if isinstance(video, dict):
                user_content.append({"type": "video", "video": video["frame_list"]})
            else:
                user_content.append(
                    {"type": "video_url", "video_url": {"url": _get_media_url(video)}}
                )
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": user_content})
    if response:
        conversation.append({"role": "assistant", "content": response})
    return conversation


def create_openai_body(
    *,
    sampling_params: SamplingOverrides,
    vision_config: VisionConfig,
    max_model_len: int | None,
) -> tuple[dict, dict]:
    """Create body and extra body for OpenAI API.
    
    
    """
    sampling_kwargs = sampling_params.model_dump(exclude_none=True)
    mm_processor_kwargs = create_mm_processor_kwargs(
        sampling_params=sampling_params,
        vision_config=vision_config,
        max_model_len=max_model_len,
    )
    body = {
        "max_completion_tokens": sampling_params.max_tokens,
    }
    extra_body = sampling_kwargs | {"mm_processor_kwargs": mm_processor_kwargs}
    return body, extra_body


def _get_media_url(path: str) -> str:
    if ":" in path:
        return path
    path = os.path.abspath(path)
    return f"file://{path}"
