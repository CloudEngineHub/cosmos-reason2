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

from typing import Any

import pydantic

"""Text processing utilities."""

SYSTEM_PROMPT = "You are a helpful assistant."
"""Default system prompt."""

REASONING_PROMPT = """Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag."""
"""Reasoning addon prompt."""

# Copied from [vllm.SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html).
# Ideally, we could auto-generate this class.
class SamplingOverrides(pydantic.BaseModel):
    """Sampling parameters for text generation."""

    model_config = pydantic.ConfigDict(extra="allow", use_attribute_docstrings=True)

    n: int | None = None
    """Number of outputs to return for the given prompt request."""
    presence_penalty: float | None = None
    """Penalizes new tokens based on whether they appear in the generated text
    so far. Values > 0 encourage the model to use new tokens, while values < 0
    encourage the model to repeat tokens."""
    repetition_penalty: float | None = None
    """Penalizes new tokens based on whether they appear in the prompt and the
    generated text so far. Values > 1 encourage the model to use new tokens,
    while values < 1 encourage the model to repeat tokens."""
    temperature: float | None = None
    """Controls the randomness of the sampling. Lower values make the model
    more deterministic, while higher values make the model more random. Zero
    means greedy sampling."""
    top_p: float | None = None
    """Controls the cumulative probability of the top tokens to consider. Must
    be in (0, 1]. Set to 1 to consider all tokens."""
    top_k: int | None = None
    """Controls the number of top tokens to consider. Set to 0 (or -1) to
    consider all tokens."""
    seed: int | None = None
    """Random seed to use for the generation."""
    max_tokens: int | None = None
    """Maximum number of tokens to generate per output sequence."""

    @classmethod
    def get_defaults(cls, *, reasoning: bool = False) -> dict:
        kwargs = dict(
            max_tokens=4096,
        )
        # Source: https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#evaluation-reproduction
        if reasoning:
            return kwargs | dict(
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                temperature=0.6,
                seed=1234,
            )
        else:
            return kwargs | dict(
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.0,
                presence_penalty=1.5,
                temperature=0.7,
                seed=3407,
            )

def create_conversation(
    *,
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    response: str = "",
    images: list[Any] | None = None,
    videos: list[Any] | None = None,
) -> list[dict]:
    """Create chat conversation for transformers.

    Args:
        user_prompt: User prompt.
        system_prompt: System prompt.
        response: Assistant response.
        images: List of images.
        videos: List of videos.

    Returns:
        conversation: Chat conversation.
    """
    user_content = []
    if images is not None:
        for image in images:
            user_content.append({"type": "image", "image": image})
    if videos is not None:
        for video in videos:
            user_content.append({"type": "video", "video": video})
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": user_content})
    if response:
        conversation.append({"role": "assistant", "content": response})
    return conversation
