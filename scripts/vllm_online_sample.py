#!/usr/bin/env -S uv run --script
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

"""vLLM online inference sample."""

# Source https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#online-serving

from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).parents[1]
SEPARATOR = "-" * 20

PIXELS_PER_TOKEN = 32**2
"""Number of pixels per visual token."""


def main():
    # Create client
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )
    model = client.models.list().data[0]

    # Limit tokens to fit in model length
    min_vision_tokens = 256
    max_new_tokens = 4096
    max_vision_tokens = int((model.max_model_len - max_new_tokens) * 0.9)

    # Create inputs
    # IMPORTANT: Media is listed before text to match training inputs
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": f"file://{ROOT}/assets/sample.mp4"},
                },
                {"type": "text", "text": "Caption the video in detail."},
            ],
        },
    ]

    # Run inference
    response = client.chat.completions.create(
        model=model.id,
        messages=conversation,
        extra_body={
            "mm_processor_kwargs": {
                "fps": 4,
                "do_sample_frames": True,
                "size": {
                    "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
                    "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
                },
            }
        },
        max_tokens=max_new_tokens,
    )
    print(SEPARATOR)
    print(response.choices[0].message.content)
    print(SEPARATOR)


if __name__ == "__main__":
    main()
