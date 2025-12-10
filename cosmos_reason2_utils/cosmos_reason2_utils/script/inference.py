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

"""Inference using vLLM."""

# Sources
# * https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#deployment

import json
import pathlib

from cosmos_reason2_utils.init import init_script

init_script()

import collections
import textwrap
import time
from functools import cached_property
from typing import Annotated

import openai
import pydantic
import tyro
import vllm
import yaml
from rich import print
from rich.pretty import pprint
from typing_extensions import assert_never

from cosmos_reason2_utils.online import (
    OnlineVisionConfig,
    create_online_body,
    create_online_conversation,
)
from cosmos_reason2_utils.text import (
    REASONING_PROMPT,
    SYSTEM_PROMPT,
    SamplingOverrides,
)

SEPARATOR = "-" * 20

DEFAULT_MODEL = "nvidia/Cosmos-Reason2-2B"
"""Default model name."""


def pprint_dict(d: dict, name: str):
    """Pretty print a dictionary."""
    pprint(collections.namedtuple(name, d.keys())(**d), expand_all=True)


class InputConfig(pydantic.BaseModel):
    """Prompt config."""

    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    user_prompt: str = ""
    """User prompt."""
    system_prompt: str = pydantic.Field(default=SYSTEM_PROMPT)
    """System prompt."""
    sampling_params: dict = pydantic.Field(default_factory=dict)
    """Override sampling parameters."""


class Args(pydantic.BaseModel):
    """Inference arguments."""

    model_config = pydantic.ConfigDict(
        extra="forbid", use_attribute_docstrings=True, frozen=True
    )

    input_file: Annotated[pydantic.FilePath | None, tyro.conf.arg(aliases=("-i",))] = (
        None
    )
    """Path to input yaml file."""
    prompt: str | None = None
    """User prompt."""
    images: list[str] = pydantic.Field(default_factory=list)
    """Image paths or URLs."""
    videos: list[str] = pydantic.Field(default_factory=list)
    """Video paths or URLs."""
    reasoning: bool = False
    """Enable reasoning trace."""
    sampling: SamplingOverrides = SamplingOverrides()
    """Sampling parameters."""

    verbose: Annotated[bool, tyro.conf.arg(aliases=("-v",))] = False
    """Verbose output"""

    @cached_property
    def input_config(self) -> InputConfig:
        if self.input_file is not None:
            input_kwargs = yaml.safe_load(open(self.input_file, "rb"))
            return InputConfig.model_validate(input_kwargs)
        else:
            return InputConfig()

    @cached_property
    def system_prompt(self) -> str:
        return self.input_config.system_prompt

    @cached_property
    def user_prompt(self) -> str:
        if self.prompt:
            user_prompt = self.prompt
        else:
            user_prompt = self.input_config.user_prompt
        if not user_prompt:
            raise ValueError("No user prompt provided.")
        user_prompt = user_prompt.strip()
        if self.reasoning:
            user_prompt += f"\n\n{REASONING_PROMPT}"
        return user_prompt

    @cached_property
    def sampling_kwargs(self) -> dict:
        sampling_kwargs = SamplingOverrides.get_defaults(reasoning=self.reasoning)
        sampling_kwargs.update(self.input_config.sampling_params)
        sampling_kwargs.update(self.sampling.model_dump(exclude_none=True))
        vllm.SamplingParams(**sampling_kwargs)
        return sampling_kwargs

    @cached_property
    def sampling_params(self) -> vllm.SamplingParams:
        return vllm.SamplingParams(**self.sampling_kwargs)


class OnlineArgs(Args):
    """Online inference arguments."""

    host: str = "localhost"
    """Server hostname."""
    port: int = 8000
    """Server port."""
    model: str | None = None
    """Model name (https://huggingface.co/collections/nvidia/cosmos-reason2).
    
    If not provided, the first model in the server will be used.
    """

    vision: OnlineVisionConfig = OnlineVisionConfig()
    """Vision config."""


class OfflineArgs(Args):
    """Offline inference arguments."""

    output_file: Annotated[pathlib.Path | None, tyro.conf.arg(aliases=("-o",))] = None
    """Output jsonl file."""

    append: bool = False
    """Append to output file."""

    model: str | None = None
    """Model name (https://huggingface.co/collections/nvidia/cosmos-reason2).
    
    If not provided, the first model in the server will be used.
    """

    vision: OnlineVisionConfig = OnlineVisionConfig()
    """Vision config."""


def online_inference(args: OnlineArgs):
    """Send sample to OpenAI API server."""
    # Create client
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url=f"http://{args.host}:{args.port}/v1",
    )
    models = client.models.list()
    if args.model is not None:
        model = client.models.retrieve(args.model)
    else:
        model = models.data[0]
    if args.verbose:
        pprint(model, expand_all=True)

    # Create conversation
    conversation = create_online_conversation(
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        images=args.images,
        videos=args.videos,
    )
    if args.verbose:
        pprint(conversation, expand_all=True)

    # Process inputs
    body, extra_body = create_online_body(
        sampling_params=args.sampling_params,
        vision_config=args.vision,
        max_model_len=model.max_model_len,
    )

    # Run inference
    chat_completion = client.chat.completions.create(
        model=model.id,
        messages=conversation,
        **body,
        extra_body=extra_body,
    )

    for output in chat_completion.choices:
        if args.verbose:
            pprint(output, expand_all=True)
        if output.message.reasoning_content:
            print(SEPARATOR)
            print("Reasoning:")
            print(textwrap.indent(output.message.reasoning_content.strip(), "  "))
        print(SEPARATOR)
        print("Assistant:")
        print(textwrap.indent(output.message.content.strip(), "  "))
        print(SEPARATOR)


def offline_inference(args: OfflineArgs):
    """Save sample to OpenAI API batch file."""
    messages = create_online_conversation(
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        images=args.images,
        videos=args.videos,
    )
    body, extra_body = create_online_body(
        sampling_params=args.sampling_params,
        vision_config=args.vision,
        max_model_len=args.max_model_len,
    )
    request = {
        "custom_id": "request-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body
        | {
            "model": args.model,
            "messages": messages,
        },
        "extra_body": extra_body,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.open("a" if args.append else "w").write(json.dumps(request) + "\n")
    print(f"Saved request to '{args.output_file}'")


def inference(args: OfflineArgs | OnlineArgs):
    print(SEPARATOR)
    print("System:")
    print(textwrap.indent(args.system_prompt.strip(), "  "))
    print(SEPARATOR)
    print("User:")
    print(textwrap.indent(args.user_prompt.strip(), "  "))
    print(SEPARATOR)
    if args.verbose:
        pprint_dict(args.sampling_kwargs, "SamplingParams")

    start = time.time()
    if isinstance(args, OfflineArgs):
        offline_inference(args)
    elif isinstance(args, OnlineArgs):
        online_inference(args)
    else:
        assert_never(args)
    elapsed = time.time() - start
    print(f"Inference time: {elapsed:.2f} seconds")


def main():
    args = tyro.cli(
        OfflineArgs | OnlineArgs,
        description=__doc__,
        config=(tyro.conf.OmitArgPrefixes,),
    )
    inference(args)


if __name__ == "__main__":
    main()
