<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/collections/nvidia/cosmos-reason2">Hugging Face</a>&nbsp | <a href="https://github.com/nvidia-cosmos/cosmos-cookbook">Cosmos Cookbook</a>
</p>

NVIDIA Cosmos Reason – an open, customizable, reasoning vision language model (VLM) for physical AI and robotics - enables robots and vision AI agents to reason like humans, using prior knowledge, physics understanding and common sense to understand and act in the real world. This model understands space, time, and fundamental physics, and can serve as a planning model to reason what steps an embodied agent might take next.

Cosmos Reason excels at navigating the long tail of diverse scenarios of the physical world with spatial-temporal understanding. Cosmos Reason is post-trained with physical common sense and embodied reasoning data with supervised fine-tuning and reinforcement learning. It uses chain-of-thought reasoning capabilities to understand world dynamics without human annotations.

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [News!](#news)
- [Model Family](#model-family)
- [Setup](#setup)
  - [Virtual Environment](#virtual-environment)
  - [Docker Container](#docker-container)
- [Inference](#inference)
  - [Transformers](#transformers)
  - [Deployment](#deployment)
    - [Online Serving](#online-serving)
    - [Offline Inference](#offline-inference)
- [Quantization](#quantization)
- [Tutorials](#tutorials)
- [Additional Resources](#additional-resources)
- [License and Contact](#license-and-contact)

______________________________________________________________________

<!--TOC-->

## News!
* [December 11, 2025] We have released the Cosmos-Reason2 models and code. The 2B and 8B models are now available. For more details, please check our blog!

## Model Family
* [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
* [Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B)

## Setup

> **This repository only contains documentation/examples/utilities. You do not need it to run inference. See [Inference example](scripts/transformers_sample.py) for a minimal inference example. The following setup instructions are only needed to run the examples in this repository.**

Clone the repository:

```shell
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git
cd cosmos-reason2
```

### Virtual Environment

Install system dependencies:

```shell
sudo apt-get install curl ffmpeg git git-lfs
```

* [uv](https://docs.astral.sh/uv/getting-started/installation/)

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

* [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

```shell
uv tool install -U huggingface_hub
hf auth login
```

Install the repository:

```shell
uv sync
```

### Docker Container

Please make sure you have access to Docker on your machine and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

Build the container:

```bash
image_tag=$(docker build -f Dockerfile -q .)
```

Run the container:

```bash
docker run -it --gpus all \
  --ipc=host \
  --rm \
  -v .:/workspace \
  -v /workspace/.venv \
  -v /workspace/examples/cosmos_rl/.venv \
  -v /root/.cache:/root/.cache \
  $image_tag
```

Optional arguments:

* `--ipc=host`: Use host system's shared memory, since parallel torchrun consumes a large amount of shared memory. If not allowed by security policy, increase `--shm-size` ([documentation](https://docs.docker.com/engine/containers/run/#runtime-constraints-on-resources)).
* `-v /root/.cache:/root/.cache`: Mount host cache to avoid re-downloading cache entries.

## Inference

Minimum requirements:

| Model | GPU Memory |
| --- | --- |
| Cosmos-Reason2-2B | 24GB |
| Cosmos-Reason2-8B | 32GB |

### Transformers

Cosmos-Reason2 is included in [`transformers>=4.57.0`](https://huggingface.co/docs/transformers/en/index).

[Minimal example](scripts/transformers_sample.py) ([sample output](assets/outputs/transformers_sample.log)):

```shell
./scripts/transformers_sample.py
```

### Deployment

For deployment and batch inference, we recommend using [`vllm`](https://docs.vllm.ai/en/stable/).

[Full example](./cosmos_reason2_utils/cosmos_reason2_utils/script/inference.py).

#### Online Serving

Start the [server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/):

```shell
uv run vllm serve nvidia/Cosmos-Reason2-2B \
  --data-parallel-size 1 \
  --async-scheduling \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --allowed-local-media-path "$(pwd)" \
  --port 8000
```

Arguments:

* `--data-parallel-size 1`: Set this to the number of GPUs available to process samples in parallel ([documentation](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)).
* `--async-scheduling`: Enable asynchronous scheduling to increase GPU utilization.
* `--max-model-len 8192`: Maximum model context length to avoid OOM. Recommended range: 8192 - 16384.
* `--reasoning-parser qwen3`: Parse reasoning trace ([documentation](https://docs.vllm.ai/en/stable/features/reasoning_outputs/#quickstart)).
* `--media-io-kwargs '{"video": {"num_frames": -1}}'`: Allow overriding FPS per sample.
* `--port 8000`: Server port. Change if you encounter `Address already in use` errors.

Wait a few minutes for the server to startup. Once complete, it will print `Application startup complete.`. Open a new terminal to run inference commands.

[Minimal example](scripts/vllm_online_sample.py) ([sample output]):

```shell
./scripts/vllm_online_sample.py
```

Caption a video ([sample output](assets/outputs/caption.log)):

```shell
uv run cosmos-reason2-inference online --port 8000 -i prompts/caption.yaml --videos assets/sample.mp4 --fps 4
```

Embodied reasoning and save request to `outputs/embodied_reasoning` for debugging ([sample output](assets/outputs/embodied_reasoning.log)):

```shell
uv run cosmos-reason2-inference online -v --port 8000 -i prompts/embodied_reasoning.yaml --reasoning --images assets/sample.png -o outputs/embodied_reasoning
```

To list available parameters:

```shell
uv run cosmos-reason2-inference online --help
```

Arguments:

* `--model nvidia/Cosmos-Reason2-2B`: Model name or path.

#### Offline Inference

vLLM provides [offline batch processing](https://docs.vllm.ai/en/latest/examples/offline_inference/openai_batch/), which inherits the same arguments as the online server:

Temporally caption a video and save the input frames to `outputs/temporal_localization` for debugging ([sample output](assets/outputs/temporal_localization.log)):

```shell
# Generate request.jsonl
uv run cosmos-reason2-inference offline -v --max-model-len 8192 -i prompts/temporal_localization.yaml --videos assets/sample.mp4 --fps 4 -o outputs/requests.jsonl

# Offline batch inference
uv run vllm run-batch --model nvidia/Cosmos-Reason2-2B \
  --data-parallel-size 1 \
  --async-scheduling \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --allowed-local-media-path "$(pwd)" \
  -i outputs/requests.jsonl \
  -o outputs/results.jsonl
```

To list available parameters:

```shell
uv run cosmos-reason2-inference offline --help
```

## Quantization

For model quantization, we recommend using [llmcompressor](https://github.com/vllm-project/llm-compressor)

[Example](scripts/quantize.py) ([sample output](assets/outputs/quantize.log)):

```shell
./scripts/quantize.py -o /tmp/cosmos-reason2/checkpoints --model nvidia/Cosmos-Reason2-2B --precision fp4
```

To list available parameters:

```shell
./scripts/quantize.py --help
```

## Tutorials

* Post-Training
  * [Cosmos-RL](examples/cosmos_rl/README.md)

## Additional Resources

* [Example prompts](prompts/README.md)
* Cosmos-Reason2 is based on the Qwen3-VL architecture.
  * [Qwen3-VL Repository](https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#quickstart)
    * More inference examples.
  * [Qwen3-VL transformers](https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl)
  * [Qwen3-VL vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
  * [Qwen3 Documentation](https://qwen.readthedocs.io/en/latest/)
    * Many useful third-party tools for deployment/quantization/post-training.
* vLLM
  * [Online Serving](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
  * [Offline Batch Inference](https://docs.vllm.ai/en/latest/examples/offline_inference/openai_batch/)
  * [Multimodal Inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/#online-serving)
  * [LoRA](https://docs.vllm.ai/en/stable/features/lora/#serving-lora-adapters)
  * [Parallel Deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
