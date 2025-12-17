# Notebooks

This directory contains a collection of Jupyter notebooks.

| Notebook | Description | Open in Colab |
| --- | --- | --- |
| [`trl_sft.ipynb`](./trl_sft.ipynb) | Supervised Fine-Tuning (SFT) with QLoRA using TRL | [![Open In Colab](https://colab.research.google.com/github/nvidia-cosmos/cosmos-reason2/blob/main/examples/notebooks/trl_sft.ipynb) |
| [`trl_grpo.ipynb`](./trl_grpo.ipynb) | GRPO with QLoRA using TRL | [![Open In Colab](https://colab.research.google.com/github/nvidia-cosmos/cosmos-reason2/blob/main/examples/notebooks/trl_grpo.ipynb) |

## Local

The notebooks can be run locally using [VS Code](https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-from-vs-code).

Prerequisites:

* [Setup](../../README.md#setup)

Install the package:

```shell
cd examples/notebooks
uv sync
```

Optionally, open the notebooks in a new window (avoids having to select the correct virtual environment):

```shell
code examples/notebooks
```

## Contributing

The `.ipynb` files are auto-generated, do not edit them! Notebooks are written in [jupytext py:percent](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-percent-format) format. To update `.ipynb` notebooks, run:

```shell
just notebooks-sync
```
