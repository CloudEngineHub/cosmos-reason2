# Troubleshooting

Resources:

* [vLLM Troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/#hangs-loading-a-model-from-disk)

### OpenAI API Connection Error

If you `openai.APIConnectionError: Connection error.`.

Check the server log. Common issues:

1. Server is not fully started. Wait until you see `Application startup complete.`.
1. Server died due to Out of Memory (OOM).
    1. Verify your GPU satisfies the [minimum requirements](../README.md#inference).
    1. Reduce `--max-model-len`. Recommended range: 8192 - 16384.
