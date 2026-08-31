# Providers

Any OpenAI- or Anthropic-compatible endpoint works: set `base_url` on the
client, or pass `provider=` to [`wildedge.llm_api()`](llm_api.md). The host is
mapped to a `model_source` name on every event.

```python
OpenAI(base_url="https://api.fireworks.ai/inference/v1")
```

## Recognized providers

| Provider | Base URL | `model_source` |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `openai` |
| Anthropic | `https://api.anthropic.com` | `anthropic` |
| Azure OpenAI | `https://<resource>.openai.azure.com/openai/v1` | `azure-openai` |
| OpenRouter | `https://openrouter.ai/api/v1` | `openrouter` |
| Fireworks AI | `https://api.fireworks.ai/inference/v1` | `fireworks` |
| Together AI | `https://api.together.xyz/v1` | `together` |
| Baseten (Model APIs) | `https://inference.baseten.co/v1` | `baseten` |
| Baseten (dedicated) | `https://model-<id>.api.baseten.co/environments/production/sync/v1` | `baseten` |
| xAI | `https://api.x.ai/v1` | `xai` |
| Mistral | `https://api.mistral.ai/v1` | `mistral` |
| Groq | `https://api.groq.com/openai/v1` | `groq` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek` |
| Cerebras | `https://api.cerebras.ai/v1` | `cerebras` |
| Perplexity | `https://api.perplexity.ai` | `perplexity` |
| NVIDIA NIM | `https://integrate.api.nvidia.com/v1` | `nvidia` |
| Hugging Face | `https://router.huggingface.co/v1` | `huggingface` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `google` |

Azure OpenAI and dedicated Baseten deployments match by domain suffix, so every
resource and deployment subdomain resolves to the same name.

## Other hosts

Unrecognized hosts are recorded verbatim, so self-hosted runtimes (vLLM, Ollama,
LM Studio, llama.cpp) all collapse onto `localhost` or `127.0.0.1`. Name them:

```python
wildedge.llm_api(model="gemma-7b", provider="ollama")
wildedge.register_model(client, source="ollama", model_id="gemma-7b")
```

## Caveats

- `tokens_in` / `tokens_out` are populated by every provider above.
- `cached_input_tokens` and `reasoning_tokens_out` are OpenAI extensions. Most
  providers omit them; they record as `None`, not `0`.
- Streamed responses carry no usage unless you pass
  `stream_options={"include_usage": True}`. Timing and TTFT are unaffected.
