# AI providers

DocAssist finds the answer in your documents itself; an AI model only **rewords the passages it finds** into a readable reply. Superadmins choose which models do that, and in what order, on **Platform → AI providers** (`/platform/ai/`).

## The answer chain

Each question tries the enabled providers in order. If one fails (no key, quota reached, timeout, wrong model name), the next one answers. If every provider fails, DocAssist **quotes the 2–3 sentences from the documents that best match the question**, with file and page, so people always get an answer.

```text
1. Google Gemini  →  2. Groq  →  3. Claude  →  …  →  last: quote the documents
```

- An organization with **Use external AI providers** switched off (its Settings, AI model tab) skips every cloud provider and uses only providers on your own server (Ollama), then quotes.
- The quoting step can be turned off in Platform settings (*Documents & answers*). Then a total failure shows "AI service unavailable" and the API returns 503.
- Admins see when a backup answered: Home lists "Gemini quota reached: N answers came from the backup", and the sidebar meter shows **Using backup**.
- Answers written by a backup are never reused from the answer cache, so the next ask gets the full provider again once it recovers.

## Supported providers

| Provider | How it's called | Starting model (check the provider's current list) | Key from |
| --- | --- | --- | --- |
| Google Gemini | Google's SDK | `gemini-flash-latest` (follows Google's current Flash model) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey); blank uses `GOOGLE_API_KEY` from `.env` |
| OpenAI (ChatGPT) | OpenAI chat API | `gpt-4o-mini` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| Anthropic Claude | Messages API | `claude-haiku-4-5-20251001` | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| Groq | OpenAI-compatible | `llama-3.3-70b-versatile` | [console.groq.com/keys](https://console.groq.com/keys) |
| DeepSeek | OpenAI-compatible | `deepseek-chat` | [platform.deepseek.com](https://platform.deepseek.com/api_keys) |
| OpenRouter | OpenAI-compatible | `meta-llama/llama-3.3-70b-instruct` | [openrouter.ai/keys](https://openrouter.ai/keys) |
| Local model (Ollama) | OpenAI-compatible, on your server | `llama3.2:3b` | No key; install [Ollama](https://ollama.com/download) and `ollama pull llama3.2:3b` |
| Other | Any OpenAI-compatible API | your choice | depends |

Model names change often. The **Test** button tells you straight away if a name or key is wrong.

## Adding a provider

1. **AI providers → Add provider**, and choose the service. Its address, a starting model and a *Get a key* link fill in.
2. Paste the API key. Change the model if you want a different one.
3. **Save and test.** A tiny request ("Reply with the single word OK") checks it; the result shows in the list.
4. Use ↑ and ↓ to set its place in the order.

Other actions: **Edit** (leave the key blank to keep the saved one, or tick *Remove the saved key*), **Disable/Enable**, and **Remove** (deletes the saved key).

## Keys and privacy

- Keys are encrypted before they're stored (Fernet, with a key derived from `DJANGO_SECRET_KEY`) and only shown as "•••• last four characters". They're never written to the audit log. If `DJANGO_SECRET_KEY` changes, the page asks you to re-enter the keys.
- What a provider receives per question: the organization's instructions, the top few passages (with file and page), the last few conversation turns, and the question. Never whole documents.
- A local Ollama model keeps everything on your server, but needs a strong machine: on a 2-core laptop expect 30–60 seconds per answer; on an 8-core server or a GPU, a few seconds.

## Good setups

- **Reliable and cheap:** Gemini first, then Groq or OpenRouter (both have free tiers) as backup, then quoting.
- **Nothing leaves the server:** only an Ollama provider, or switch off *Use external AI providers* per organization.
- **Highest quality:** Claude or OpenAI first, Gemini as backup.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Test fails with `401` or `403` | Wrong or revoked key | Paste a fresh key |
| Test fails with `404` / "model not found" | Model name changed or misspelled | Copy the exact name from the provider's model list |
| Test fails with `429` | Quota or rate limit | Wait, upgrade the plan, or move another provider first |
| "Can't reach …" | Network, firewall, or Ollama not running | Check the address; for Ollama run `ollama serve` |
| Answers end mid-sentence | Output limit too low for a "thinking" model | Choose *Detailed* answer length, or another model |

## For developers

The clients live in `django_app/rag/llm.py` (Gemini SDK, an OpenAI-compatible HTTP client, and an Anthropic Messages client, using only the Python standard library for HTTP). The chain is in `rag/pipeline.py` (`providers`, `writers`, `answer_question`, `stream_answer`); quoting is `rag/extractive.py`. The model is `portal.models.AIProvider`; presets are `portal.models.PRESETS`.
