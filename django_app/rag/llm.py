"""Calls to the answer-writing models, whole or streamed.

Three ways of calling cover every provider a superadmin can add in the portal:
Gemini through Google's SDK; the OpenAI chat-completions API, which OpenAI,
Groq, DeepSeek, OpenRouter and Ollama all speak; and Anthropic's Messages API
for Claude. The OpenAI and Anthropic calls use plain HTTPS, with no extra SDKs.
"""
import json
import urllib.error
import urllib.request

from django.conf import settings


class LLMError(Exception):
    pass


# Gemini's current Flash models "think" before answering, and those hidden tokens
# count against max_output_tokens: with a 600 cap they used ~580 and cut answers
# off after a few words. This headroom is for thinking; the answer length itself
# is steered by the prompt's length rules.
GEMINI_THINKING_HEADROOM = 1536
ANTHROPIC_VERSION = "2023-06-01"


def _max_tokens(max_output_tokens):
    return max_output_tokens or settings.RAG_MAX_OUTPUT_TOKENS


def _temperature(temperature):
    return settings.RAG_TEMPERATURE if temperature is None else temperature


def _echo(prompt, model):
    question = prompt.rsplit("Question:", 1)[-1].strip()
    return f"[echo:{model}] {question}"


def _echo_stream(prompt, model):
    for word in _echo(prompt, model).split(" "):
        yield word + " "


# --- Google Gemini (SDK) ---------------------------------------------------------

def _generation_config(max_output_tokens=None, temperature=None):
    # Answers are short and grounded, so a modest cap keeps them fast: the model's
    # time grows with the length of what it writes. Organizations can change both.
    return {
        "max_output_tokens": _max_tokens(max_output_tokens) + GEMINI_THINKING_HEADROOM,
        "temperature": _temperature(temperature),
    }


def _model(model, api_key="", max_output_tokens=None, temperature=None):
    key = api_key or settings.GOOGLE_API_KEY
    if not key:
        raise LLMError("No Gemini API key: add one to the provider or set GOOGLE_API_KEY.")
    import google.generativeai as genai

    genai.configure(api_key=key)
    return genai.GenerativeModel(model, generation_config=_generation_config(max_output_tokens, temperature))


def generate(prompt, model, api_key="", **options):
    """options: max_output_tokens, temperature (platform defaults when left out)."""
    if settings.RAG_LLM_BACKEND == "echo":
        return _echo(prompt, model)
    try:
        return _model(model, api_key, **options).generate_content(prompt).text
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(str(exc)) from exc


def generate_stream(prompt, model, api_key="", **options):
    """Yield the answer in pieces as the model writes it."""
    if settings.RAG_LLM_BACKEND == "echo":
        yield from _echo_stream(prompt, model)
        return
    try:
        for chunk in _model(model, api_key, **options).generate_content(prompt, stream=True):
            try:
                text = chunk.text
            except ValueError:  # a chunk with no text (e.g. safety metadata)
                continue
            if text:
                yield text
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(str(exc)) from exc


# --- HTTP helpers ------------------------------------------------------------------

def _post(url, headers, body, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        try:
            error = json.loads(detail).get("error")
            detail = error.get("message", detail) if isinstance(error, dict) else (error or detail)
        except (ValueError, AttributeError):
            pass
        raise LLMError(f"{exc.code} {detail}".strip()) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise LLMError(f"Can't reach {url.split('/')[2] if '//' in url else url}: {exc}") from exc


def _sse_data(response):
    """The JSON payloads of a server-sent-events stream ("data: {...}" lines)."""
    for raw in response:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


# --- OpenAI-compatible: OpenAI, Groq, DeepSeek, OpenRouter, Ollama ------------------

def _chat_request(base_url, api_key, model, prompt, timeout, stream, max_output_tokens=None, temperature=None):
    headers = {"X-Title": "DocAssist"}  # OpenRouter shows this as the app name
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": _max_tokens(max_output_tokens),
        "temperature": _temperature(temperature),
        "stream": stream,
    }
    return _post(f"{base_url}/chat/completions", headers, body, timeout)


def chat_generate(prompt, model, base_url, api_key="", timeout=60, **options):
    if settings.RAG_LLM_BACKEND == "echo":
        return _echo(prompt, model)
    try:
        with _chat_request(base_url, api_key, model, prompt, timeout, False, **options) as response:
            data = json.loads(response.read())
        return data["choices"][0]["message"]["content"] or ""
    except LLMError:
        raise
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected reply: {exc}") from exc


def chat_stream(prompt, model, base_url, api_key="", timeout=60, **options):
    if settings.RAG_LLM_BACKEND == "echo":
        yield from _echo_stream(prompt, model)
        return
    response = _chat_request(base_url, api_key, model, prompt, timeout, True, **options)
    try:
        with response:
            for data in _sse_data(response):
                if data.get("error"):
                    raise LLMError(str(data["error"].get("message", data["error"])))
                choices = data.get("choices") or [{}]
                text = (choices[0].get("delta") or {}).get("content")
                if text:
                    yield text
    except LLMError:
        raise
    except (OSError, ValueError, AttributeError) as exc:
        raise LLMError(f"Stream failed: {exc}") from exc


# --- Anthropic Claude (Messages API) --------------------------------------------------

def _claude_request(base_url, api_key, model, prompt, timeout, stream, max_output_tokens=None, temperature=None):
    if not api_key:
        raise LLMError("No Anthropic API key.")
    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
    body = {
        "model": model,
        "max_tokens": _max_tokens(max_output_tokens),
        "temperature": _temperature(temperature),
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }
    return _post(f"{base_url}/messages", headers, body, timeout)


def claude_generate(prompt, model, base_url, api_key="", timeout=60, **options):
    if settings.RAG_LLM_BACKEND == "echo":
        return _echo(prompt, model)
    try:
        with _claude_request(base_url, api_key, model, prompt, timeout, False, **options) as response:
            data = json.loads(response.read())
        return "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    except LLMError:
        raise
    except (OSError, ValueError, AttributeError) as exc:
        raise LLMError(f"Unexpected reply: {exc}") from exc


def claude_stream(prompt, model, base_url, api_key="", timeout=60, **options):
    if settings.RAG_LLM_BACKEND == "echo":
        yield from _echo_stream(prompt, model)
        return
    response = _claude_request(base_url, api_key, model, prompt, timeout, True, **options)
    try:
        with response:
            for data in _sse_data(response):
                if data.get("type") == "error":
                    raise LLMError(str(data.get("error", {}).get("message", "Claude returned an error")))
                if data.get("type") == "content_block_delta":
                    text = data.get("delta", {}).get("text")
                    if text:
                        yield text
                elif data.get("type") == "message_stop":
                    return
    except LLMError:
        raise
    except (OSError, ValueError, AttributeError) as exc:
        raise LLMError(f"Stream failed: {exc}") from exc


# --- One provider, ready to call -------------------------------------------------------

def writer(provider, model, **options):
    """(write(prompt), stream(prompt)) for a portal.models.AIProvider. options: max_output_tokens, temperature."""
    key = provider.api_key
    if provider.style == "gemini":
        return (lambda p: generate(p, model, key, **options),
                lambda p: generate_stream(p, model, key, **options))
    call = {"base_url": provider.effective_base_url, "api_key": key, "timeout": provider.timeout, **options}
    if provider.style == "anthropic":
        return (lambda p: claude_generate(p, model, **call), lambda p: claude_stream(p, model, **call))
    return (lambda p: chat_generate(p, model, **call), lambda p: chat_stream(p, model, **call))
