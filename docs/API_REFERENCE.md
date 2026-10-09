# API reference

The public REST API lets another system ask questions about an organization's documents. It's served by the platform at `/api/v1/`.

- **Base URL:** `http://localhost:8001/api/v1/` in Docker (`http://127.0.0.1:8123/api/v1/` for the local dev server).
- **Authentication:** an organization API key, created by an admin on **API keys** (`/keys/`). Send it as `Authorization: Bearer sk_…` or `X-API-Key: sk_…`. A key only ever reaches its own organization's documents.
- **Try it in the browser:** the **Developer** page (`/developer/`) sends real requests, shows the response, and generates cURL, PowerShell, Python and JavaScript code.
- **Format:** JSON in and out. Errors always look like `{"success": false, "error": "...", "detail": "..."}`.

## Endpoints

| Method | Path | Key | Purpose |
| --- | --- | --- | --- |
| POST | `chat` | yes | Ask a question |
| GET | `documents` | yes | List the organization's ready documents |
| GET | `validate-key` | yes | Check a key |
| GET | `health` | optional | Service health; with a valid key, adds document counts |

### `POST /api/v1/chat`

Request:

```json
{ "question": "What support does Whitebox give innovators?", "session_id": "optional" }
```

- `question` (required, up to 4,000 characters). `query` is accepted instead, for old clients.
- `session_id` (optional): the value from a previous response, to continue that conversation so follow-up questions have context. Leave it out to start a new conversation.

Response `200`:

```json
{
  "success": true,
  "answer": "WhiteBox supports innovators through market linkages, incubation …",
  "session_id": "3f1c0d6e-…",
  "sources": ["doc2.pdf"],
  "citations": [{ "document_id": 2, "filename": "doc2.pdf", "page": 10 }],
  "timestamp": "2026-10-05T08:41:35.120Z"
}
```

`citations` lists every file and page used; `sources` is the list of file names only. The API always returns sources, even when an organization hides them in its web chat.

If every AI provider fails, the answer quotes the best-matching sentences from the documents, still with `200` and citations. Only when quoting is turned off on the platform does a total failure return `503`.

### `GET /api/v1/documents`

```json
{ "success": true, "documents": [
  { "id": 2, "filename": "doc2.pdf", "pages": 14, "chunks": 28, "size": 507392, "uploaded_at": "2026-10-01T10:02:11Z" }
] }
```

Paused documents are left out, because answers don't use them.

### `GET /api/v1/validate-key`

`{"success": true, "message": "API key is valid", "organization": "ICTA"}`, or `401`.

### `GET /api/v1/health`

`{"status": "healthy", "timestamp": "…", "version": "…"}`. With a valid key it also returns `documents_loaded` and `documents_count`. Docker uses this endpoint for its health check.

## Status codes

| Code | Meaning | What to do |
| --- | --- | --- |
| `200` | Success | |
| `400` | Bad input (missing or too long question, invalid JSON) | Fix the request |
| `401` | Missing, unknown or revoked key | Check the key |
| `405` | Wrong HTTP method | |
| `429` | The organization's monthly question limit is reached | Wait for next month, or ask an admin to raise the limit |
| `503` | The AI service is unavailable and quoting is off, **or** the platform is in maintenance mode | Retry later (`Retry-After` is sent during maintenance) |

## Examples

```bash
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Authorization: Bearer sk_your_key" -H "Content-Type: application/json" \
  -d '{"question": "Tell me about KIPI"}'
```

```python
import requests

r = requests.post(
    "http://localhost:8001/api/v1/chat",
    headers={"Authorization": "Bearer sk_your_key"},
    json={"question": "Tell me about KIPI"},
    timeout=120,
)
data = r.json()
print(data["answer"], data["citations"])
# Follow-up in the same conversation:
requests.post(..., json={"question": "And how do I apply?", "session_id": data["session_id"]})
```

```javascript
const r = await fetch("http://localhost:8001/api/v1/chat", {
  method: "POST",
  headers: { "Authorization": "Bearer sk_your_key", "Content-Type": "application/json" },
  body: JSON.stringify({ question: "Tell me about KIPI" }),
});
const data = await r.json();
console.log(data.answer);
```

Answers can take several seconds (longer on the first question after a restart). Use a timeout of 60–120 seconds.

## Compatibility with the old Streamlit API

`/api/chat`, `/api/health` and `/api/validate-key` (without `v1`) still work, so clients of the old `shit_api.py`, including `static/index.html`, only need their base URL changed. Keys imported from the Streamlit app (e.g. `ICTA_CHAT`) keep working.

## Legacy APIs

These belong to the old Streamlit-era apps and are kept for reference only.

- **v2, `shit_api.py`** (port 8006): `POST /api/chat`, `GET /api/health`, `GET /api/validate-key`, an unauthenticated `GET /api/generate-key`, `GET /api`, `GET /` (web chat page). Keys had to be listed in the `API_KEYS` environment variable; `session_id` gave no conversation memory; `documents_count` was always 0 or 1.
- **v1, `api.py`** (port 8006, no authentication): `POST /upload`, `POST /chat` (`{"query", "session_id"}`), `GET /files`, `DELETE /files/{file_id}` (left the file's chunks behind), `DELETE /files`, `GET /health`. It used the retired `gemini-pro` model, so `/chat` returns an error.
