# Admin guide

For organization **owners** and **admins**. Admins manage documents, members, settings and quality; owners can also make other people owners. Everything here applies only to your own organization.

## Roles

| | Member | Admin | Owner |
| --- | :-: | :-: | :-: |
| Ask questions, see the document list | ✓ | ✓ | ✓ |
| Upload, pause, re-index and delete documents | | ✓ | ✓ |
| Analytics, Questions, Feedback, Knowledge gaps | | ✓ | ✓ |
| API keys and organization Settings | | ✓ | ✓ |
| Add or remove members, change roles | | ✓ (not owners) | ✓ |
| Make someone an owner | | | ✓ |

## Documents

**Upload:** drop PDF, TXT or Markdown files on the upload area (several at once). Each file is checked, then indexed in the background: its status goes *Queued → Processing → Ready*. Only Ready documents are used in answers. Uploading the same file twice is caught and skipped.

| Action | What happens |
| --- | --- |
| ⏸ **Pause** | The document is kept but answers stop using it straight away. Use it for a document under review. |
| ▶ **Resume** | Answers use it again immediately, with no re-indexing. |
| ↻ **Re-index** | The text is read and indexed again (a few seconds to minutes). Use it after a failure. |
| 🗑 **Delete** | The file and its index are removed permanently. |

Every action asks for confirmation. A document that shows **Failed** gives the reason; the most common is a scanned PDF with no text layer, which needs OCR before uploading.

## Members

**Members & roles** (`/org/members/`): add an existing user by username, or create an account for someone; change roles; remove people. Pending **access requests** from members appear here too; approve or decline them.

## Settings

**Settings** (`/org/settings/`) has six tabs. Changes apply to the next question. A dot marks tabs with unsaved changes.

| Tab | What you can set |
| --- | --- |
| General | Organization name; the assistant's name |
| Chat experience | Welcome message; up to 6 suggested questions; whether answers show their file and page |
| Answers | Length (short, balanced, detailed); creativity (precise, balanced, flexible); knowledge scope (documents only, or documents first with general knowledge clearly labelled); reply language |
| AI model | Use external AI providers (off keeps everything on your server); Gemini model; your own instructions for the assistant; a read-only preview of exactly what the assistant is told |
| Retrieval & quality | Passages read per answer (1–12); conversation memory (0–10 turns); the knowledge-gap threshold |
| Usage & performance | Monthly question limit with a usage bar; reuse of answers to repeated questions |

Which AI providers answer, and in what order, is chosen by the platform's superadmins (see [AI_PROVIDERS.md](AI_PROVIDERS.md)).

## Watching quality

| Page | Use it to |
| --- | --- |
| **Home** | See volume and trends, answer quality (answered, not covered, failed), busiest times, most-used documents, and a *Needs attention* list (failed answers, backup answers, access requests, gaps) |
| **Analytics** | Questions per day, average answer time, top questions, most active people |
| **Questions** | Browse every question with filters (period, outcome, source, API key, topic, type). Click a row for the full question, answer and sources. Export to CSV. Manage **topics** so questions are grouped the way you think about them. |
| **Knowledge gaps** | Questions the documents couldn't answer well. Add or update a document, then mark the gap resolved. |
| **Feedback** | Answers people rated 👎, with their comments. Mark them reviewed when handled. |

A good routine: check *Needs attention* on Home weekly, clear Knowledge gaps by adding documents, and read new Feedback.

## API keys

**API keys** (`/keys/`) let another system ask questions through the [API](API_REFERENCE.md). A new key is shown **once**; copy it then. Revoke a key to stop it immediately. The **Developer** page lets anyone try the API in the browser with a key.

## Good practice

- Keep documents current: pause or delete outdated ones, so answers don't quote old rules.
- Prefer text-based PDFs. Scanned pages can't be read without OCR.
- Use the Settings preview ("What the assistant is told") to check the effect of your changes before people see them.
