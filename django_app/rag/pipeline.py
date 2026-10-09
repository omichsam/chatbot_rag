"""Retrieve → prompt → generate, scoped to one organization.

Answers are written by the first AI provider that works, in the order a
superadmin set in the portal (Gemini, OpenAI, Claude, Groq, DeepSeek,
OpenRouter, a local Ollama model...), then by quoting sentences straight from
the documents. So a question gets an answer even when providers are down or out
of quota. Organizations that turn external AI off use only local providers.
"""
import logging
from dataclasses import dataclass, field


from . import extractive, llm, vectorstore

logger = logging.getLogger(__name__)

# Who wrote an answer, stored on the message: a provider kind ("gemini", "groq",
# "anthropic"...), or one of these.
QUOTED, CACHE = "quoted", "cache"

# Who the assistant is. An organization's own system prompt replaces this; the
# rules below (scope, length, language, citations) come from its settings and always apply.
DEFAULT_SYSTEM_PROMPT = "You are {assistant}, a friendly document assistant for {org}."

SCOPE_RULES = {
    "strict": "Answer the user's question using only the document excerpts below. If the answer is not in "
              "the excerpts, say politely that you don't have that information.",
    "assisted": "Base your answer on the document excerpts below. If they don't fully cover the question, you "
                "may add general knowledge, but say clearly which part does not come from the documents.",
}
LENGTH_RULES = {
    "short": "Be very brief: 1 to 3 sentences.",
    "balanced": "Be brief: answer in 2 to 6 sentences or a short list, and only go into more detail when the "
                "user asks for it.",
    "detailed": "Give a thorough answer, using short headings or lists where they help.",
}
CITATION_RULE = "When it helps, cite sources like [filename, page N]."

NO_DOCUMENTS_ANSWER = (
    "There are no documents in this workspace yet. Ask an admin to upload some, "
    "then try again."
)


@dataclass
class Answer:
    text: str
    hits: list = field(default_factory=list)
    answered_by: str = ""       # a provider kind, QUOTED, CACHE, or "" (nothing to search)
    fallback_reason: str = ""   # why earlier writers were skipped, for admins

    @property
    def citations(self):
        seen, citations = set(), []
        for hit in self.hits:
            key = (hit.document_id, hit.page)
            if key not in seen:
                seen.add(key)
                citations.append(
                    {"document_id": hit.document_id, "filename": hit.filename, "page": hit.page}
                )
        return citations

    @property
    def top_score(self):
        return max((hit.score for hit in self.hits), default=None)

    @property
    def sources(self):
        return list(dict.fromkeys(hit.filename for hit in self.hits))


def system_instructions(organization):
    persona = organization.system_prompt or DEFAULT_SYSTEM_PROMPT
    persona = persona.replace("{org}", organization.name).replace("{assistant}", organization.display_assistant_name)
    rules = [
        SCOPE_RULES.get(organization.knowledge_scope, SCOPE_RULES["strict"]),
        LENGTH_RULES.get(organization.answer_length, LENGTH_RULES["balanced"]),
        f"Always answer in {organization.response_language}." if organization.response_language
        else "Answer in the same language as the question.",
    ]
    if organization.show_sources:
        rules.append(CITATION_RULE)
    return " ".join([persona.strip(), *rules])


def _generate_options(organization):
    return {"max_output_tokens": organization.max_output_tokens, "temperature": organization.temperature}


def build_prompt(organization, question, hits, history=()):
    system = system_instructions(organization)
    excerpts = "\n\n".join(
        f"[{i}] ({hit.filename}, page {hit.page})\n{hit.text}" for i, hit in enumerate(hits, start=1)
    )
    parts = [system, f"Document excerpts:\n{excerpts}"]
    if history:
        conversation = "\n".join(f"{role.title()}: {content}" for role, content in history)
        parts.append(f"Conversation so far:\n{conversation}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def retrieve(organization, question, history=()):
    """The passages to answer from. history is (role, content) pairs, oldest first."""
    # Follow-ups like "what about the second one?" retrieve poorly on their own,
    # so search with the previous question as well.
    previous_questions = [content for role, content in history if role == "user"]
    search_text = f"{previous_questions[-1]}\n{question}" if previous_questions else question
    from documents.models import Document  # documents imports rag, so import here

    paused = Document.objects.paused_ids(organization.id)
    return vectorstore.search(organization.id, search_text, organization.top_k, exclude_document_ids=paused)


def providers(organization):
    """The enabled AI providers this organization may use, in answer order."""
    from portal.models import AIProvider  # portal imports rag, so import here

    return [p for p in AIProvider.objects.filter(is_enabled=True)
            if (organization.external_ai_allowed or not p.is_external) and p.model_for(organization)]


def writers(organization):
    """The AI writers to try, in order: (provider, write(prompt), stream(prompt))."""
    options = _generate_options(organization)
    return [(p, *llm.writer(p, p.model_for(organization), **options)) for p in providers(organization)]


def _failed(organization, provider, exc, failures):
    logger.warning("%s could not answer for organization %s: %s", provider.name, organization.pk, exc)
    failures.append(f"{provider.name}: {exc}")


def _quoted(organization, question, hits, failures):
    """The last resort: sentences quoted from the passages, no AI needed."""
    reason = "; ".join(failures)
    from portal.models import platform

    if not platform().quoted_fallback:
        raise llm.LLMError(reason or "No answer writer is enabled.")
    return Answer(extractive.answer(organization, question, hits), hits, QUOTED, reason)


def answer_question(organization, question, history=()):
    hits = retrieve(organization, question, history)
    if not hits:
        return Answer(NO_DOCUMENTS_ANSWER)
    prompt = build_prompt(organization, question, hits, history)
    failures = []
    for provider, write, _ in writers(organization):
        try:
            return Answer(write(prompt), hits, provider.kind, "; ".join(failures))
        except llm.LLMError as exc:
            _failed(organization, provider, exc, failures)
    return _quoted(organization, question, hits, failures)


def stream_answer(organization, question, history=()):
    """(hits, iterator of text pieces, outcome).

    The iterator moves to the next writer when one fails before writing anything,
    and raises llm.LLMError only when one fails part-way (what was shown can't be
    unsaid) or when every writer failed and quoting is turned off. Once it is used
    up, outcome holds answered_by and fallback_reason.
    """
    outcome = {"answered_by": "", "fallback_reason": ""}
    hits = retrieve(organization, question, history)
    if not hits:
        return [], iter([NO_DOCUMENTS_ANSWER]), outcome
    prompt = build_prompt(organization, question, hits, history)

    def pieces():
        failures = []
        for provider, _, stream in writers(organization):
            started = False
            try:
                for piece in stream(prompt):
                    started = True
                    yield piece
            except llm.LLMError as exc:
                if started:
                    raise
                _failed(organization, provider, exc, failures)
                continue
            outcome.update(answered_by=provider.kind, fallback_reason="; ".join(failures))
            return
        answer = _quoted(organization, question, hits, failures)
        outcome.update(answered_by=answer.answered_by, fallback_reason=answer.fallback_reason)
        yield answer.text

    return hits, pieces(), outcome
