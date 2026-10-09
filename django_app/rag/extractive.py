"""Answers quoted straight from the passages, with no language model.

The last link in the fallback chain (Gemini → local model → this). The sentences
that best match the question are picked from the retrieved passages with the same
local embedding model search uses, then shown with their file and page. It needs
no network, costs nothing and can't make anything up, but it quotes rather than
rewords.
"""
import re

from .embeddings import embed_query, embed_texts

MAX_SENTENCES = 3
# Below this a sentence doesn't really answer the question. Normalized vectors,
# so the score is cosine similarity (0 unrelated, 1 same meaning).
MIN_SCORE = 0.25
# Earlier passages ranked higher in search; nudge their sentences up a little.
RANK_PENALTY = 0.02

INTRO = "Here is what the documents say:"
NOTE = "_Quoted directly from your documents._"
NOT_FOUND = (
    "I couldn't find a clear answer to that in the documents. "
    "Try rephrasing the question, or ask about a specific topic they cover."
)

# Sentence ends, bullet starts and blank lines.
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n\s*[-•*]\s+|\n{2,}")


def _sentences(text):
    for part in _SPLIT.split(text):
        sentence = " ".join(part.split()).strip(" -•*")
        if 25 <= len(sentence) <= 500 and re.search(r"[A-Za-z]{3}", sentence):
            yield sentence


def _greeting(organization, question):
    from chat.categorize import Intent, classify_intent  # chat imports rag, so import here

    intent = classify_intent(question)
    if intent == Intent.GREETING:
        return f"Hello! I can answer questions about {organization.name}'s documents. What would you like to know?"
    if intent == Intent.ABOUT:
        return (f"I'm {organization.display_assistant_name}. I answer questions using {organization.name}'s "
                "documents and show the file and page each answer comes from.")
    return None


def answer(organization, question, hits):
    """Markdown: the best-matching sentences as a list, each with its source."""
    small_talk = _greeting(organization, question)
    if small_talk:
        return small_talk

    candidates, seen = [], set()
    for rank, hit in enumerate(hits):
        for position, sentence in enumerate(_sentences(hit.text)):
            key = sentence.lower()
            if key not in seen:  # overlapping passages repeat sentences
                seen.add(key)
                candidates.append((rank, position, sentence, hit))
    if not candidates:
        return NOT_FOUND

    question_vector = embed_query(question)
    vectors = embed_texts(sentence for _, _, sentence, _ in candidates)
    scored = [
        (sum(q * v for q, v in zip(question_vector, vector)) - RANK_PENALTY * candidate[0], candidate)
        for vector, candidate in zip(vectors, candidates)
    ]
    best = sorted((s for s in scored if s[0] >= MIN_SCORE), key=lambda s: -s[0])[:MAX_SENTENCES]
    if not best:
        return NOT_FOUND

    best.sort(key=lambda s: (s[1][0], s[1][1]))  # in document order, which reads more naturally
    lines = []
    for _, (_, _, sentence, hit) in best:
        source = f" [{hit.filename}, page {hit.page}]" if organization.show_sources else ""
        lines.append(f"- {sentence}{source}")
    return "\n\n".join([INTRO, "\n".join(lines), NOTE])
