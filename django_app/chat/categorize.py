"""Sort questions into a type (intent) and an admin-defined topic.

Runs locally: intents use simple rules and topics use the same embedding
model as search, so categorizing costs no Gemini calls.
"""
import logging
import math
import re
from functools import lru_cache

from rag.embeddings import embed_texts

logger = logging.getLogger(__name__)

# Below this cosine similarity a question is left as "Other" rather than forced into a topic.
TOPIC_MIN_SCORE = 0.25
RECATEGORIZE_LIMIT = 5000


class Intent:
    GREETING = "greeting"
    ABOUT = "about"
    FOLLOW_UP = "follow_up"
    DOCUMENT = "document"

    LABELS = {
        GREETING: "Greeting / small talk",
        ABOUT: "About the assistant",
        FOLLOW_UP: "Follow-up / rewrite",
        DOCUMENT: "Document question",
    }


_GREETING_WORDS = {
    "hi", "hello", "hey", "hiya", "yo", "habari", "mambo", "sasa", "jambo", "niaje", "thanks", "thank",
    "asante", "ok", "okay", "cool", "good", "morning", "afternoon", "evening", "you", "there", "test", "testing",
}
_STRETCHED_GREETING = re.compile(r"^(h+e+l+o+|h+i+|h+e+y+)$")
_ABOUT = re.compile(
    r"\b(who|what) (are|is|r) (you|u)\b|\bwhat (do|can) you (do|help)|\byour name\b|\bhow do you work\b"
    r"|\bwhat (documents?|files?|info(rmation)?) (do|did|to) you have\b|\bare you (a )?(bot|human|ai)\b"
)
_FOLLOW_UP = re.compile(
    r"^(make|rewrite|rephrase|simplify|shorten|summari[sz]e|explain) (it|that|this|again|simpler|more)"
    r"|\b(in )?(simple|simpler|plain|better) (language|words|english|terms)\b|^(tell me )?more\b|^(and|what about|how about) "
)


def classify_intent(text):
    clean = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    words = clean.split()
    if not words:
        return Intent.GREETING
    if len(words) <= 4 and (all(w in _GREETING_WORDS for w in words) or _STRETCHED_GREETING.match(words[0])):
        return Intent.GREETING
    if _ABOUT.search(clean):
        return Intent.ABOUT
    if _FOLLOW_UP.search(clean):
        return Intent.FOLLOW_UP
    return Intent.DOCUMENT


@lru_cache(maxsize=512)
def _topic_vector(text):
    return tuple(embed_texts([text])[0])


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def best_topics(texts, topics):
    """For each text, the closest topic (or None when nothing is close enough)."""
    topics = list(topics)
    if not topics or not texts:
        return [None] * len(texts)
    topic_vectors = [_topic_vector(t.match_text) for t in topics]
    results = []
    for vector in embed_texts(texts):
        scored = max(((_cosine(vector, tv), t) for tv, t in zip(topic_vectors, topics)), key=lambda st: st[0])
        results.append(scored[1] if scored[0] >= TOPIC_MIN_SCORE else None)
    return results


def categorize_question(message):
    """Set intent and topic on a single user message (called right after it's asked)."""
    from .models import Topic

    message.intent = classify_intent(message.content)
    message.topic = None
    if message.intent == Intent.DOCUMENT:
        try:
            message.topic = best_topics([message.content], Topic.objects.filter(organization_id=message.organization_id))[0]
        except Exception:  # categorizing must never break chat
            logger.exception("Topic matching failed for message %s", message.pk)
    message.save(update_fields=["intent", "topic"])


def recategorize(organization):
    """Re-run categorization for an organization's questions, e.g. after topics change."""
    from .models import Message, Topic

    questions = list(
        Message.objects.filter(organization=organization, role=Message.Role.USER).order_by("-id")[:RECATEGORIZE_LIMIT]
    )
    topics = list(Topic.objects.filter(organization=organization))
    for q in questions:
        q.intent = classify_intent(q.content)
    document_questions = [q for q in questions if q.intent == Intent.DOCUMENT]
    for q, topic in zip(document_questions, best_topics([q.content for q in document_questions], topics)):
        q.topic = topic
    for q in questions:
        if q.intent != Intent.DOCUMENT:
            q.topic = None
    Message.objects.bulk_update(questions, ["intent", "topic"], batch_size=500)
    return len(questions)
