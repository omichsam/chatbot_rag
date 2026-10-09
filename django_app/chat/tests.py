import json
from unittest import mock

from django.test import TestCase
from django.urls import reverse

from documents import services as document_services
from documents.models import Document
from rag.llm import LLMError
from rag.pipeline import NO_DOCUMENTS_ANSWER
from rag.testing import RagTestMixin, platform_settings
from tenants.models import Membership, Organization
from tenants.tests import make_member

from .models import ChatSession, Message


def add_document(organization, name, text):
    from django.core.files.base import ContentFile

    document = Document(organization=organization, original_name=name, checksum=name, size=len(text))
    document.file.save(name, ContentFile(text.encode()), save=True)
    document_services.ingest_document(document.pk)
    return document


class ChatTests(RagTestMixin, TestCase):
    def setUp(self):
        self.acme = Organization.objects.create(name="Acme")
        self.rival = Organization.objects.create(name="Rival")
        self.alice = make_member("alice", self.acme, Membership.Role.ADMIN)
        self.bob = make_member("bob", self.rival, Membership.Role.ADMIN)
        add_document(self.acme, "zebras.txt", "Zebras live in Africa and have black and white stripes.")
        add_document(self.rival, "volcanoes.txt", "Volcanoes erupt lava and ash from magma chambers.")

    def ask(self, question, session_id=None):
        return self.client.post(
            reverse("chat:send"), json.dumps({"question": question, "session_id": session_id}),
            content_type="application/json",
        )

    def test_answer_uses_only_own_organizations_documents(self):
        self.client.force_login(self.bob)
        data = self.ask("Tell me about zebras stripes").json()
        filenames = {s["filename"] for s in data["sources"]}
        self.assertEqual(filenames, {"volcanoes.txt"})

        self.client.force_login(self.alice)
        data = self.ask("Tell me about zebras stripes").json()
        self.assertEqual({s["filename"] for s in data["sources"]}, {"zebras.txt"})
        self.assertIn("zebras", data["answer"])

    def test_messages_and_session_are_stored(self):
        self.client.force_login(self.alice)
        first = self.ask("zebras?").json()
        self.ask("where do they live?", first["session_id"])
        session = ChatSession.objects.get(public_id=first["session_id"])
        self.assertEqual(session.messages.count(), 4)
        self.assertEqual(session.title, "zebras?")

    def test_follow_up_includes_history_in_prompt(self):
        self.client.force_login(self.alice)
        first = self.ask("zebras?").json()
        with mock.patch("rag.llm.generate", return_value="ok") as generate:
            self.ask("where do they live?", first["session_id"])
        prompt = generate.call_args.args[0]
        self.assertIn("Conversation so far", prompt)
        self.assertIn("User: zebras?", prompt)

    def test_cannot_use_another_users_session(self):
        self.client.force_login(self.alice)
        session_id = self.ask("zebras?").json()["session_id"]
        self.client.force_login(self.bob)
        self.assertEqual(self.ask("hi", session_id).status_code, 404)
        self.assertEqual(self.client.get(reverse("chat:session", args=[session_id])).status_code, 404)

    def test_no_documents_answer(self):
        empty = Organization.objects.create(name="Empty")
        self.client.force_login(make_member("carol", empty))
        self.assertEqual(self.ask("anything").json()["answer"], NO_DOCUMENTS_ANSWER)

    def test_monthly_quota(self):
        self.acme.monthly_query_limit = 1
        self.acme.save()
        self.client.force_login(self.alice)
        self.assertEqual(self.ask("one").status_code, 200)
        self.assertEqual(self.ask("two").status_code, 429)

    def test_gemini_failure_falls_back_to_quoted_answer(self):
        self.client.force_login(self.alice)
        with mock.patch("rag.llm.generate", side_effect=LLMError("down")):
            data = self.ask("Where do zebras live?").json()
        self.assertFalse(data["is_error"])
        reply = Message.objects.get(role=Message.Role.ASSISTANT)
        self.assertEqual(reply.answered_by, "quoted")
        self.assertIn("Gemini: down", reply.error_detail)
        self.assertIn("Quoted directly from your documents", reply.content)

    def test_failure_is_recorded_when_every_writer_fails(self):
        self.client.force_login(self.alice)
        with platform_settings(quoted_fallback=False), mock.patch("rag.llm.generate", side_effect=LLMError("down")):
            data = self.ask("zebras?").json()
        self.assertTrue(data["is_error"])
        self.assertTrue(Message.objects.filter(is_error=True).exists())

    def test_answer_html_is_escaped_in_page(self):
        self.client.force_login(self.alice)
        with mock.patch("rag.llm.generate", return_value="<script>alert(1)</script>"):
            session_id = self.ask("zebras?").json()["session_id"]
        page = self.client.get(reverse("chat:session", args=[session_id]))
        self.assertNotContains(page, "<script>alert(1)</script>")
        self.assertContains(page, "&lt;script&gt;")

    def test_dashboard_shows_own_organization_numbers(self):
        self.client.force_login(self.alice)
        self.ask("zebras?")
        page = self.client.get(reverse("dashboard"))
        self.assertContains(page, "Needs attention")
        self.assertContains(page, "Where questions come from")
        self.assertEqual(page.context["total"], 1)       # bob's org questions are not counted
        self.assertEqual(page.context["docs"], 1)
        self.assertEqual(self.client.get(reverse("dashboard") + "?period=7").context["days"], 7)

        member = make_member("mia", self.acme)
        self.client.force_login(member)
        member_page = self.client.get(reverse("dashboard"))
        self.assertContains(member_page, "Your recent chats")
        self.assertNotContains(member_page, "Needs attention")

    def test_analytics_admin_only_and_renders(self):
        self.client.force_login(self.alice)
        self.ask("zebras?")
        self.assertContains(self.client.get(reverse("chat:analytics")), "zebras?")
        member = make_member("mia", self.acme)
        self.client.force_login(member)
        self.assertEqual(self.client.get(reverse("chat:analytics")).status_code, 403)


class InsightsTests(RagTestMixin, TestCase):
    def setUp(self):
        self.acme = Organization.objects.create(name="Acme")
        self.admin = make_member("ada", self.acme, Membership.Role.ADMIN)
        self.member = make_member("max", self.acme)
        add_document(self.acme, "zebras.txt", "Zebras live in Africa and have black and white stripes.")

    def ask(self, question, user):
        self.client.force_login(user)
        return self.client.post(reverse("chat:send"), json.dumps({"question": question}), content_type="application/json").json()

    def post_json(self, url, body=None):
        return self.client.post(url, json.dumps(body or {}), content_type="application/json")

    def test_rate_and_feedback_review(self):
        data = self.ask("zebras stripes?", self.member)
        response = self.post_json(reverse("chat:rate", args=[data["message_id"]]), {"rating": -1, "comment": "too vague"})
        self.assertEqual(response.json()["rating"], -1)

        self.client.force_login(self.admin)
        page = self.client.get(reverse("chat:feedback"))
        self.assertContains(page, "too vague")
        self.assertContains(page, "zebras stripes?")
        self.client.post(reverse("chat:mark_reviewed", args=[data["message_id"]]))
        self.assertNotContains(self.client.get(reverse("chat:feedback")), "too vague")

    def test_cannot_rate_someone_elses_answer(self):
        data = self.ask("zebras?", self.member)
        self.client.force_login(self.admin)
        self.assertEqual(self.post_json(reverse("chat:rate", args=[data["message_id"]]), {"rating": 1}).status_code, 404)

    def test_save_toggle_and_saved_page(self):
        data = self.ask("zebras stripes?", self.member)
        url = reverse("chat:save", args=[data["message_id"]])
        self.assertTrue(self.post_json(url).json()["saved"])
        self.assertContains(self.client.get(reverse("chat:saved")), "zebras stripes?")
        self.assertFalse(self.post_json(url).json()["saved"])
        self.assertContains(self.client.get(reverse("chat:saved")), "Nothing saved yet")

    def test_knowledge_gaps_lists_weak_matches_and_resolves(self):
        self.ask("quantum chromodynamics lattice?", self.member)   # nothing in the docs matches
        self.client.force_login(self.admin)
        page = self.client.get(reverse("chat:gaps"))
        self.assertContains(page, "quantum chromodynamics lattice?")
        ids = [m.pk for m in Message.objects.filter(role=Message.Role.ASSISTANT)]
        self.client.post(reverse("chat:resolve_gap"), {"ids": ids})
        self.assertContains(self.client.get(reverse("chat:gaps")), "No gaps found")

    def test_members_see_admin_tools_card_and_friendly_403(self):
        self.client.force_login(self.member)
        page = self.client.get(reverse("dashboard"))
        for label in ("Saved answers", "My history", "Developer", "Help"):
            self.assertContains(page, label)
        # Admin areas are named in one "Admin tools" card instead of locked menu items.
        self.assertContains(page, "Admin tools ·")
        for label in ("Analytics", "Feedback", "Knowledge gaps", "API keys", "Members &amp; roles", "Settings"):
            self.assertContains(page, label)
        self.assertNotContains(page, 'href="/chat/gaps/"')
        denied = self.client.get(reverse("chat:gaps"))
        self.assertEqual(denied.status_code, 403)
        self.assertContains(denied, "This page is for admins", status_code=403)

    def test_admin_menu_has_no_locks_and_shows_badges(self):
        data = self.ask("zebras?", self.member)
        self.post_json(reverse("chat:rate", args=[data["message_id"]]), {"rating": -1})
        self.client.force_login(self.admin)
        page = self.client.get(reverse("dashboard"))
        self.assertNotContains(page, "(admins only)")
        self.assertContains(page, '<em class="tag alert">1</em>')

    def test_help_page_for_everyone(self):
        self.client.force_login(self.member)
        self.assertContains(self.client.get(reverse("chat:help")), "Getting good answers")


class CategorizeTests(TestCase):
    def test_intents(self):
        from .categorize import Intent, classify_intent

        cases = {
            "hi": Intent.GREETING, "heeelooo": Intent.GREETING, "Hello there!": Intent.GREETING,
            "who is you?": Intent.ABOUT, "what do you do": Intent.ABOUT, "what document to you have": Intent.ABOUT,
            "make it tobe in a better language that one can understand": Intent.FOLLOW_UP,
            "What is Huduma WhiteBox?": Intent.DOCUMENT, "Tell me about KIPI": Intent.DOCUMENT,
        }
        for text, expected in cases.items():
            self.assertEqual(classify_intent(text), expected, text)


class ExplorerTests(RagTestMixin, TestCase):
    def setUp(self):
        from api.models import APIKey

        from .models import Topic

        self.org = Organization.objects.create(name="Acme")
        self.admin = make_member("ada", self.org, Membership.Role.ADMIN)
        add_document(self.org, "zebras.txt", "Zebras live in Africa and have black and white stripes.")
        Topic.objects.create(organization=self.org, name="Animals", description="zebras stripes wildlife africa")
        self.key, self.raw = APIKey.create_key(self.org, "website")

    def api_ask(self, question, test=False):
        headers = {"HTTP_AUTHORIZATION": f"Bearer {self.raw}"}
        if test:
            headers["HTTP_X_DOCASSIST_CLIENT"] = "playground"
        return self.client.post("/api/v1/chat", json.dumps({"question": question}), content_type="application/json", **headers)

    def test_sources_topics_and_outcomes(self):
        self.api_ask("zebras stripes africa")                 # live API
        self.api_ask("zebras stripes africa", test=True)      # playground test
        self.client.force_login(self.admin)
        self.client.post(reverse("chat:send"), json.dumps({"question": "hi"}), content_type="application/json")

        questions = Message.objects.filter(role=Message.Role.USER)
        self.assertEqual(questions.filter(topic__name="Animals").count(), 2)
        self.assertEqual(questions.get(content="hi").intent, "greeting")
        self.assertTrue(ChatSession.objects.filter(is_test=True).exists())

        page = self.client.get(reverse("chat:questions"))
        self.assertContains(page, "zebras stripes africa")
        self.assertContains(page, 'class="src test"')
        self.assertContains(page, 'class="src api"')
        self.assertContains(page, 'class="src web"')
        self.assertContains(page, "website")          # per-key table

        only_tests = self.client.get(reverse("chat:questions") + "?source=test")
        self.assertEqual(only_tests.context["total"], 1)
        by_key = self.client.get(reverse("chat:questions") + f"?key={self.key.pk}")
        self.assertEqual(by_key.context["total"], 2)
        failed = self.client.get(reverse("chat:questions") + "?outcome=failed")
        self.assertEqual(failed.context["total"], 0)

    def test_csv_export_and_topic_management(self):
        self.api_ask("zebras stripes africa")
        self.client.force_login(self.admin)
        csv_response = self.client.get(reverse("chat:questions") + "?format=csv")
        self.assertEqual(csv_response["Content-Type"], "text/csv")
        self.assertIn("zebras stripes africa", csv_response.content.decode())

        self.client.post(reverse("chat:topic_add"), {"name": "Pricing", "description": "cost price fees"})
        self.assertTrue(self.org.topics.filter(name="Pricing").exists())
        animals = self.org.topics.get(name="Animals")
        self.client.post(reverse("chat:topic_delete", args=[animals.pk]))
        self.assertFalse(Message.objects.filter(topic__name="Animals").exists())

    def test_explorer_is_admin_only(self):
        member = make_member("mo", self.org)
        self.client.force_login(member)
        self.assertEqual(self.client.get(reverse("chat:questions")).status_code, 403)


class SpeedTests(RagTestMixin, TestCase):
    """Streaming answers, the repeat-question cache and startup warm-up."""

    def setUp(self):
        from django.core.cache import cache

        cache.clear()
        self.org = Organization.objects.create(name="Acme")
        self.user = make_member("ada", self.org, Membership.Role.ADMIN)
        add_document(self.org, "zebras.txt", "Zebras live in Africa and have black and white stripes.")
        self.client.force_login(self.user)

    def stream(self, question, session_id=None):
        response = self.client.post(reverse("chat:stream"), json.dumps({"question": question, "session_id": session_id}),
                                    content_type="application/json")
        self.assertEqual(response["Content-Type"], "application/x-ndjson")
        return [json.loads(line) for line in b"".join(response.streaming_content).decode().splitlines() if line]

    def test_stream_sends_pieces_then_the_saved_answer(self):
        events = self.stream("zebras stripes?")
        self.assertEqual(events[0]["type"], "start")
        self.assertGreater(sum(e["type"] == "delta" for e in events), 1)
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["sources"][0]["filename"], "zebras.txt")
        reply = Message.objects.get(pk=done["message_id"])
        self.assertEqual(reply.content, done["answer"])
        self.assertIn("zebras stripes?", reply.content)

    def test_repeated_first_question_is_answered_from_cache(self):
        with mock.patch("rag.llm.generate", return_value="Zebras have stripes.") as generate:
            self.client.post(reverse("chat:send"), json.dumps({"question": "Zebras stripes?"}), content_type="application/json")
            second = self.client.post(reverse("chat:send"), json.dumps({"question": "  zebras STRIPES "}),
                                      content_type="application/json").json()
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(second["answer"], "Zebras have stripes.")
        self.assertEqual(second["sources"][0]["filename"], "zebras.txt")

    def test_cache_resets_when_documents_change_and_skips_follow_ups(self):
        with mock.patch("rag.llm.generate", return_value="first") as generate:
            first = self.client.post(reverse("chat:send"), json.dumps({"question": "zebras?"}), content_type="application/json").json()
            add_document(self.org, "more.txt", "Zebras also run fast.")            # knowledge base changed
            self.client.post(reverse("chat:send"), json.dumps({"question": "zebras?"}), content_type="application/json")
            self.client.post(reverse("chat:send"), json.dumps({"question": "zebras?", "session_id": first["session_id"]}),
                             content_type="application/json")                     # follow-up in a conversation
        self.assertEqual(generate.call_count, 3)

    def test_stream_ai_failure_is_reported_and_stored(self):
        with platform_settings(quoted_fallback=False), \
                mock.patch("rag.llm.generate_stream", side_effect=LLMError("429 quota exceeded")):
            events = self.stream("zebras?")
        self.assertIn("error", [e["type"] for e in events])
        self.assertTrue(events[-1]["is_error"])
        self.assertIn("quota", Message.objects.get(pk=events[-1]["message_id"]).error_detail)

    def test_stream_falls_back_to_next_provider_then_quotes(self):
        from portal.models import AIProvider

        AIProvider.objects.create(name="Local model", kind="ollama", model="llama3.2:3b", priority=20)
        with mock.patch("rag.llm.generate_stream", side_effect=LLMError("429 quota exceeded")), \
                mock.patch("rag.llm.chat_stream", return_value=iter(["Zebras live ", "in Africa."])):
            events = self.stream("zebras?")
        self.assertNotIn("error", [e["type"] for e in events])
        reply = Message.objects.get(pk=events[-1]["message_id"])
        self.assertEqual((reply.content, reply.answered_by, reply.is_error), ("Zebras live in Africa.", "ollama", False))

        with mock.patch("rag.llm.generate_stream", side_effect=LLMError("429 quota exceeded")), \
                mock.patch("rag.llm.chat_stream", side_effect=LLMError("connection refused")):
            events = self.stream("Where do zebras live?")
        reply = Message.objects.get(pk=events[-1]["message_id"])
        self.assertEqual(reply.answered_by, "quoted")
        self.assertIn("Local model: connection refused", reply.error_detail)

    def test_warmup_does_not_run_in_tests_or_commands(self):
        from rag.warmup import is_serving

        self.assertFalse(is_serving())


class OrganizationSettingsTests(RagTestMixin, TestCase):
    def setUp(self):
        self.org = Organization.objects.create(name="Acme")
        self.admin = make_member("ada", self.org, Membership.Role.ADMIN)
        self.member = make_member("max", self.org)
        for i in range(4):
            add_document(self.org, f"zebras{i}.txt", f"Zebras group {i} live in Africa and have black and white stripes. " * 20)

    def settings_data(self, **changes):
        data = {
            "name": "Acme", "assistant_name": "", "welcome_message": "", "suggested_questions": "", "show_sources": "on",
            "answer_length": "balanced", "creativity": "precise", "knowledge_scope": "strict", "response_language": "",
            "llm_model": "", "system_prompt": "", "top_k": 5, "history_turns": 3, "gap_threshold": 0.35,
            "monthly_query_limit": 0, "cache_answers": "on", "use_external_ai": "on",
        }
        data.update(changes)
        return {k: v for k, v in data.items() if v is not None}

    def save(self, **changes):
        self.client.force_login(self.admin)
        return self.client.post(reverse("tenants:settings"), self.settings_data(**changes))

    def ask(self, question="zebras?"):
        self.client.force_login(self.member)
        return self.client.post(reverse("chat:send"), json.dumps({"question": question}), content_type="application/json")

    def test_admin_saves_and_member_is_denied(self):
        self.assertRedirects(self.save(assistant_name="Zed", top_k=3, answer_length="short", show_sources=None),
                             reverse("tenants:settings"))
        self.org.refresh_from_db()
        self.assertEqual((self.org.assistant_name, self.org.top_k, self.org.answer_length, self.org.show_sources),
                         ("Zed", 3, "short", False))
        self.client.force_login(self.member)
        self.assertEqual(self.client.get(reverse("tenants:settings")).status_code, 403)

    def test_invalid_values_are_rejected(self):
        response = self.save(top_k=50, suggested_questions="\n".join(f"q{i}" for i in range(9)))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "at most 6 suggestions")
        self.org.refresh_from_db()
        self.assertEqual(self.org.top_k, 5)

    def test_answer_settings_shape_the_prompt_and_model_call(self):
        self.save(assistant_name="Zed", knowledge_scope="assisted", answer_length="detailed",
                  response_language="Kiswahili", show_sources=None, creativity="flexible", top_k=2)
        with mock.patch("rag.llm.generate", return_value="ok") as generate:
            self.ask()
        prompt, model = generate.call_args.args[:2]
        options = generate.call_args.kwargs
        self.assertIn("You are Zed", prompt)
        self.assertIn("may add general knowledge", prompt)
        self.assertIn("thorough answer", prompt)
        self.assertIn("Always answer in Kiswahili", prompt)
        self.assertNotIn("cite sources", prompt)
        self.assertIn("[2] (", prompt)                              # top_k passages only
        self.assertNotIn("[3] (", prompt)
        self.assertEqual(options["temperature"], 0.8)
        self.assertGreater(options["max_output_tokens"], 600)

    def test_conversation_memory_zero_sends_no_history(self):
        self.save(history_turns=0)
        with mock.patch("rag.llm.generate", return_value="first"):
            session_id = self.ask().json()["session_id"]
        with mock.patch("rag.llm.generate", return_value="second") as generate:
            self.client.post(reverse("chat:send"), json.dumps({"question": "and then?", "session_id": session_id}),
                             content_type="application/json")
        self.assertNotIn("Conversation so far", generate.call_args.args[0])

    def test_cache_can_be_switched_off(self):
        self.save(cache_answers=None)
        with mock.patch("rag.llm.generate", return_value="ok") as generate:
            self.ask("zebras?")
            self.ask("zebras?")
        self.assertEqual(generate.call_count, 2)

    def test_changing_answer_settings_bypasses_cached_answers(self):
        with mock.patch("rag.llm.generate", return_value="ok") as generate:
            self.ask("zebras?")
            self.save(answer_length="short")
            self.ask("zebras?")
        self.assertEqual(generate.call_count, 2)

    def test_chat_page_uses_welcome_suggestions_and_hides_sources(self):
        self.save(welcome_message="Ask about our zebra programme.", suggested_questions="Where do zebras live?\nWhy stripes?",
                  show_sources=None)
        with mock.patch("rag.llm.generate", return_value="In Africa."):
            session_id = self.ask().json()["session_id"]
        new_chat = self.client.get(reverse("chat:index"))
        self.assertContains(new_chat, "Ask about our zebra programme.")
        self.assertContains(new_chat, 'data-ask="Why stripes?"')
        self.assertNotContains(new_chat, "Summarize zebras")
        self.assertNotContains(self.client.get(reverse("chat:session", args=[session_id])), "· p.1</span>")

    def test_gap_threshold_is_per_organization(self):
        with mock.patch("rag.llm.generate", return_value="ok"):
            self.ask("zebras?")
        # Test embeddings are fake, so fix the match score.
        Message.objects.filter(role=Message.Role.ASSISTANT).update(retrieval_score=0.5)
        self.client.force_login(self.admin)
        self.assertNotContains(self.client.get(reverse("chat:gaps")), "zebras?")
        self.save(gap_threshold=0.6)
        self.assertContains(self.client.get(reverse("chat:gaps")), "zebras?")


class PaginationTests(RagTestMixin, TestCase):
    def test_questions_list_shows_ten_per_page(self):
        org = Organization.objects.create(name="Acme")
        admin = make_member("ada", org, Membership.Role.ADMIN)
        add_document(org, "zebras.txt", "Zebras live in Africa and have black and white stripes.")
        self.client.force_login(admin)
        for i in range(12):
            self.client.post(reverse("chat:send"), json.dumps({"question": f"zebra question {i}?"}), content_type="application/json")
        first = self.client.get(reverse("chat:questions"))
        self.assertEqual(len(first.context["rows"]), 10)
        self.assertContains(first, "Showing 1–10 of 12")
        second = self.client.get(reverse("chat:questions") + "?page=2")
        self.assertEqual(len(second.context["rows"]), 2)
        self.assertContains(second, "Showing 11–12 of 12")

    def test_tables_get_the_paging_script(self):
        org = Organization.objects.create(name="Acme")
        self.client.force_login(make_member("ada", org, Membership.Role.ADMIN))
        page = self.client.get(reverse("tenants:members"))
        self.assertContains(page, 'data-table data-label="members"')
        self.assertContains(page, 'querySelectorAll("table[data-table], [data-paginate]")')


class FallbackTests(RagTestMixin, TestCase):
    def setUp(self):
        self.org = Organization.objects.create(name="Acme")
        self.admin = make_member("ada", self.org, Membership.Role.ADMIN)
        add_document(self.org, "zebras.txt",
                     "Zebras live in the grasslands of Africa. Their black and white stripes confuse biting flies. "
                     "A group of zebras is called a dazzle.")
        self.client.force_login(self.admin)

    def ask(self, question):
        return self.client.post(reverse("chat:send"), json.dumps({"question": question}), content_type="application/json")

    def reply(self):
        return Message.objects.filter(role=Message.Role.ASSISTANT).latest("id")

    def test_next_provider_answers_when_gemini_fails(self):
        from portal.models import AIProvider

        groq = AIProvider(name="Groq", kind="groq", model="llama-3.3-70b-versatile", priority=20)
        groq.set_api_key("gsk_test_key")
        groq.save()
        with mock.patch("rag.llm.generate", side_effect=LLMError("429 quota")), \
                mock.patch("rag.llm.chat_generate", return_value="Zebras live in Africa.") as chat:
            self.ask("Where do zebras live?")
        self.assertEqual(chat.call_args.args[1], "llama-3.3-70b-versatile")
        self.assertEqual(chat.call_args.kwargs["api_key"], "gsk_test_key")
        self.assertEqual(chat.call_args.kwargs["base_url"], "https://api.groq.com/openai/v1")
        self.assertEqual((self.reply().answered_by, self.reply().content), ("groq", "Zebras live in Africa."))
        self.assertIn("Google Gemini: 429 quota", self.reply().error_detail)

    def test_provider_order_and_external_switch(self):
        from portal.models import AIProvider

        AIProvider.objects.filter(kind="gemini").update(priority=30)
        claude = AIProvider(name="Claude", kind="anthropic", model="claude-haiku-4-5-20251001", priority=10)
        claude.set_api_key("sk-ant-test")
        claude.save()
        AIProvider.objects.create(name="Local", kind="ollama", model="llama3.2:3b", priority=20)
        with mock.patch("rag.llm.claude_generate", return_value="From Claude.") as first, \
                mock.patch("rag.llm.generate") as gemini:
            self.ask("Where do zebras live?")
        first.assert_called_once()
        gemini.assert_not_called()
        self.assertEqual(self.reply().answered_by, "anthropic")

        self.org.use_external_ai = False
        self.org.save()
        with mock.patch("rag.llm.claude_generate") as claude_call, \
                mock.patch("rag.llm.chat_generate", return_value="From the local model.") as local:
            self.ask("Where do zebras live?")
        claude_call.assert_not_called()
        local.assert_called_once()
        self.assertEqual(self.reply().answered_by, "ollama")

    def test_quoted_answer_picks_the_matching_sentence(self):
        with mock.patch("rag.llm.generate", side_effect=LLMError("429 quota")):
            self.ask("Why do zebras have black and white stripes?")
        text = self.reply().content
        self.assertIn("stripes confuse biting flies", text)
        self.assertIn("[zebras.txt, page 1]", text)

    def test_quoted_answer_handles_greetings(self):
        with mock.patch("rag.llm.generate", side_effect=LLMError("429 quota")):
            self.ask("hello")
        self.assertIn("I can answer questions about Acme", self.reply().content)

    def test_turning_gemini_off_never_calls_it(self):
        self.org.use_external_ai = False
        self.org.save()
        with mock.patch("rag.llm.generate") as gemini:
            self.ask("Where do zebras live?")
        gemini.assert_not_called()
        reply = self.reply()
        self.assertEqual((reply.answered_by, reply.error_detail), ("quoted", ""))  # chosen, not a failure

    def test_backup_answers_are_not_cached(self):
        with mock.patch("rag.llm.generate", side_effect=LLMError("429 quota")):
            self.ask("Where do zebras live?")
        with mock.patch("rag.llm.generate", return_value="Gemini is back.") as gemini:
            self.ask("Where do zebras live?")
        gemini.assert_called_once()
        self.assertEqual(self.reply().answered_by, "gemini")

    def test_admins_are_told_the_backup_was_used(self):
        with mock.patch("rag.llm.generate", side_effect=LLMError("429 quota exceeded")):
            self.ask("Where do zebras live?")
        home = self.client.get(reverse("dashboard"))
        self.assertContains(home, "Gemini quota reached: 1 answer came from the backup")
        self.assertEqual(home.context["ai_status"]["label"], "Using backup")
