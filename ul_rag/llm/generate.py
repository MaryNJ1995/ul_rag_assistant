from typing import List, Dict, Any

from openai import OpenAI

from ul_rag_assistant.ul_rag.llm.prompts import STUDENT_SYSTEM, STAFF_SYSTEM, USER_TEMPLATE
from ul_rag_assistant.ul_rag.config import get_settings
from ul_rag_assistant.ul_rag.logging import get_logger

log = get_logger(__name__)


class Generator:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.gen_model
        if not settings.openai_api_key:
            self.client = None
        else:
            self.client = OpenAI(api_key=settings.openai_api_key)

    async def answer(
        self,
        question: str,
        ctx: List[Dict[str, Any]],
        mode: str = "student",
        locale: str = "IE",
    ) -> Dict[str, Any]:
        if not ctx:
            return {
                "answer": (
                    "I couldn't find any University of Limerick documents that match your question. "
                    "This usually means the site or page hasn't been ingested yet."
                ),
                "citations": [],
                "meta": {"model": None},
            }

        context_str, cites = self._format_context(ctx)
        system = STUDENT_SYSTEM if mode == "student" else STAFF_SYSTEM
        user = USER_TEMPLATE.format(question=question, context=context_str)

        if self.client is None:
            log.warning("OPENAI_API_KEY not set, using fallback summarisation.")
            answer = self._fallback_answer(question, ctx)
        else:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.3,
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                log.warning(f"OpenAI call failed, using fallback summarisation: {e}")
                answer = self._fallback_answer(question, ctx)

        return {"answer": answer, "citations": cites, "meta": {"model": self.model}}

    async def answer_chitchat(self, question: str, mode: str = "student", locale: str = "IE") -> str:
        if self.client is None:
            if mode == "student":
                return "Hi! I'm the University of Limerick assistant. Ask me anything about UL whenever you're ready."
            else:
                return "Hello. I'm the University of Limerick assistant. Let me know if you have any UL-related questions."

        system = (
            "You are a friendly assistant for the University of Limerick.\n"
            "The user message is a greeting or small-talk.\n"
            "Respond with 1–2 short, natural sentences.\n"
            "Do NOT add sources, citations, or 'Next steps'."
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            temperature=0.6,
        )
        return resp.choices[0].message.content.strip()

    async def answer_nonsense(self, question: str, mode: str = "student", locale: str = "IE") -> str:
        if self.client is None:
            return (
                "I'm not sure what you meant there. "
                "I can help with questions about the University of Limerick if you'd like to ask one."
            )

        system = (
            "You are an assistant for the University of Limerick.\n"
            "The user message is mostly gibberish, nonsense, or not clearly understandable as a question.\n"
            "You must NOT invent any UL information.\n"
            "Respond briefly (1–3 sentences), saying you didn't understand and inviting the user to ask a clear UL-related question.\n"
            "Do NOT add sources, citations, or 'Next steps'."
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def _strip_frontmatter(self, text: str) -> str:
        stripped = text.lstrip()
        if stripped.startswith("---"):
            parts = stripped.split("---", 2)
            if len(parts) == 3:
                return parts[2]
        return text

    def _shorten(self, text: str, max_len: int = 600) -> str:
        text = " ".join(text.split())
        if len(text) <= max_len:
            return text
        return text[: max_len].rsplit(" ", 1)[0] + "…"

    def _format_context(self, ctx: List[Dict[str, Any]]):
        lines = []
        cites = []
        for i, item in enumerate(ctx, start=1):
            raw = item.get("text", "")
            snippet = self._strip_frontmatter(raw)
            snippet = self._shorten(snippet, 550)
            meta = item.get("meta", {}) or {}
            path = meta.get("source_url") or meta.get("path") or meta.get("source") or "document"
            lines.append(f"[{i}] {snippet}\n(Source: {path})\n")
            cites.append({"n": i, "source": path})
        return "\n".join(lines), cites

    def _fallback_answer(self, question: str, ctx: List[Dict[str, Any]]) -> str:
        snippets = []
        for i, item in enumerate(ctx[:3], start=1):
            meta = item.get("meta", {}) or {}
            path = meta.get("source_url") or meta.get("path") or meta.get("source") or "document"
            snippet = self._strip_frontmatter(item.get("text", ""))
            snippet = self._shorten(snippet, 350)
            snippets.append(f"From source {i} ({path}): {snippet}")
        joined = "\n\n".join(snippets) if snippets else "(no text available)"
        return (
            "I can't use the language model right now because no OpenAI API key is configured.\n\n"
            "Here is a short summary of the most relevant University of Limerick information I could find:\n\n"
            f"{joined}"
        )
