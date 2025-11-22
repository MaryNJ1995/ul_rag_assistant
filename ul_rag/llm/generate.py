from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI

from ul_rag_assistant.ul_rag.llm.prompts import STUDENT_SYSTEM, STAFF_SYSTEM, USER_TEMPLATE
from ul_rag_assistant.ul_rag.config import get_settings
from ul_rag_assistant.ul_rag.logging import get_logger

log = get_logger(__name__)


class Generator:
    """
    High-level answer generator for the UL RAG assistant.

    Responsibilities:
      - Format retrieved context chunks into a compact, LLM-friendly string.
      - Call the OpenAI Chat API with a strict system prompt and user template.
      - Provide special handling for chit-chat and nonsense queries.
      - Be robust to malformed or heterogeneous context objects (e.g. dicts,
        LangChain Documents, missing text/metadata, None values).
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.gen_model
        if not settings.openai_api_key:
            log.warning("OPENAI_API_KEY not set; generator will use fallback summaries.")
            self.client: Optional[OpenAI] = None
        else:
            self.client = OpenAI(api_key=settings.openai_api_key)

    # -------------------------------------------------------------------------
    # Public answer methods
    # -------------------------------------------------------------------------

    async def answer(
        self,
        question: str,
        ctx: List[Dict[str, Any]],
        mode: str = "student",
        locale: str = "IE",
    ) -> Dict[str, Any]:
        """
        Main entry point used by the RAG graph.

        Parameters
        ----------
        question : str
            The user question.
        ctx : list of dict-like
            Retrieved documents; each should carry text and metadata.
        mode : {"student", "staff"}
        locale : e.g. "IE"

        Returns
        -------
        dict
            {"answer": str, "citations": list, "meta": dict}
        """
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

        # If for some reason all context items were unusable, degrade gracefully
        if not context_str.strip():
            log.warning("All retrieved docs had empty or unusable text; returning no-context answer.")
            return {
                "answer": (
                    "I retrieved some documents, but none of them contained readable text. "
                    "This might indicate a parsing issue with the source pages. "
                    "Please try rephrasing your question or consult the official UL website."
                ),
                "citations": [],
                "meta": {"model": None},
            }

        system = STUDENT_SYSTEM if mode == "student" else STAFF_SYSTEM
        user = USER_TEMPLATE.format(question=question, context=context_str)

        if self.client is None:
            log.warning("OPENAI_API_KEY not set, using fallback summarisation.")
            answer = self._fallback_answer(question, ctx)
            return {"answer": answer, "citations": cites, "meta": {"model": None}}

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            log.warning(f"OpenAI call failed in Generator.answer, using fallback summarisation: {e}")
            answer = self._fallback_answer(question, ctx)

        return {"answer": answer, "citations": cites, "meta": {"model": self.model}}

    async def answer_chitchat(self, question: str, mode: str = "student", locale: str = "IE") -> str:
        """
        Special path for greetings / small talk.

        No retrieval, no citations, just a short friendly response.
        """
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
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()

    async def answer_nonsense(self, question: str, mode: str = "student", locale: str = "IE") -> str:
        """
        Special path for nonsensical / gibberish input.

        The model politely says it didn't understand and invites a clearer,
        UL-related question. No retrieval and no citations.
        """
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
            temperature=0.1,
        )
        return (resp.choices[0].message.content or "").strip()

    # -------------------------------------------------------------------------
    # Internal helpers: context extraction & formatting
    # -------------------------------------------------------------------------

    def _extract_text_and_meta(self, item: Any) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Robustly extract (text, meta) from a context item.

        Supports several shapes, e.g.:
          - {"text": ..., "meta": {...}}
          - {"page_content": ..., "metadata": {...}}
          - LangChain-style Document objects with .page_content / .metadata
        """
        text = None
        meta: Dict[str, Any] = {}

        # Case 1: dict-like
        if isinstance(item, dict):
            # Prefer "text", fall back to "page_content"
            text = item.get("text") or item.get("page_content")
            raw_meta = item.get("meta") or item.get("metadata") or {}

        else:
            # Case 2: object with attributes (e.g. Document)
            text = getattr(item, "text", None) or getattr(item, "page_content", None)
            raw_meta = getattr(item, "meta", None) or getattr(item, "metadata", None) or {}

        # Normalise text
        if text is None:
            # No usable text
            return None, {}

        if not isinstance(text, str):
            text = str(text)

        # Normalise metadata to a dict
        if isinstance(raw_meta, dict):
            meta = raw_meta
        else:
            # If it's None or some other type, wrap it
            if raw_meta not in (None, {}):
                meta = {"_raw_meta": raw_meta}
            else:
                meta = {}

        return text, meta

    def _strip_frontmatter(self, text: Any) -> str:
        """
        Remove leading YAML frontmatter if present; be robust to None/non-strings.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        stripped = text.lstrip()
        if stripped.startswith("---"):
            parts = stripped.split("---", 2)
            if len(parts) == 3:
                return parts[2]
        return text

    def _shorten(self, text: str, max_len: int = 600) -> str:
        """
        Collapse whitespace and truncate to at most max_len characters,
        cutting at a word boundary where possible.
        """
        text = " ".join(text.split())
        if len(text) <= max_len:
            return text
        return text[: max_len].rsplit(" ", 1)[0] + "…"

    def _format_context(self, ctx: List[Dict[str, Any]]):
        lines: List[str] = []
        cites: List[Dict[str, Any]] = []

        idx = 1
        for item in ctx:
            # 1. Robust text extraction
            raw = item.get("text")
            if raw is None:
                raw = item.get("page_content")
            if raw is None:
                # Nothing usable; skip
                continue

            snippet = self._strip_frontmatter(raw)
            snippet = self._shorten(snippet, 550)

            # 2. Robust metadata handling
            meta = item.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}

            path = (
                meta.get("source_url")
                or meta.get("path")
                or meta.get("source")
                or meta.get("url")
                or "document"
            )

            lines.append(f"[{idx}] {snippet}\n(Source: {path})\n")
            cites.append({"n": idx, "source": path})
            idx += 1

        return "\n".join(lines), cites


    # -------------------------------------------------------------------------
    # Fallback answer when no OpenAI key or API failure
    # -------------------------------------------------------------------------

    def _fallback_answer(self, question: str, ctx: List[Dict[str, Any]]) -> str:
        """
        Simple summarisation of a few top context snippets, used when
        no OpenAI API key is configured or the API call fails.

        This never calls the LLM itself; it is purely string processing.
        """
        snippets: List[str] = []

        for i, raw_item in enumerate(ctx[:3], start=1):
            text, meta = self._extract_text_and_meta(raw_item)
            if not text:
                continue

            snippet = self._strip_frontmatter(text)
            snippet = self._shorten(snippet, 350)

            source_url = None
            if isinstance(meta, dict):
                source_url = (
                    meta.get("source_url")
                    or meta.get("url")
                    or meta.get("path")
                    or meta.get("source")
                )
            path = source_url or "document"

            snippets.append(f"From source {i} ({path}): {snippet}")

        joined = "\n\n".join(snippets) if snippets else "(no text available)"
        return (
            "I can't use the language model right now because no OpenAI API key is configured "
            "or the model call failed.\n\n"
            "Here is a short summary of the most relevant University of Limerick information I could find:\n\n"
            f"{joined}"
        )
