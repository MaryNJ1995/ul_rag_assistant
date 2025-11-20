from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Optional

from openai import OpenAI

from ul_rag_assistant.ul_rag.config import get_settings
from ul_rag_assistant.ul_rag.logging import get_logger

log = get_logger(__name__)

QueryType = Literal[
    "who_is",
    "programme_or_module",
    "campus_directions",
    "admin_process",
    "research",
    "general",
    "chitchat",
    "nonsense",
]

RetrievalMode = Literal["hybrid", "dense_only", "sparse_only"]


@dataclass
class QueryPlan:
    query_type: QueryType
    topic: str
    needs_multi_hop: bool
    retrieval_mode: RetrievalMode
    max_chunks: int
    domain_hint: Optional[str]


class Router:
    def __init__(self):
        settings = get_settings()
        if not settings.openai_api_key:
            self.client = None
        else:
            self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.gen_model

    def _default_plan(self, question: str) -> QueryPlan:
        q_lower = question.lower()
        topic = ""
        if "lero" in q_lower:
            topic = "lero"
        elif "csis" in q_lower:
            topic = "csis"
        elif "accommodation" in q_lower:
            topic = "accommodation"
        return QueryPlan(
            query_type="general",
            topic=topic,
            needs_multi_hop=False,
            retrieval_mode="hybrid",
            max_chunks=6,
            domain_hint=None,
        )

    def _system_prompt(self) -> str:
        return (
            "You are an intent classifier and planner for a University of Limerick (UL) assistant.\n\n"
            "You must look at the USER MESSAGE and decide:\n"
            "1) What kind of message it is.\n"
            "2) If it is a UL question, what high-level type and topic it has.\n\n"
            "You MUST choose one of these values for query_type:\n"
            "- 'who_is'              : asking about a person (staff, lecturer, professor, researcher, etc.)\n"
            "- 'programme_or_module' : asking about a degree programme, course, module, or subject\n"
            "- 'campus_directions'   : asking about campus map, directions, locations, buildings, transport, parking\n"
            "- 'admin_process'       : asking about admissions, registration, exams, fees, regulations, policies\n"
            "- 'research'            : asking about research centres, Lero, SFI Research Centre for Software, grants, projects\n"
            "- 'general'             : UL-related question that does not fit the above categories\n"
            "- 'chitchat'            : greeting / small talk / social message (e.g. 'hi', 'hello', 'thanks', 'how are you') "
            "that is NOT clearly asking for UL information\n"
            "- 'nonsense'            : mostly random characters, spam, or clearly not understandable as a UL-related question\n\n"
            "Additional fields:\n"
            "- topic: a short keyword for the main topic, or '' if none.\n"
            "- needs_multi_hop: true if the question clearly requires combining information from multiple documents.\n"
            "- retrieval_mode: one of 'hybrid', 'dense_only', 'sparse_only' (use 'hybrid' for most questions).\n"
            "- max_chunks: integer, approx number of chunks to retrieve (e.g. 4, 6, 8).\n"
            "- domain_hint: optional host/domain preference (e.g. 'pure.ul.ie', 'ul.ie/buildings'), or null if no preference.\n\n"
            "You MUST respond with ONLY a single JSON object, no extra text."
        )

    def _parse_json(self, content: str) -> QueryPlan:
        raw = content.strip()
        if not raw:
            raise ValueError("Empty router output")
        try:
            data = json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object found in router output")
            data = json.loads(raw[start : end + 1])

        allowed_types = {
            "who_is",
            "programme_or_module",
            "campus_directions",
            "admin_process",
            "research",
            "general",
            "chitchat",
            "nonsense",
        }
        qt_raw = data.get("query_type", "general")
        qt: QueryType = qt_raw if qt_raw in allowed_types else "general"  # type: ignore[assignment]

        rm_raw = data.get("retrieval_mode", "hybrid") or "hybrid"
        rm: RetrievalMode = rm_raw if rm_raw in {"hybrid", "dense_only", "sparse_only"} else "hybrid"  # type: ignore[assignment]

        topic = data.get("topic", "")
        if not isinstance(topic, str):
            topic = ""

        needs_multi_hop = bool(data.get("needs_multi_hop", False))

        max_chunks = data.get("max_chunks", 6)
        try:
            max_chunks_int = int(max_chunks)
        except Exception:
            max_chunks_int = 6

        dh = data.get("domain_hint")
        domain_hint = dh if isinstance(dh, str) and dh else None

        return QueryPlan(
            query_type=qt,
            topic=topic,
            needs_multi_hop=needs_multi_hop,
            retrieval_mode=rm,
            max_chunks=max_chunks_int,
            domain_hint=domain_hint,
        )

    def route(self, question: str) -> QueryPlan:
        if self.client is None:
            log.warning("Router: no OpenAI client configured, using default plan.")
            return self._default_plan(question)

        system = self._system_prompt()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"USER MESSAGE:\n{question}"},
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
        except Exception as e:
            log.warning(f"Router: OpenAI call failed, using default plan: {e}")
            return self._default_plan(question)

        try:
            plan = self._parse_json(content)
        except Exception as e:
            log.warning(f"Router: failed to parse JSON plan, using default plan: {e}")
            plan = self._default_plan(question)

        q_lower = question.lower()
        if plan.query_type == "who_is" and plan.domain_hint is None:
            plan.domain_hint = "pure.ul.ie"
        elif plan.query_type == "campus_directions" and plan.domain_hint is None:
            plan.domain_hint = "ul.ie"

        return plan
