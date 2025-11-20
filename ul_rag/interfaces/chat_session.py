from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, List, Dict, Any, Optional
from datetime import datetime

from ul_rag_assistant.ul_rag.graph.graph import run_ul_rag

Role = Literal["user", "assistant"]


@dataclass
class ChatTurn:
    role: Role
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class RAGChatSession:
    """Predefined chat interface around the UL RAG system."""

    def __init__(
        self,
        mode: Literal["student", "staff"] = "student",
        locale: str = "IE",
        session_id: Optional[str] = None,
    ) -> None:
        self.mode = mode
        self.locale = locale
        self.session_id = session_id or "default"
        self.history: List[ChatTurn] = []

    def reset(self) -> None:
        self.history.clear()

    def set_mode(self, mode: Literal["student", "staff"]) -> None:
        self.mode = mode

    def set_locale(self, locale: str) -> None:
        self.locale = locale

    def ask(self, text: str) -> ChatTurn:
        user_turn = ChatTurn(role="user", content=text)
        self.history.append(user_turn)

        resp = run_ul_rag(question=text, mode=self.mode, locale=self.locale)

        bot_content = resp.get("answer", "") or "Sorry, I could not generate an answer."
        bot_citations = resp.get("citations", [])
        bot_meta = resp.get("meta", {})

        bot_turn = ChatTurn(
            role="assistant",
            content=bot_content,
            citations=bot_citations,
            meta=bot_meta,
        )
        self.history.append(bot_turn)
        return bot_turn

    def get_history(self) -> List[ChatTurn]:
        return list(self.history)
