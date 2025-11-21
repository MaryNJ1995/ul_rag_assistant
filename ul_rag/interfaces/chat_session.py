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
    """Predefined chat interface around the UL RAG system, with light conversational memory."""

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
        """Clear the conversation history."""
        self.history.clear()

    def set_mode(self, mode: Literal["student", "staff"]) -> None:
        self.mode = mode

    def set_locale(self, locale: str) -> None:
        self.locale = locale

    # ---------- NEW: memory helper ----------

    def _build_query_with_context(self, new_question: str) -> str:
        """
        Very simple conversational memory:

        - If there is a previous user question in the history, prepend it as:
            "Previous question: <...>\\nFollow-up question: <...>"
        - This helps the RAG pipeline resolve pronouns like 'he', 'she', 'they', 'it'
          in short follow-up questions (e.g. 'what he does?').

        We only look at past history; we do NOT mutate it here.
        """
        last_user_msg: Optional[str] = None

        # Walk history backwards to find the *previous* user message
        for turn in reversed(self.history):
            if turn.role == "user":
                last_user_msg = turn.content
                break

        if last_user_msg is None:
            # This is the first user message in the conversation
            return new_question

        # Combine previous user question with the new follow-up
        combined = (
            f"Previous question: {last_user_msg}\n"
            f"Follow-up question: {new_question}"
        )
        return combined

    # ---------- Main entry point ----------

    def ask(self, text: str) -> ChatTurn:
        """
        Add the user's message to history, then call the RAG graph
        with a question that includes minimal conversational context.
        """
        # Build the query that will be sent to RAG (with memory)
        question_with_context = self._build_query_with_context(text)

        # Store the *raw* user message in history (what the human actually typed)
        user_turn = ChatTurn(role="user", content=text)
        self.history.append(user_turn)

        # Call the RAG pipeline
        resp = run_ul_rag(question=question_with_context, mode=self.mode, locale=self.locale)

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
        """Return a copy of the conversation history."""
        return list(self.history)
