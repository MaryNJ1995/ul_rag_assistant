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

    # ---------- NEW: memory helper (last 2 user questions) ----------

    def _build_query_with_context(self, new_question: str) -> str:
        """
        Simple conversational memory:

        - Look back through history and collect up to the last TWO user messages.
        - If none exist, just return the new question.
        - Otherwise, build a prompt like:

            Previous questions:
            1) <older previous question>
            2) <most recent previous question>
            Current question: <new question>

        This helps the RAG pipeline resolve pronouns like 'he', 'she', 'they', 'it'
        across multiple follow-up questions.
        """

        # Collect up to 2 previous user messages from history (IN REVERSE ORDER)
        prev_user_msgs: List[str] = []
        for turn in reversed(self.history):
            if turn.role == "user":
                prev_user_msgs.append(turn.content)
                if len(prev_user_msgs) == 2:
                    break

        # No previous messages -> no context, just return current question
        if not prev_user_msgs:
            return new_question

        # We collected in reverse order (most recent first), so flip to chronological
        prev_user_msgs = list(reversed(prev_user_msgs))

        # Build the combined query
        lines: List[str] = []

        if len(prev_user_msgs) == 1:
            # Only one previous question
            lines.append(f"Previous question: {prev_user_msgs[0]}")
        else:
            # Two previous questions
            lines.append("Previous questions:")
            lines.append(f"1) {prev_user_msgs[0]}")
            lines.append(f"2) {prev_user_msgs[1]}")

        lines.append(f"Current question: {new_question}")

        return "\n".join(lines)

    # ---------- Main entry point ----------

    def ask(self, text: str) -> ChatTurn:
        """
        Add the user's message to history, then call the RAG graph
        with a question that includes minimal conversational context.
        """
        # IMPORTANT: build query-with-context *before* appending this user turn,
        # so history contains only previous messages.
        question_with_context = self._build_query_with_context(text)

        # Store the raw user message in history
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
