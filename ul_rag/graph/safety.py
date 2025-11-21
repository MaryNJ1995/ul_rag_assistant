from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ul_rag_assistant.ul_rag.logging import get_logger

log = get_logger(__name__)


@dataclass
class SafetyResult:
    escalate: bool
    reason: Optional[str] = None


class Safety:
    """Very lightweight safety / escalation checker."""

    def check_escalation(self, text: str) -> SafetyResult:
        lowered = text.lower()
        crisis_keywords = ["suicide", "kill myself", "self-harm", "end my life"]
        if any(k in lowered for k in crisis_keywords):
            return SafetyResult(escalate=True, reason="crisis")
        return SafetyResult(escalate=False)

    def escalation_message(self, locale: str = "IE") -> str:
        return (
            "I'm really sorry you're feeling this way. I'm not able to provide the help you deserve. "
            "If you are at risk / suicidal please immediately contact either the crisis liaison mental health team at the University Hospital Limerick (061 301111) or your local hospital, or your GP immediately. ."
        )
