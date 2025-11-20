from __future__ import annotations

from ul_rag_assistant.ul_rag.graph.graph import run_ul_rag
#!/usr/bin/env python
import argparse
import json
from typing import Any, Dict



def infer(
    question: str,
    mode: str = "student",
    locale: str = "IE",
) -> Dict[str, Any]:
    """
    Convenience wrapper around `run_ul_rag`.

    Parameters
    ----------
    question : str
        The user question (about the University of Limerick).
    mode : str
        "student" or "staff". Controls the system prompt style.
    locale : str
        Locale string (e.g. "IE"). Reserved for future localisation.

    Returns
    -------
    dict
        {
          "answer": str,
          "citations": List[{"n": int, "source": str}],
          "mode": str,
          "meta": dict,
          "plan": dict or None
        }
    """
    return run_ul_rag(question=question, mode=mode, locale=locale)





def format_output(resp: Dict[str, Any], show_plan: bool, show_meta: bool, show_citations: bool) -> str:
    lines = []

    answer = resp.get("answer", "") or "(no answer)"
    lines.append("=== ANSWER ===")
    lines.append(answer)
    lines.append("")

    if show_citations:
        lines.append("=== CITATIONS ===")
        citations = resp.get("citations") or []
        if not citations:
            lines.append("(no citations)")
        else:
            for c in citations:
                n = c.get("n", "?")
                src = c.get("source", "unknown source")
                lines.append(f"[{n}] {src}")
        lines.append("")

    if show_plan:
        lines.append("=== ROUTER PLAN ===")
        plan = resp.get("plan")
        if plan is None:
            lines.append("(no plan or router disabled)")
        else:
            lines.append(json.dumps(plan, indent=2, ensure_ascii=False))
        lines.append("")

    if show_meta:
        lines.append("=== META ===")
        meta = resp.get("meta") or {}
        lines.append(json.dumps(meta, indent=2, ensure_ascii=False))
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="One-shot UL RAG inferencer (Maryam Najafi’s UL assistant)."
    )

    parser.add_argument(
        "--mode",
        choices=["student", "staff"],
        default="student",
        help="Prompt mode (student or staff).",
    )
    parser.add_argument(
        "--locale",
        default="IE",
        help="Locale string (default: IE).",
    )
    parser.add_argument(
        "--no-citations",
        action="store_true",
        help="Do not show citations in the output.",
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the router QueryPlan (intent, retrieval settings).",
    )
    parser.add_argument(
        "--show-meta",
        action="store_true",
        help="Print meta information from the graph (e.g., intent flags).",
    )

    args = parser.parse_args()
    question = "compare lero with research ireland and csis"

    resp = infer(
        question=question,
        mode=args.mode,
        locale=args.locale,
    )

    text = format_output(
        resp,
        show_plan=args.show_plan,
        show_meta=args.show_meta,
        show_citations=not args.no_citations,
    )
    print(text)


if __name__ == "__main__":
    main()
