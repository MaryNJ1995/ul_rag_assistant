# ul_rag/graph/graph.py
from __future__ import annotations

from typing import List, Literal, TypedDict, Optional, Dict, Any
import asyncio

from langgraph.graph import StateGraph, END

from ..retrieval.retriever import Retriever
from ..llm.generate import Generator
from .safety import Safety
from .router import Router, QueryPlan
from ..logging import get_logger

log = get_logger(__name__)

retriever = Retriever()
generator = Generator()
safety = Safety()
router = Router()


class ULDoc(TypedDict):
    text: str
    meta: Dict[str, Any]


class ULState(TypedDict):
    question: str
    mode: Literal["student", "staff"]
    locale: str
    plan: Optional[Dict[str, Any]]
    docs: List[ULDoc]
    answer: Optional[str]
    citations: List[Dict[str, Any]]
    meta: Dict[str, Any]


def safety_node(state: ULState) -> ULState:
    esc = safety.check_escalation(state["question"])
    if esc.escalate:
        answer = safety.escalation_message(state["locale"])
        return {
            **state,
            "answer": answer,
            "citations": [],
            "meta": {"escalation": esc.reason},
        }
    return state


def route_node(state: ULState) -> ULState:
    plan: QueryPlan = router.route(state["question"])
    return {
        **state,
        "plan": {
            "query_type": plan.query_type,
            "topic": plan.topic,
            "needs_multi_hop": plan.needs_multi_hop,
            "retrieval_mode": plan.retrieval_mode,
            "max_chunks": plan.max_chunks,
            "domain_hint": plan.domain_hint,
        },
    }


def retrieve_node(state: ULState) -> ULState:
    plan_dict = state.get("plan") or {}
    qp = QueryPlan(
        query_type=plan_dict.get("query_type", "general"),  # type: ignore[arg-type]
        topic=plan_dict.get("topic", "ul"),
        needs_multi_hop=bool(plan_dict.get("needs_multi_hop", False)),
        retrieval_mode=plan_dict.get("retrieval_mode", "hybrid"),  # type: ignore[arg-type]
        max_chunks=int(plan_dict.get("max_chunks", 10)),
        domain_hint=plan_dict.get("domain_hint"),
    )

    if qp.query_type in ("chitchat", "nonsense"):
        return {**state, "docs": []}

    docs = retriever.retrieve(state["question"], max_chunks=qp.max_chunks, domain_hint=qp.domain_hint)
    return {**state, "docs": docs}


def generate_node(state: ULState) -> ULState:
    # If safety already set an answer, do nothing
    if state.get("answer") is not None:
        return state

    plan_dict = state.get("plan") or {}
    query_type = plan_dict.get("query_type", "general")

    if query_type == "chitchat":
        ans = asyncio.run(
            generator.answer_chitchat(
                state["question"],
                mode=state["mode"],
                locale=state["locale"],
            )
        )
        return {**state, "answer": ans, "citations": [], "meta": {"intent": "chitchat"}}

    if query_type == "nonsense":
        ans = asyncio.run(
            generator.answer_nonsense(
                state["question"],
                mode=state["mode"],
                locale=state["locale"],
            )
        )
        return {**state, "answer": ans, "citations": [], "meta": {"intent": "nonsense"}}

    docs = state.get("docs") or []
    if not docs:
        return {
            **state,
            "answer": (
                "Sorry, I couldn't find any University of Limerick documents clearly "
                "related to that question. Try rephrasing it, or check the official UL "
                "website or department directly."
            ),
            "citations": [],
            "meta": {"ctx": 0},
        }

    resp = asyncio.run(
        generator.answer(
            state["question"],
            docs,
            mode=state["mode"],
            locale=state["locale"],
        )
    )
    return {
        **state,
        "answer": resp.get("answer"),
        "citations": resp.get("citations", []),
        "meta": resp.get("meta", {}),
    }


def build_ul_graph():
    workflow = StateGraph(ULState)
    workflow.add_node("safety", safety_node)
    workflow.add_node("route", route_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("safety")
    workflow.add_edge("safety", "route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


ul_graph = build_ul_graph()


def run_ul_rag(question: str, mode: str = "student", locale: str = "IE") -> Dict[str, Any]:
    initial_state: ULState = {
        "question": question,
        "mode": mode,  # type: ignore[assignment]
        "locale": locale,
        "plan": None,
        "docs": [],
        "answer": None,
        "citations": [],
        "meta": {},
    }
    final_state = ul_graph.invoke(initial_state)
    return {
        "answer": final_state["answer"],
        "citations": final_state["citations"],
        "mode": final_state["mode"],
        "meta": final_state["meta"],
        "plan": final_state.get("plan"),
    }


def run_ul_rag_debug(question: str, mode: str = "student", locale: str = "IE") -> Dict[str, Any]:
    """
    Evaluation-friendly path that returns contexts as plain text and plan.
    """
    plan = router.route(question)
    if plan.query_type in ("chitchat", "nonsense"):
        docs = []
    else:
        docs = retriever.retrieve(question, max_chunks=plan.max_chunks, domain_hint=plan.domain_hint)

    gen_resp = asyncio.run(generator.answer(question, docs, mode=mode, locale=locale))
    answer = gen_resp["answer"]
    citations = gen_resp["citations"]
    contexts = [d.get("text", "") for d in docs]

    return {
        "answer": answer,
        "contexts": contexts,
        "citations": citations,
        "meta": {
            "mode": mode,
            "locale": locale,
            "plan": {
                "query_type": plan.query_type,
                "topic": plan.topic,
                "needs_multi_hop": plan.needs_multi_hop,
                "retrieval_mode": plan.retrieval_mode,
                "max_chunks": plan.max_chunks,
                "domain_hint": plan.domain_hint,
            },
    },
    }
