#!/usr/bin/env python
"""
DeepEval-based evaluation for the UL RAG assistant.

This uses the RAG triad:
  - Answer Relevancy
  - Faithfulness
  - Contextual Relevancy
on My UL-specific eval dataset (ul_eval.jsonl).
"""

import json
from typing import List, Dict, Any

from tqdm import tqdm
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from deepeval import evaluate

from ul_rag_assistant.ul_rag.graph.graph import run_ul_rag_debug


# Limit how many eval questions to run to save quota
MAX_EVAL: int | None = 20  # set to None to use all


def load_eval_data(path: str) -> List[Dict[str, Any]]:
    """Load My UL eval dataset from JSONL."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj["question"].strip()
            gt = obj.get("ground_truth", "").strip()
            data.append({"question": q, "ground_truth": gt})
    return data


def build_llm_test_cases(eval_data: List[Dict[str, Any]]) -> List[LLMTestCase]:
    """
    For each eval question, run the UL RAG pipeline,
    then wrap (question, answer, contexts) into DeepEval's LLMTestCase.
    """
    test_cases: List[LLMTestCase] = []

    for row in tqdm(eval_data, desc="Building test cases"):
        q = row["question"]

        # Call My RAG pipeline (debug version that returns contexts)
        resp = run_ul_rag_debug(q, mode="student", locale="IE")

        answer = resp["answer"]
        contexts = resp.get("contexts", [])  # list[str]

        # DeepEval expects list[str] for retrieval_context
        test_case = LLMTestCase(
            input=q,
            actual_output=answer,
            retrieval_context=contexts,
        )
        test_cases.append(test_case)

    return test_cases


def main():
    eval_path = "data/eval/ul_eval.jsonl"
    eval_data = load_eval_data(eval_path)
    print(f"Loaded {len(eval_data)} eval questions from {eval_path}")

    # Apply cap to reduce cost
    if MAX_EVAL is not None and len(eval_data) > MAX_EVAL:
        eval_data = eval_data[:MAX_EVAL]
        print(f"Using first {MAX_EVAL} questions for DeepEval (to reduce quota usage).")

    test_cases = build_llm_test_cases(eval_data)

    # --- Define RAG metrics (RAG triad) with cheaper model ---
    # Change model to any cheaper OpenAI model you have (e.g. gpt-4o-mini)
    judge_model = "gpt-4o-mini"

    answer_relevancy = AnswerRelevancyMetric(model=judge_model)
    faithfulness = FaithfulnessMetric(model=judge_model)
    contextual_relevancy = ContextualRelevancyMetric(model=judge_model)

    metrics = [answer_relevancy, faithfulness, contextual_relevancy]

    # --- Run evaluation ---
    results = evaluate(test_cases, metrics=metrics)

    # Print per-test-case scores
    for i, res in enumerate(results):
        print(f"\n=== Test case {i+1}: {eval_data[i]['question']} ===")
        for metric_name, score in res.items():
            print(f"{metric_name}: {score}")

    # Print averages
    print("\n=== Averages ===")
    # 'results' is a list of dicts: {metric_name: score, ...}
    # We'll compute simple arithmetic mean per metric
    metric_names = results[0].keys() if results else []
    for name in metric_names:
        scores = [r[name] for r in results]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"{name}: {avg:.3f}")


if __name__ == "__main__":
    main()
