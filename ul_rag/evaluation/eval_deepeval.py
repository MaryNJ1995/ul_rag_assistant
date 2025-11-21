#!/usr/bin/env python
"""
DeepEval-based evaluation for the UL RAG assistant.

This uses the RAG triad:
  - Answer Relevancy
  - Faithfulness
  - Contextual Relevancy
on my UL-specific eval dataset (ul_eval.jsonl).

It evaluates each question individually and saves
per-question results into an Excel file.
"""

import json
from itertools import islice
from typing import List, Dict, Any

from tqdm import tqdm
import pandas as pd

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)

from ul_rag_assistant.ul_rag.graph.graph import run_ul_rag_debug


# Limit how many eval questions to run to save quota.
# Set to None to use all.
MAX_EVAL: int | None = 20

# Cheaper OpenAI judge model
JUDGE_MODEL = "gpt-4o-mini"


def load_eval_data(path: str, max_rows:int=20) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in islice(f, max_rows):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj["question"].strip()
            gt = obj["ground_truth"].strip()
            data.append({"question": q, "ground_truth": gt})
    return data


def main():
    eval_path = "/home/maryam_najafi/ul_bot/ul_rag_assistant/data/eval/ul_eval.jsonl"
    out_excel = "/home/maryam_najafi/ul_bot/ul_rag_assistant/data/eval/deepeval_results.xlsx"

    eval_data = load_eval_data(eval_path)
    print(f"Loaded {len(eval_data)} eval questions from {eval_path}")

    # Apply cap to reduce cost
    if MAX_EVAL is not None and len(eval_data) > MAX_EVAL:
        eval_data = eval_data[:MAX_EVAL]
        print(f"Using first {MAX_EVAL} questions for DeepEval (to reduce quota usage).")

    rows: List[Dict[str, Any]] = []

    for row in tqdm(eval_data, desc="Evaluating with DeepEval"):
        question = row["question"]
        ground_truth = row["ground_truth"]

        # 1) Run RAG pipeline
        resp = run_ul_rag_debug(question, mode="student", locale="IE")
        answer = resp.get("answer", "")
        contexts = resp.get("contexts", [])  # list[str]

        # 2) Build DeepEval test case
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth if ground_truth else None,
            retrieval_context=contexts,
        )

        # 3) Define metrics for THIS test case, using cheaper judge model
        answer_relevancy = AnswerRelevancyMetric(model=JUDGE_MODEL)
        faithfulness = FaithfulnessMetric(model=JUDGE_MODEL)
        contextual_relevancy = ContextualRelevancyMetric(model=JUDGE_MODEL)

        # 4) Safely measure metrics (don’t crash if quota/rate errors)
        def safe_measure(metric, name: str):
            try:
                metric.measure(test_case)
                return metric.score, metric.reason
            except Exception as e:
                # If quota/rate limit/etc., record failure instead of aborting
                return None, f"{name} failed: {e}"

        ans_rel_score, ans_rel_reason = safe_measure(answer_relevancy, "answer_relevancy")
        faith_score, faith_reason = safe_measure(faithfulness, "faithfulness")
        ctx_rel_score, ctx_rel_reason = safe_measure(contextual_relevancy, "contextual_relevancy")

        rows.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "contexts": " ||| ".join(contexts),
                "answer_relevancy_score": ans_rel_score,
                "answer_relevancy_reason": ans_rel_reason,
                "faithfulness_score": faith_score,
                "faithfulness_reason": faith_reason,
                "contextual_relevancy_score": ctx_rel_score,
                "contextual_relevancy_reason": ctx_rel_reason,
            }
        )

    # 5) Save to Excel
    df = pd.DataFrame(rows)
    df.to_excel(out_excel, index=False)
    print(f"\nSaved detailed DeepEval results to {out_excel}")

    # 6) Print averages (ignoring failed / None scores)
    if not df.empty:
        def safe_mean(col: str) -> float:
            vals = [v for v in df[col] if isinstance(v, (int, float, float))]
            return sum(vals) / len(vals) if vals else 0.0

        avg_ans_rel = safe_mean("answer_relevancy_score")
        avg_faith = safe_mean("faithfulness_score")
        avg_ctx_rel = safe_mean("contextual_relevancy_score")

        print("\n=== Average DeepEval Scores (successful cases only) ===")
        print(f"Answer Relevancy       : {avg_ans_rel:.3f}")
        print(f"Faithfulness           : {avg_faith:.3f}")
        print(f"Contextual Relevancy   : {avg_ctx_rel:.3f}")
    else:
        print("No rows in dataframe; nothing to average.")


if __name__ == "__main__":
    main()
