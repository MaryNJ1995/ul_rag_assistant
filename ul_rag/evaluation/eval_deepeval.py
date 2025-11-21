#!/usr/bin/env python
"""
DeepEval-based evaluation for the UL RAG assistant.

This script:
  - Loads UL eval questions from data/eval/ul_eval.jsonl
  - Runs your RAG pipeline (run_ul_rag_debug) for each question
  - Evaluates:
      - Answer Relevancy
      - Faithfulness
      - Contextual Relevancy
    using DeepEval metrics
  - Saves per-question results into an Excel file:
      data/eval/deepeval_results.xlsx
"""

import json
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


def load_eval_data(path: str) -> List[Dict[str, Any]]:
    """Load your UL eval dataset from JSONL."""
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


def main():
    eval_path = "/home/maryam_najafi/ul_bot/ul_rag_assistant/data/eval/ul_eval.jsonl"
    out_excel = "deepeval_results.xlsx"

    eval_data = load_eval_data(eval_path)
    print(f"Loaded {len(eval_data)} eval questions from {eval_path}")

    rows: List[Dict[str, Any]] = []

    # We compute metrics per test case (without relying on evaluate()'s return)
    for row in tqdm(eval_data, desc="Evaluating with DeepEval"):
        question = row["question"]
        ground_truth = row["ground_truth"]

        # 1) Run your RAG pipeline
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

        # 3) Define metrics for THIS test case
        answer_relevancy = AnswerRelevancyMetric()
        faithfulness = FaithfulnessMetric()
        contextual_relevancy = ContextualRelevancyMetric()

        # 4) Measure metrics (each will call the judge LLM once)
        answer_relevancy.measure(test_case)
        faithfulness.measure(test_case)
        contextual_relevancy.measure(test_case)

        # 5) Collect scores + reasons
        row_out: Dict[str, Any] = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            # Store contexts as a single string; you can keep as list if you prefer
            "contexts": " ||| ".join(contexts),
            "answer_relevancy_score": answer_relevancy.score,
            "answer_relevancy_reason": answer_relevancy.reason,
            "faithfulness_score": faithfulness.score,
            "faithfulness_reason": faithfulness.reason,
            "contextual_relevancy_score": contextual_relevancy.score,
            "contextual_relevancy_reason": contextual_relevancy.reason,
        }

        rows.append(row_out)

    # 6) Save to Excel
    df = pd.DataFrame(rows)
    df.to_excel(out_excel, index=False)
    print(f"\nSaved detailed DeepEval results to {out_excel}")

    # 7) Also print average scores for a quick view
    if not df.empty:
        avg_answer_rel = df["answer_relevancy_score"].mean()
        avg_faithfulness = df["faithfulness_score"].mean()
        avg_ctx_rel = df["contextual_relevancy_score"].mean()

        print("\n=== Average DeepEval Scores ===")
        print(f"Answer Relevancy       : {avg_answer_rel:.3f}")
        print(f"Faithfulness           : {avg_faithfulness:.3f}")
        print(f"Contextual Relevancy   : {avg_ctx_rel:.3f}")
    else:
        print("No rows in dataframe; nothing to average.")


if __name__ == "__main__":
    main()
