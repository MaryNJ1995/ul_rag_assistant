#!/usr/bin/env python
import json
from typing import List, Dict, Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness, ContextRecall, ContextPrecision
from tqdm import tqdm
from ul_rag_assistant.ul_rag.graph.graph import run_ul_rag_debug


def load_eval_data(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj["question"].strip()
            gt = obj["ground_truth"].strip()
            data.append({"question": q, "ground_truth": gt})
    return data


def build_ragas_dataset(eval_data: List[Dict[str, Any]]) -> Dataset:
    """Run My UL RAG pipeline on each question and prepare HF Dataset for RAGAS."""
    records = []

    for row in tqdm(eval_data):
        q = row["question"]
        gt = row["ground_truth"]

        # Call My debug pipeline to get answer + contexts
        resp = run_ul_rag_debug(q, mode="student", locale="IE")

        answer = resp["answer"]
        contexts = resp.get("contexts", [])

        records.append(
            {
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "ground_truths": [gt],  # RAGAS expects a list
            }
        )

    return Dataset.from_list(records)


def main():
    eval_path = "/home/maryam_najafi/ul_bot/ul_rag_assistant/data/eval/ul_eval.jsonl"
    eval_data = load_eval_data(eval_path)
    ragas_ds = build_ragas_dataset(eval_data)

    metrics = [
        ContextPrecision(),
        ContextRecall(),
        Faithfulness(),
        AnswerRelevancy(),
    ]

    result = evaluate(ragas_ds, metrics=metrics)
    df = result.to_pandas()
    print(df)
    df.to_excel("/home/maryam_najafi/ul_bot/ul_rag_assistant/data/eval/ragas_result.xlsx")

    print("\n--- Averages ---")
    print(df.mean(numeric_only=True))


if __name__ == "__main__":
    main()
