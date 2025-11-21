#!/usr/bin/env python
"""
End-to-end workflow runner for the UL RAG Assistant.

This script orchestrates:

  1. Web crawling / ingestion
  2. Index building (web + md + pdf)
  3. Inference smoke test
  4. RAGAS evaluation
  5. DeepEval evaluation

Usage examples:

  # Run everything
  python scripts/run_workflow.py

  # Run only crawl + index
  python scripts/run_workflow.py --skip-infer --skip-ragas --skip-deepeval

  # Run only evaluations on existing index
  python scripts/run_workflow.py --skip-crawl --skip-index

Adjust the paths in WORKFLOW_CONFIG if your directory structure differs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# ----- Project root & paths -----

ROOT = Path(__file__).resolve().parents[1]  # ul_rag_assistant/
SCRIPTS_DIR = ROOT / "scripts"

WORKFLOW_CONFIG = {
    # 1) Crawl / ingest
    "seeds_path": ROOT / "data" / "ul" / "ul_seeds.jsonl",
    "out_jsonl": ROOT / "data" / "ul" / "ul_docs.jsonl",

    # 2) Build index
    "index_path": ROOT / "storage" / "index" / "ul_index.pkl",
    "md_dir": ROOT / "data" / "ul" / "md",
    "pdf_dir": ROOT / "data" / "ul" / "pdf",

    # 3) Inference smoke test
    "infer_script": SCRIPTS_DIR / "inferencer.py",
    "infer_test_question": "Who is J.J. Collins?",

    # 4) RAGAS
    "ragas_module": "ul_rag_assistant.ul_rag.evaluation.eval_ragas",

    # 5) DeepEval
    "deepeval_module": "ul_rag_assistant.ul_rag.evaluation.eval_deepeval",
}


# ----- Helper to run subprocess commands -----

def run_cmd(cmd, cwd: Path | None = None, step_name: str | None = None) -> None:
    """
    Run a subprocess command with basic logging.
    """
    if cwd is None:
        cwd = ROOT

    label = step_name or " ".join(str(c) for c in cmd)
    print(f"\n=== [{label}] ===")
    print(f"cwd: {cwd}")
    print(f"cmd: {' '.join(str(c) for c in cmd)}")
    print("--------------------------------------------------")

    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{label}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"[OK] Step '{label}' completed successfully.")


# ----- Individual steps -----

def step_crawl() -> None:
    cfg = WORKFLOW_CONFIG
    seeds = cfg["seeds_path"]
    out_jsonl = cfg["out_jsonl"]

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "ingest_ul_web.py"),
        "--seeds", str(seeds),
        "--out_jsonl", str(out_jsonl),
        "--max_depth", "8",           # adjust if needed
        "--max_pages", "20000",       # safety valve
        "--delay", "0.7",             # politeness
    ]
    run_cmd(cmd, step_name="crawl/ingest_ul_web")


def step_build_index(delete_existing: bool = True) -> None:
    cfg = WORKFLOW_CONFIG
    index_path: Path = cfg["index_path"]
    input_jsonl: Path = cfg["out_jsonl"]
    md_dir: Path = cfg["md_dir"]
    pdf_dir: Path = cfg["pdf_dir"]

    # Optional: delete existing index to avoid unpickling issues from old formats
    if delete_existing and index_path.exists():
        print(f"\n[INFO] Removing existing index at {index_path}")
        index_path.unlink()

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "build_index.py"),
        "--input", str(input_jsonl),
        "--index_path", str(index_path),
        "--md_dir", str(md_dir),
        "--pdf_dir", str(pdf_dir),
    ]
    run_cmd(cmd, step_name="build_index")


def step_infer_smoke() -> None:
    """
    Simple smoke test: run the inferencer script with a fixed test question.
    Adjust this if your inferencer.py has different CLI args.
    """
    cfg = WORKFLOW_CONFIG
    infer_script: Path = cfg["infer_script"]
    question: str = cfg["infer_test_question"]

    if not infer_script.exists():
        print(f"[WARN] inferencer script not found at {infer_script}, skipping smoke test.")
        return

    # Assumes inferencer.py supports a --question flag
    cmd = [
        sys.executable,
        str(infer_script),
        "--question", question,
    ]
    run_cmd(cmd, step_name="inferencer_smoke_test")


def step_ragas_eval() -> None:
    """
    Run the RAGAS evaluation module.
    Make sure OPENAI_API_KEY is set before running this step.
    """
    cfg = WORKFLOW_CONFIG
    module = cfg["ragas_module"]

    cmd = [
        sys.executable,
        "-m",
        module,
    ]
    run_cmd(cmd, step_name="eval_ragas")


def step_deepeval_eval() -> None:
    """
    Run the DeepEval evaluation module.
    Make sure OPENAI_API_KEY is set before running this step.
    """
    cfg = WORKFLOW_CONFIG
    module = cfg["deepeval_module"]

    cmd = [
        sys.executable,
        "-m",
        module,
    ]
    run_cmd(cmd, step_name="eval_deepeval")


# ----- Main CLI -----

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UL RAG Assistant end-to-end workflow runner.")

    parser.add_argument(
        "--skip-crawl",
        default=True,
        help="Skip web crawling / ingestion step.",
    )
    parser.add_argument(
        "--skip-index",
        default=True,
        help="Skip index building step.",
    )
    parser.add_argument(
        "--skip-infer",
        default=False,
        help="Skip inference smoke test.",
    )
    parser.add_argument(
        "--skip-ragas",
        default=True,
        help="Skip RAGAS evaluation.",
    )
    parser.add_argument(
        "--skip-deepeval",
        default=False,
        help="Skip DeepEval evaluation.",
    )

    parser.add_argument(
        "--keep-index",
        default=True,
        help="Do not delete existing index file before building (default is to delete).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("==================================================")
    print(" UL RAG Assistant Workflow Runner")
    print(" Project root:", ROOT)
    print("==================================================")

    if not args.skip_crawl:
        step_crawl()
    else:
        print("[SKIP] crawl/ingest_ul_web")

    if not args.skip_index:
        step_build_index(delete_existing=not args.keep_index)
    else:
        print("[SKIP] build_index")

    if not args.skip_infer:
        step_infer_smoke()
    else:
        print("[SKIP] inferencer_smoke_test")

    if not args.skip_ragas:
        step_ragas_eval()
    else:
        print("[SKIP] eval_ragas")

    if not args.skip_deepeval:
        step_deepeval_eval()
    else:
        print("[SKIP] eval_deepeval")

    print("\n[ALL DONE] Workflow completed.")


if __name__ == "__main__":
    main()
