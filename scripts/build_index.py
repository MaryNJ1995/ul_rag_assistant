import argparse

from ul_rag_assistant.ul_rag.ingest.build_index import build_index_from_jsonl


def main():
    parser = argparse.ArgumentParser(description="Build UL index (BM25 + embeddings) from JSONL corpus.")
    parser.add_argument("--input", type=str, default="/home/maryam_najafi/ul_bot/ul_rag_assistant/data/ul/ul_data.jsonl", help="Input JSONL corpus.")
    parser.add_argument("--index_path", type=str, default="/home/maryam_najafi/ul_bot/ul_rag_assistant/storage/index/ul_index.pkl", help="Where to store index pickle (overrides INDEX_PATH env).")
    args = parser.parse_args()

    build_index_from_jsonl(args.input, index_path=args.index_path)


if __name__ == "__main__":
    main()
