import argparse

from ul_rag_assistant.ul_rag.ingest.web import fetch_ul_pages


def main():
    parser = argparse.ArgumentParser(description="Fetch UL pages into a JSONL corpus.")
    parser.add_argument("--seeds", type=str, default="/home/maryam_najafi/ul_bot/ul_rag_assistant/data/ul/ul_seeds.jsonl", help="Path to UL seeds JSONL (one {url} per line).")
    parser.add_argument("--out_jsonl", type=str, default="/home/maryam_najafi/ul_bot/ul_rag_assistant/data/ul/ul_data.jsonl", help="Output JSONL corpus file.")
    args = parser.parse_args()

    fetch_ul_pages(args.seeds, args.out_jsonl)


if __name__ == "__main__":
    main()
