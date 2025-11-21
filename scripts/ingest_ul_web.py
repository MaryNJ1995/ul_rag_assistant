import argparse

from ul_rag_assistant.ul_rag.ingest.web import fetch_ul_pages, crawl_ul
#!/usr/bin/env python

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Fetch UL pages into a JSONL corpus.")
    parser.add_argument("--seeds", type=str, default="/home/maryam_najafi/ul_bot/ul_rag_assistant/data/ul/ul_seeds.jsonl", help="Path to UL seeds JSONL (one {url} per line).")
    parser.add_argument("--out_jsonl", type=str, default="/home/maryam_najafi/ul_bot/ul_rag_assistant/data/ul/ul_data.jsonl", help="Output JSONL corpus file.")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--max_pages", type=int, default=2000)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    crawl_ul(
        seeds_path=args.seeds,
        out_path=args.out_jsonl,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        delay=args.delay,
    )

if __name__ == "__main__":
    main()
