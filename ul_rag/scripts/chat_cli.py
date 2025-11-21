import argparse

from ul_rag_assistant.ul_rag.interfaces.chat_session import RAGChatSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["student", "staff"], default="student")
    parser.add_argument("--locale", default="IE")
    args = parser.parse_args()

    session = RAGChatSession(mode=args.mode, locale=args.locale)

    print(f"UL RAG Assistant (mode={args.mode}, locale={args.locale})")
    print("Type your question (or 'quit' to exit).\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not q or q.lower() in {"quit", "exit"}:
            print("Bye.")
            break

        bot_turn = session.ask(q)
        print(f"Bot: {bot_turn.content}\n")


if __name__ == "__main__":
    main()
