"""
HAUI Student Handbook Chatbot – CLI entry point

Usage:
    python main.py
"""

from agent import HAUIAgent


BANNER = """
╔══════════════════════════════════════════════════════════╗
║      HAUI Student Handbook Chatbot (RAG + LangGraph)     ║
║  Type your question in Vietnamese or English.            ║
║  Commands:  /reset  – clear history  |  /quit – exit     ║
╚══════════════════════════════════════════════════════════╝
"""

SHOW_SOURCES = True  # Set to False to hide retrieval source info


def main():
    print(BANNER)
    agent = HAUIAgent()

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break

        if query.lower() == "/reset":
            agent.reset()
            print("── Conversation history cleared. ──\n")
            continue

        result = agent.chat(query)

        print(f"\nBot [{result.intent}]: {result.answer}\n")

        if SHOW_SOURCES and result.sources:
            print("── Sources ──")
            for src in result.sources:
                label = " > ".join(filter(None, [src.section, src.subsection]))
                print(f"  [{src.chunk_id}] {label}  (score: {src.score})")
            print()


if __name__ == "__main__":
    main()
