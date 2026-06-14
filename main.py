"""
HAUI Student Handbook Chatbot – CLI entry point

Usage:
    python main.py
"""

import os
import sys

# Ensure project root is on path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.agent.haui_agent import HAUIAgent


BANNER = """
╔══════════════════════════════════════════════════════════╗
║      HAUI Student Handbook Chatbot (RAG + LangGraph)     ║
║  Type your question in Vietnamese or English.            ║
║  Commands:  /reset  – clear history  |  /quit – exit     ║
╚══════════════════════════════════════════════════════════╝
"""

SHOW_SOURCES = True


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

        try:
            result = agent.chat(query)
        except Exception as e:
            print(f"\n[Error] {e}\n")
            continue

        print(f"\nBot [{result.intent}]: {result.answer}\n")

        if SHOW_SOURCES and getattr(result, "sources", None):
            print("── Sources ──")
            for src in result.sources:
                label = " > ".join(filter(None, [src.section, src.subsection]))
                print(f"  [{src.chunk_id}] {label} (score: {src.score})")
            print()


if __name__ == "__main__":
    main()