"""
HAUI SICT Chatbot — Streamlit UI
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG ENV (QUAN TRỌNG)
# =========================
os.environ.setdefault("RERANKER_TYPE", "cross_encoder")

# =========================
# PAGE
# =========================
st.set_page_config(
    page_title="HAUI SICT Chatbot",
    page_icon="🎓",
    layout="wide",
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("HAUI SICT Chatbot")

    chat_model = st.selectbox(
        "Chat model",
        [
            "nex-agi/nex-n2-pro:free",
            "anthropic/claude-3.5-sonnet",
        ],
        index=0
    )

    os.environ["OPENROUTER_CHAT_MODEL"] = chat_model

    top_k = st.slider("Top-K", 1, 10, 5)
    os.environ["RETRIEVAL_TOP_K"] = str(top_k)

    reranker_type = st.selectbox(
        "Reranker",
        ["cross_encoder", "llm", "cohere", "none"],
        index=0
    )

    os.environ["RERANKER_TYPE"] = reranker_type

    reload_btn = st.button("🔄 Reload reranker")

# =========================
# IMPORT RERANKER (SAFE)
# =========================
import src.service.reranker as reranker_module


def reload_reranker():
    """
    reload an toàn - KHÔNG gọi get_reranker từ app
    """
    try:
        reranker_module.reranker = reranker_module.get_reranker()
        st.success("Reranker reloaded!")
    except Exception as e:
        st.error(f"Reranker error: {e}")


if reload_btn:
    reload_reranker()

# =========================
# AGENT
# =========================
from src.agent.haui_agent import HAUIAgent

if "agent" not in st.session_state:
    st.session_state.agent = HAUIAgent()

agent = st.session_state.agent

# =========================
# CHAT HISTORY
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =========================
# CHAT INPUT
# =========================
if prompt := st.chat_input("Hỏi về HAUI..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = agent.chat(prompt)

        answer = result.answer
        intent = result.intent

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "intent": intent
    })