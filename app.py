"""
HAUI Student Handbook Chatbot — Streamlit UI
Run with:
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HAUI Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Main chat area */
    .main .block-container { padding-top: 1.5rem; padding-bottom: 0; }

    /* Intent badges */
    .badge-related {
        background: #d4edda; color: #155724;
        padding: 2px 8px; border-radius: 10px;
        font-size: 0.72rem; font-weight: 600; margin-left: 6px;
    }
    .badge-unrelated {
        background: #fff3cd; color: #856404;
        padding: 2px 8px; border-radius: 10px;
        font-size: 0.72rem; font-weight: 600; margin-left: 6px;
    }

    /* Source cards */
    .source-card {
        background: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .source-card .score { color: #6c757d; font-size: 0.78rem; float: right; }
    .source-card .section-label { font-weight: 600; color: #0d6efd; }

    /* Sidebar */
    section[data-testid="stSidebar"] { min-width: 280px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/"
        "HAUI_logo.png/200px-HAUI_logo.png",
        width=120,
    )
    st.title("HAUI Chatbot")
    st.caption("Student Handbook Assistant")
    st.divider()

    # API key input (fallback if not set in .env)
    api_key_input = st.text_input(
        "OpenRouter API Key",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
        help="Set OPENROUTER_API_KEY in your .env file or enter it here. Get one at openrouter.ai/keys",
    )
    if api_key_input:
        os.environ["OPENROUTER_API_KEY"] = api_key_input

    st.divider()
    st.subheader("Settings")

    chat_model = st.selectbox(
        "Chat model",
        [
            "google/gemini-2.0-flash-001",
            "google/gemini-2.5-pro-preview-03-25",
            "anthropic/claude-3.5-sonnet",
            "deepseek/deepseek-chat-v3-0324",
            "meta-llama/llama-3.3-70b-instruct",
        ],
        index=0,
    )
    os.environ["OPENROUTER_CHAT_MODEL"] = chat_model

    top_k = st.slider("Retrieval top-K", min_value=1, max_value=10, value=5)
    os.environ["RETRIEVAL_TOP_K"] = str(top_k)

    show_sources = st.toggle("Show retrieved sources", value=True)

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent = None
        st.rerun()

    st.divider()
    st.caption(
        "Built with LangGraph · FAISS · OpenRouter\n\n"
        "Ask anything about the HAUI student handbook in Vietnamese or English."
    )

# ─── Agent initialisation ─────────────────────────────────────────────────────

# Import here so env vars set above are picked up before the module loads defaults
from agent import HAUIAgent  # noqa: E402

INDEX_FILE = "faiss_index.bin"


def get_agent() -> HAUIAgent | None:
    """Return a cached HAUIAgent, or None if the index hasn't been built yet."""
    if "agent" not in st.session_state or st.session_state.agent is None:
        if not os.path.exists(INDEX_FILE):
            return None
        st.session_state.agent = HAUIAgent()
    return st.session_state.agent


# ─── Main UI ──────────────────────────────────────────────────────────────────

st.header("🎓 HAUI Student Handbook Chatbot")
st.caption(
    "Ask me anything about academic programmes, learning outcomes, curriculum, "
    "and more from the HAUI student handbook."
)

# Index not built yet — show setup instructions
if not os.path.exists(INDEX_FILE):
    st.warning(
        "**Vector index not found.** "
        "Run the following command to build it, then refresh this page:",
        icon="⚠️",
    )
    st.code("python build_vectorstore.py", language="bash")
    st.stop()

# No API key
if not os.environ.get("OPENROUTER_API_KEY"):
    st.error("Please enter your OpenRouter API key in the sidebar.", icon="🔑")
    st.stop()

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "intent", "sources"}

agent = get_agent()

# ─── Render existing messages ─────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎓"):
        if msg["role"] == "assistant":
            intent = msg.get("intent", "")
            badge_class = "badge-related" if intent == "related" else "badge-unrelated"
            badge_label = "handbook" if intent == "related" else "off-topic"
            st.markdown(
                f'{msg["content"]}'
                f' <span class="{badge_class}">{badge_label}</span>',
                unsafe_allow_html=True,
            )
            if show_sources and msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)", expanded=False):
                    for src in msg["sources"]:
                        label = " › ".join(
                            filter(None, [src.get("section", ""), src.get("subsection", "")])
                        )
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="score">score: {src["score"]}</span>'
                            f'<span class="section-label">[{src["chunk_id"]}] {label}</span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        else:
            st.markdown(msg["content"])

# ─── Chat input ───────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about the student handbook…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Run the agent
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking…"):
            result = agent.chat(prompt)

        intent = result["intent"]
        answer = result["answer"]
        sources = result["sources"]

        badge_class = "badge-related" if intent == "related" else "badge-unrelated"
        badge_label = "handbook" if intent == "related" else "off-topic"
        st.markdown(
            f"{answer} "
            f'<span class="{badge_class}">{badge_label}</span>',
            unsafe_allow_html=True,
        )

        if show_sources and sources:
            with st.expander(f"📚 Sources ({len(sources)} chunks)", expanded=True):
                for src in sources:
                    label = " › ".join(
                        filter(None, [src.get("section", ""), src.get("subsection", "")])
                    )
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="score">score: {src["score"]}</span>'
                        f'<span class="section-label">[{src["chunk_id"]}] {label}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # Persist to session
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "intent": intent,
            "sources": sources,
        }
    )
