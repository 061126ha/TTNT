"""
HAUI SICT Chatbot — Streamlit UI

Run with:
    streamlit run app.py
"""

import os

import html as _html

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HAUI SICT Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 0; }

    .badge {
        display: inline-block;
        padding: 2px 10px; border-radius: 10px;
        font-size: 0.72rem; font-weight: 600; margin-left: 6px;
    }
    .badge-curriculum  { background: #d4edda; color: #155724; }
    .badge-regulations { background: #cce5ff; color: #004085; }
    .badge-general     { background: #fff3cd; color: #856404; }

    .graph-info {
        background: #f1f3f5; border: 1px solid #dee2e6;
        border-radius: 8px; padding: 8px 12px; margin-top: 6px;
        font-size: 0.8rem; color: #495057;
    }

    .rerank-pill {
        display: inline-block; padding: 1px 8px; border-radius: 10px;
        font-size: 0.72rem; font-weight: 600; margin-left: 4px;
        background: #e8d5f5; color: #5b2d8e;
    }

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
    st.title("HAUI SICT Chatbot")
    st.caption("Multi-agent RAG · Trường CNTT&TT · ĐH Công nghiệp Hà Nội")
    st.divider()

    st.subheader("Cài đặt")

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

    prev_model = st.session_state.get("_prev_chat_model")
    if prev_model and prev_model != chat_model:
        import src.llm.client as _llm_client
        _llm_client._chat_model = None
        st.session_state.agent = None

    os.environ["OPENROUTER_CHAT_MODEL"] = chat_model
    st.session_state["_prev_chat_model"] = chat_model

    top_k = st.slider("Retrieval top-K", min_value=1, max_value=10, value=5)
    os.environ["RETRIEVAL_TOP_K"] = str(top_k)

    reranker_type = st.selectbox(
        "Reranker",
        ["cohere", "cross_encoder", "llm", "none"],
        format_func={
            "cohere": "Cohere Rerank",
            "cross_encoder": "Cross-encoder local",
            "llm": "LLM rerank",
            "none": "Không rerank",
        }.get,
        index=["cohere", "cross_encoder", "llm", "none"].index(
            os.environ.get("RERANKER_TYPE", "cohere")
        ) if os.environ.get("RERANKER_TYPE", "cohere") in ["cohere", "cross_encoder", "llm", "none"] else 0,
    )
    os.environ["RERANKER_TYPE"] = reranker_type

    if reranker_type != "none":
        def _safe_int(val: str, default: int) -> int:
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        rerank_top_k = st.slider(
            "Rerank top-K",
            min_value=1,
            max_value=top_k,
            value=min(_safe_int(os.environ.get("RERANK_TOP_K", ""), top_k), top_k),
        )
        fetch_multiplier = st.slider(
            "Rerank fetch multiplier",
            min_value=1,
            max_value=5,
            value=_safe_int(os.environ.get("RERANK_FETCH_MULTIPLIER", ""), 3),
        )
        os.environ["RERANK_TOP_K"] = str(rerank_top_k)
        os.environ["RERANK_FETCH_MULTIPLIER"] = str(fetch_multiplier)
    else:
        os.environ.pop("RERANK_TOP_K", None)
        os.environ.pop("RERANK_FETCH_MULTIPLIER", None)

    if reranker_type == "cohere":
        cohere_model = st.text_input(
            "Cohere rerank model",
            value=os.environ.get("COHERE_RERANK_MODEL", "rerank-multilingual-v3.0"),
        )
        
        os.environ["COHERE_RERANK_MODEL"] = cohere_model
        

    rerank_config = (
        reranker_type,
        os.environ.get("RERANK_TOP_K"),
        os.environ.get("RERANK_FETCH_MULTIPLIER"),
        os.environ.get("COHERE_RERANK_MODEL"),
    )
    prev_rerank_config = st.session_state.get("_prev_rerank_config")
    if prev_rerank_config is not None and prev_rerank_config != rerank_config:
        import src.service.reranker as _reranker_module
        try:
            _reranker_module.reranker = _reranker_module.get_reranker()
        except ValueError as _exc:
            st.sidebar.error(f"Reranker init failed: {_exc}")
        st.session_state.agent = None
    st.session_state["_prev_rerank_config"] = rerank_config

    show_sources = st.toggle("Hiển thị ngữ cảnh truy xuất", value=True)
    show_rerank = st.toggle("Hiển thị rerank info", value=True)

    st.divider()

    if st.button("🗑️ Xóa cuộc trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent = None
        st.rerun()

    st.divider()
    st.markdown(
        "**Agents:**\n"
        "- 🎓 Chương trình đào tạo\n"
        "- 🏛️ Thông tin & quy chế trường\n"
        "- 💬 Câu hỏi chung\n\n"
        "Built with **LangGraph** · FAISS · OpenRouter"
    )

# ─── Agent initialisation ─────────────────────────────────────────────────────

from src.agent.haui_agent import HAUIAgent  # noqa: E402
import src.agent.tools as _tools_module  # noqa: E402

INDEX_FILE = "faiss_curriculum.bin"

_BADGE = {
    "curriculum":  ("badge-curriculum",  "🎓 Chương trình đào tạo"),
    "regulations": ("badge-regulations", "🏛️ Thông tin trường"),
    "general":     ("badge-general",     "💬 Chung"),
}


def _badge_html(intent: str) -> str:
    cls, label = _BADGE.get(intent, ("badge-general", "💬 Chung"))
    return f'<span class="badge {cls}">{label}</span>'


def get_agent() -> HAUIAgent | None:
    if "agent" not in st.session_state or st.session_state.agent is None:
        if not os.path.exists(INDEX_FILE):
            return None
        st.session_state.agent = HAUIAgent()
    return st.session_state.agent


_TOOL_LABEL = {
    "curriculum": "🎓 Chương trình đào tạo",
    "regulations": "🏛️ Thông tin trường",
}


def _render_rerank_info(rerank_info: list[dict]) -> None:
    if not rerank_info:
        return
    for entry in rerank_info:
        tool_label = _TOOL_LABEL.get(entry["tool"], entry["tool"])
        reranker_type = entry["reranker_type"]
        model_name = entry.get("reranker_model")
        pill_label = f"{reranker_type}: {_html.escape(model_name)}" if model_name else reranker_type
        pill = f'<span class="rerank-pill">{pill_label}</span>'
        st.markdown(f"**{tool_label}** {pill}", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Fetched từ FAISS", entry["fetch_k"])
        col2.metric("Thực tế truy xuất", entry["retrieved"])
        col3.metric("Sau rerank", entry["reranked"])

        chunks = entry.get("chunks", [])
        if chunks:
            rows = []
            for i, chunk in enumerate(chunks, 1):
                label = " > ".join(filter(None, [chunk.section, chunk.subsection])) or str(chunk.chunk_id)
                rerank_score = (
                    f"{chunk.rerank_score:.4f}" if chunk.rerank_score is not None else "—"
                )
                rows.append({
                    "Rank": i,
                    "Nguồn": label,
                    "FAISS score": round(chunk.score, 4),
                    "Rerank score": rerank_score,
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)


def extract_graph_trace(agent: HAUIAgent) -> list[str]:
    """Reconstruct which multi-agent nodes ran from the message history."""
    trace: list[str] = []
    messages = agent._messages
    tool_called_this_turn = False
    for msg in messages:
        if not hasattr(msg, "type"):
            continue
        if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_name = msg.tool_calls[0].get("name", "tool") if msg.tool_calls else "tool"
            trace.append(f"generate → 🔧 {tool_name}")
        elif msg.type == "tool":
            tool_called_this_turn = True
            trace.append("retrieve ✅")
        elif msg.type == "ai" and msg.content:
            if tool_called_this_turn:
                trace.append("answer →  ✅")
            else:
                trace.append("respond directly 💬")
            tool_called_this_turn = False
    return trace


def extract_tool_context(agent: HAUIAgent) -> str | None:
    for msg in reversed(agent._messages):
        if hasattr(msg, "type") and msg.type == "tool" and msg.content:
            return msg.content
    return None


# ─── Main UI ──────────────────────────────────────────────────────────────────

st.header("🎓 HAUI SICT — Trợ lý AI")
st.caption(
    "Hỏi về chương trình đào tạo, ngành học, chuẩn đầu ra, quy chế, "
    "thông tin trường và các vấn đề liên quan đến Trường CNTT&TT - ĐH Công nghiệp Hà Nội."
)

if not os.path.exists(INDEX_FILE):
    st.warning(
        "**Vector index chưa được build.** Chạy lần lượt các lệnh sau rồi reload trang:",
        icon="⚠️",
    )
    st.code(
        "python scripts/crawl_website.py\n"
        "python scripts/split_chunks.py\n"
        "python scripts/build_index.py --input data/curriculum_chunks.json --index faiss_curriculum.bin --meta faiss_curriculum_meta.pkl\n"
        "python scripts/build_index.py --input data/regulation_chunks.json --index faiss_regulation.bin --meta faiss_regulation_meta.pkl",
        language="bash",
    )
    st.stop()

if not os.environ.get("OPENROUTER_API_KEY"):
    st.error(
        "Chưa có OPENROUTER_API_KEY. Thêm vào file .env rồi khởi động lại.",
        icon="🔑",
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

agent = get_agent()

# ─── Render existing messages ─────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎓"):
        if msg["role"] == "assistant":
            st.markdown(
                msg["content"] + _badge_html(msg.get("intent", "general")),
                unsafe_allow_html=True,
            )
            if msg.get("trace"):
                trace_str = " → ".join(msg["trace"])
                st.markdown(
                    f'<div class="graph-info">🔄 <b>Agent trace:</b> {trace_str}</div>',
                    unsafe_allow_html=True,
                )
            if show_sources and msg.get("context"):
                with st.expander("📄 Ngữ cảnh truy xuất", expanded=False):
                    st.code(msg["context"][:3000], language=None)
            if show_rerank and msg.get("rerank_info"):
                with st.expander("📊 Rerank info", expanded=False):
                    _render_rerank_info(msg["rerank_info"])
        else:
            st.markdown(msg["content"])

# ─── Chat input ───────────────────────────────────────────────────────────────

if prompt := st.chat_input("Hỏi về chương trình đào tạo, quy chế, thông tin trường…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Đang xử lý…"):
            _tools_module.rerank_debug.clear()
            result = agent.chat(prompt)
            rerank_info = list(_tools_module.rerank_debug)

        intent = result.intent
        answer = result.answer

        st.markdown(answer + _badge_html(intent), unsafe_allow_html=True)

        tool_context = extract_tool_context(agent)
        graph_trace = extract_graph_trace(agent)

        if graph_trace:
            trace_str = " → ".join(graph_trace)
            st.markdown(
                f'<div class="graph-info">🔄 <b>Agent trace:</b> {trace_str}</div>',
                unsafe_allow_html=True,
            )

        if show_sources and tool_context:
            with st.expander("📄 Ngữ cảnh truy xuất", expanded=True):
                st.code(tool_context[:3000], language=None)

        if show_rerank and rerank_info:
            with st.expander("📊 Rerank info", expanded=True):
                _render_rerank_info(rerank_info)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "intent": intent,
            "context": tool_context,
            "trace": graph_trace,
            "rerank_info": rerank_info,
        }
    )
