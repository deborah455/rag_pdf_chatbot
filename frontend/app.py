import requests
import streamlit as st
from pypdf import PdfReader

BACKEND_QUERY_URL = "http://127.0.0.1:8000/rag/query"
BACKEND_INGEST_URL = "http://127.0.0.1:8000/rag/ingest"

st.set_page_config(
    page_title="ContextFlow – Local RAG (Ollama)",
    page_icon="🧠",
    layout="centered",
)

# ---------- CSS: clean, but with nicer color & badges ----------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.8rem;
        padding-bottom: 2rem;
        max-width: 820px;
    }
    body {
        margin: 0;
        background: radial-gradient(circle at top, #0f172a 0%, #020617 50%, #020617 100%);
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    }
    .title {
        font-size: 2.0rem;
        font-weight: 750;
        letter-spacing: -0.03em;
        text-align: center;
        margin-bottom: 0.25rem;
        background: linear-gradient(120deg, #38bdf8, #a855f7, #22c55e);
        -webkit-background-clip: text;
        color: transparent;
    }
    .subtitle {
        font-size: 0.96rem;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .card {
        border-radius: 20px;
        padding: 1.1rem 1.25rem 1rem;
        background-color: rgba(15,23,42,0.98);
        border: 1px solid rgba(55,65,81,0.9);
        box-shadow: 0 24px 60px rgba(15,23,42,0.95);
    }
    .stack-line {
        font-size: 0.8rem;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 0.9rem;
    }
    .badge-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-bottom: 0.9rem;
    }
    .badge {
        font-size: 0.74rem;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.6);
        background: rgba(15,23,42,0.95);
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #9ca3af;
    }
    .badge-dot {
        width: 7px;
        height: 7px;
        border-radius: 999px;
        background: radial-gradient(circle, #22c55e, #16a34a);
        box-shadow: 0 0 10px rgba(34,197,94,0.9);
    }
    .chat-container {
        max-height: 340px;
        overflow-y: auto;
        padding-right: 4px;
        margin-bottom: 0.7rem;
    }
    .bubble-user {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: #022c22;
        padding: 8px 11px;
        border-radius: 14px;
        border-bottom-right-radius: 4px;
        margin-bottom: 6px;
        font-size: 0.88rem;
        max-width: 80%;
        margin-left: auto;
    }
    .bubble-bot {
        background-color: #020617;
        border-radius: 14px;
        border-bottom-left-radius: 4px;
        border: 1px solid rgba(75,85,99,0.9);
        padding: 8px 11px;
        margin-bottom: 6px;
        font-size: 0.88rem;
        max-width: 100%;
        color: #e5e7eb;
    }
    .bubble-meta {
        font-size: 0.7rem;
        color: #9ca3af;
        margin-bottom: 2px;
    }
    .tiny-label {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 0.3rem;
    }
    .footer-note {
        font-size: 0.72rem;
        color: #6b7280;
        text-align: center;
        margin-top: 0.8rem;
    }
    .stTextInput > div > div > input {
        background-color: #020617;
        border-radius: 999px;
        border: 1px solid rgba(75,85,99,0.9);
        color: #e5e7eb;
        font-size: 0.9rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px rgba(56,189,248,0.55);
    }
    .send-button button {
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #38bdf8, #22c55e);
        color: #020617;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.45rem 0.95rem;
        box-shadow: 0 10px 24px rgba(37,99,235,0.55);
    }
    .send-button button:hover {
        filter: brightness(1.05);
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(37,99,235,0.75);
    }
    .send-button button:active {
        transform: translateY(0px) scale(0.99);
        box-shadow: 0 8px 20px rgba(37,99,235,0.6);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# ---------- Header ----------
st.markdown('<div class="title">ContextFlow – Local RAG (Ollama)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a document and chat with a local LLM. Answers are grounded in your content using a vector database.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="stack-line">Stack: FastAPI · ChromaDB · Streamlit · Ollama embeddings & chat LLM.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="card">', unsafe_allow_html=True)

# Badges row
st.markdown(
    """
    <div class="badge-row">
        <div class="badge"><span class="badge-dot"></span> Local LLM (Ollama)</div>
        <div class="badge">Document Q&A (RAG)</div>
        <div class="badge">PDF / TXT upload</div>
        <div class="badge">No external API keys</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Upload & ingest ----------
st.markdown("#### 📂 Document upload")
st.markdown(
    '<div class="tiny-label">Upload a PDF or TXT file. The first part of the text will be embedded via Ollama and stored in ChromaDB.</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"], label_visibility="collapsed")

if uploaded is not None:
    doc_id = uploaded.name
    text_content = ""

    if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
        try:
            reader = PdfReader(uploaded)
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    text_content += txt + "\n"
        except Exception as e:
            st.error(f"Could not read PDF: {e}")
    else:
        try:
            raw = uploaded.read()
            text_content = raw.decode("utf-8", errors="ignore")
        except Exception as e:
            st.error(f"Could not read text file: {e}")

    # Limit text length for the demo to avoid timeouts
    MAX_CHARS = 4000
    if len(text_content) > MAX_CHARS:
        text_content = text_content[:MAX_CHARS]
        st.info("Document is long; only the first part was ingested for this demo.")

    if text_content.strip():
        if st.button("Ingest document into RAG store"):
            try:
                resp = requests.post(
                    BACKEND_INGEST_URL,
                    json={"doc_id": doc_id, "text": text_content},
                    timeout=40,
                )
                if resp.status_code == 200:
                    st.success(f"Document '{doc_id}' ingested successfully.")
                else:
                    st.error(f"Ingest error: {resp.status_code}")
            except Exception as e:
                st.error(f"Could not reach backend ingest endpoint: {e}")
    else:
        st.info("No text extracted from this file. Try another document.")

st.markdown("---")

# ---------- Chat ----------
st.markdown("#### 💬 Chat over your document")
st.markdown(
    '<div class="tiny-label">Ask a question. The backend embeds your query with Ollama, retrieves similar chunks from ChromaDB, and the local LLM generates an answer.</div>',
    unsafe_allow_html=True,
)

chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            st.markdown(
                f'<div class="bubble-user">{content}</div>',
                unsafe_allow_html=True,
            )
        else:
            meta = msg.get("meta", "")
            if meta:
                st.markdown(
                    f'<div class="bubble-meta">{meta}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="bubble-bot">{content}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1.5])
with col_input:
    user_question = st.text_input(
        "Ask something...",
        value=st.session_state.user_input,
        key="user_input",
        label_visibility="collapsed",
        placeholder="e.g. Summarize the main idea of this document.",
    )

with col_btn:
    with st.container():
        st.markdown('<div class="send-button">', unsafe_allow_html=True)
        send = st.button("Send", key="send_btn")
        st.markdown("</div>", unsafe_allow_html=True)

if send and st.session_state.user_input.strip():
    q = st.session_state.user_input.strip()
    st.session_state.messages.append({"role": "user", "content": q})

    try:
        resp = requests.post(
            BACKEND_QUERY_URL,
            params={"question": q},
            timeout=40,
        )
        if resp.status_code == 200:
            data = resp.json()
            answer_text = data.get("answer", "")
            chunks = data.get("chunks_used", {})

            meta_text = ""
            if isinstance(chunks, dict) and "ids" in chunks:
                ids = chunks["ids"][0] if chunks["ids"] else []
                meta_text = f"Retrieved chunks: {', '.join(ids) if ids else 'n/a'}"

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text or "(No response text returned.)",
                    "meta": meta_text,
                }
            )
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "The backend returned an error. Please try again or check the API.",
                    "meta": f"Status: {resp.status_code}",
                }
            )
    except Exception as e:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "I couldn't reach the FastAPI server. Is it running on port 8000?",
                "meta": str(e),
            }
        )

    st.rerun()

st.markdown(
    '<div class="footer-note">ContextFlow – Local RAG demo · FastAPI · ChromaDB · Streamlit · Ollama (embeddings + chat LLM).</div>',
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
