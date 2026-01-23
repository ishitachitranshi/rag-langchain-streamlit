import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings


# -------------------- BASIC SETUP --------------------
BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "data" / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RAG Engine", layout="wide")
st.title("📄 Retrieval Augmented Generation (RAG) Engine")

# -------------------- SIDEBAR (PRODUCTION STYLE) --------------------
with st.sidebar:
    st.header("🔐 API Configuration")

    # Recruiter-ready pattern: user provides key
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        st.success("✅ OpenAI API key loaded from secrets")
    else:
        key = st.text_input("OpenAI API Key", type="password")
        if key:
            os.environ["OPENAI_API_KEY"] = key
            st.success("✅ API key set for this session")
        else:
            st.info("ℹ️ Paste your OpenAI API key to enable answers")

    st.markdown("---")
    show_sources = st.toggle("Show retrieved chunks", value=False)
    top_k = st.slider("Top-K retrieved chunks", 2, 10, 5)

    if st.button("🧹 Reset session"):
        st.session_state.clear()
        st.success("Session cleared")

# -------------------- HELPERS --------------------
def load_pdf(path: str):
    return PyPDFLoader(path).load()

def split_documents(docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

@st.cache_resource(show_spinner=False)
def get_embeddings():
    # FREE, local embeddings (cost-efficient)
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_llm():
    # Production LLM (user-provided key)
    return ChatOpenAI(temperature=0)

def build_prompt(question: str, chunks: List[str], history: List[Tuple[str, str]]) -> str:
    history_text = ""
    for q, a in history[-6:]:
        history_text += f"User: {q}\nAssistant: {a}\n"

    context = "\n\n---\n\n".join(chunks)

    return f"""
You are a helpful assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say:
"I don't know based on the uploaded documents."

CHAT HISTORY:
{history_text}

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly and concisely.
"""

def answer_question(retriever, question, history, k):
    docs = retriever.get_relevant_documents(question)
    chunks = [d.page_content for d in docs][:k]

    if not chunks:
        return "I don't know based on the uploaded documents.", []

    prompt = build_prompt(question, chunks, history)

    try:
        llm = get_llm()
        resp = llm.invoke(prompt)
        answer = getattr(resp, "content", str(resp))
        return answer, chunks

    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "429" in msg:
            return (
                "❌ OpenAI quota not available for this API key.\n\n"
                "✅ Use an OpenAI key with billing enabled.\n"
                "This app is designed so users bring their own key.",
                chunks,
            )
        return f"❌ LLM error: {e}", chunks

# -------------------- STATE --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------- FILE UPLOAD --------------------
uploaded_files = st.file_uploader(
    "📤 Upload PDF documents",
    type="pdf",
    accept_multiple_files=True
)

if st.button("📥 Process Documents"):
    if "OPENAI_API_KEY" not in os.environ:
        st.warning("Please provide an OpenAI API key.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        all_docs = []

        with st.spinner("Processing documents..."):
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=TMP_DIR) as tmp:
                    tmp.write(file.read())
                    path = tmp.name

                all_docs.extend(load_pdf(path))
                os.remove(path)

            texts = split_documents(all_docs)
            embeddings = get_embeddings()

            # In-memory Chroma → NO deployment DB issues
            vectordb = Chroma.from_documents(texts, embedding=embeddings)
            st.session_state.retriever = vectordb.as_retriever(
                search_kwargs={"k": top_k}
            )

        st.success("✅ Documents processed successfully!")

# -------------------- CHAT --------------------
if "retriever" in st.session_state:
    query = st.chat_input("Ask a question about your documents")
    if query:
        st.chat_message("human").write(query)

        answer, chunks = answer_question(
            st.session_state.retriever,
            query,
            st.session_state.chat_history,
            top_k
        )

        st.chat_message("ai").write(answer)

        if show_sources and chunks:
            with st.expander("Retrieved chunks"):
                for i, c in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(c)

        st.session_state.chat_history.append((query, answer))
else:
    st.info("⬆️ Upload and process documents to start chatting.")
