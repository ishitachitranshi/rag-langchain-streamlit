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


# -------------------- PATHS --------------------
BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "data" / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(page_title="RAG Engine", layout="wide")
st.title("📄 Retrieval Augmented Generation (RAG) Engine")

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("🔐 API Configuration")

    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        st.success("✅ OpenAI API key loaded from Streamlit Secrets")
    else:
        key = st.text_input("OpenAI API Key", type="password")
        if key:
            os.environ["OPENAI_API_KEY"] = key
            st.success("✅ OpenAI API key set for this session")
        else:
            st.warning("⚠️ Add OPENAI_API_KEY in Streamlit Secrets or paste it here")

    st.markdown("---")
    show_sources = st.toggle("Show retrieved chunks", value=False)
    top_k = st.slider("Top-K chunks", min_value=2, max_value=10, value=5, step=1)

    st.markdown("---")
    if st.button("🧹 Reset Session Index"):
        st.session_state.pop("retriever", None)
        st.session_state.pop("chat_history", None)
        st.success("✅ Cleared in-memory index. Re-upload & process documents.")

# -------------------- HELPERS --------------------
def load_pdf(file_path: str):
    return PyPDFLoader(file_path).load()

def split_documents(documents):
    return CharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(documents)

@st.cache_resource(show_spinner=False)
def get_local_embeddings():
    # Free local embeddings
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOpenAI(temperature=0)

def build_prompt(question: str, context_chunks: List[str], chat_history: List[Tuple[str, str]]) -> str:
    history_text = ""
    for q, a in chat_history[-6:]:
        history_text += f"User: {q}\nAssistant: {a}\n"

    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else ""

    return f"""You are a helpful assistant answering ONLY from the provided context.
If the answer is not in the context, say: "I don't know based on the uploaded documents."

CHAT HISTORY:
{history_text}

CONTEXT:
{context_text}

QUESTION:
{question}

Answer clearly and concisely.
"""

def answer_question(retriever, question: str, chat_history: List[Tuple[str, str]], k: int):
    docs = retriever.get_relevant_documents(question)
    chunks = [d.page_content for d in docs][:k]

    if not chunks:
        return "I don't know based on the uploaded documents.", []

    llm = get_llm()
    prompt = build_prompt(question, chunks, chat_history)

    resp = llm.invoke(prompt)
    answer = getattr(resp, "content", str(resp))
    return answer, chunks

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
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
        st.warning("OpenAI API key missing. Add it in Streamlit Secrets or enter it in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        all_docs = []
        with st.spinner("Processing documents..."):
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=TMP_DIR.as_posix()) as tmp:
                    tmp.write(file.read())
                    pdf_path = tmp.name

                all_docs.extend(load_pdf(pdf_path))
                os.remove(pdf_path)

            texts = split_documents(all_docs)

            embeddings = get_local_embeddings()

            # ✅ IN-MEMORY Chroma (NO persist_directory) -> avoids tenant/db errors
            vectordb = Chroma.from_documents(texts, embedding=embeddings)
            st.session_state.retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

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
            k=top_k
        )

        st.chat_message("ai").write(answer)

        if show_sources and chunks:
            with st.expander("Retrieved chunks"):
                for i, c in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(c)

        st.session_state.chat_history.append((query, answer))
else:
    st.info("⬆️ Upload and process documents to start chatting.")
