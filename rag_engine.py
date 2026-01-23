import os
import tempfile
from pathlib import Path

import streamlit as st

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings


# -------------------- PATHS --------------------
BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "data" / "tmp"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

TMP_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(page_title="RAG Engine", layout="wide")
st.title("📄 Retrieval Augmented Generation (RAG) Engine")

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("🔐 API Configuration")

    # Use Streamlit Secrets first, fallback to manual input
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
    st.markdown("📌 Upload PDFs and ask questions using RAG")

# -------------------- HELPERS --------------------
def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_documents(documents):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

@st.cache_resource(show_spinner=False)
def get_local_embeddings():
    # Local, free embeddings (no HuggingFace API key required)
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def create_retriever(texts):
    embeddings = get_local_embeddings()

    vectordb = Chroma.from_documents(
        texts,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR.as_posix()
    )
    vectordb.persist()

    return vectordb.as_retriever(search_kwargs={"k": 5})

def get_qa_chain(retriever):
    llm = ChatOpenAI(temperature=0)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

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
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                    dir=TMP_DIR.as_posix()
                ) as tmp:
                    tmp.write(file.read())
                    pdf_path = tmp.name

                docs = load_pdf(pdf_path)
                all_docs.extend(docs)
                os.remove(pdf_path)

            texts = split_documents(all_docs)
            st.session_state.retriever = create_retriever(texts)
            st.session_state.qa_chain = get_qa_chain(st.session_state.retriever)

        st.success("✅ Documents processed successfully!")

# -------------------- CHAT --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" in st.session_state:
    query = st.chat_input("Ask a question about your documents")  # ✅ no trailing _
    if query:
        st.chat_message("human").write(query)

        result = st.session_state.qa_chain({
            "question": query,
            "chat_history": st.session_state.chat_history
        })

        answer = result["answer"]
        st.chat_message("ai").write(answer)

        st.session_state.chat_history.append((query, answer))
else:
    st.info("⬆️ Upload and process documents to start chatting.")
