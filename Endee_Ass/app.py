import os
import uuid
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from endee_client import create_index, reset_index, upsert_vectors, query_endee
from agentic_rag import build_crew, get_crewai_llm


# -----------------------------
# ENV
# -----------------------------
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# -----------------------------
# MODELS
# -----------------------------
@st.cache_resource
def get_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embedding_model = get_embedder()
crew_llm = get_crewai_llm(API_KEY, MODEL)

# -----------------------------
# MEMORY
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

# -----------------------------
# DATA PIPELINE
# -----------------------------
def load_documents(file):
    file.seek(0)

    try:
        df = pd.read_csv(
            file,
            encoding="utf-8",
            on_bad_lines="skip",
            engine="python"
        )
    except:
        file.seek(0)
        df = pd.read_csv(
            file,
            encoding="latin1",   # 🔥 BEST fallback for your dataset
            on_bad_lines="skip",
            engine="python"
        )

    # 🔥 Clean weird characters
    df = df.applymap(lambda x: str(x).encode("ascii", "ignore").decode())

    return [
        Document(page_content=" | ".join(map(str, row)))
        for _, row in df.iterrows()
    ]

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def index_documents(chunks):
    upsert_vectors(chunks, embedding_model)

# -----------------------------
# UI
# -----------------------------
st.title("🥗 NutriGenie AI — Agentic RAG")

health_profile = st.text_area("Health Profile")
file = st.file_uploader("Upload Dataset (CSV)")

if st.button("Process Dataset"):
    reset_index()
    docs = load_documents(file)
    chunks = chunk_documents(docs)
    index_documents(chunks)
    st.session_state.vector_ready = True
    st.success("Indexed successfully!")

query = st.text_input("Ask your query")

if st.button("Generate Plan"):

    crew = build_crew(embedding_model, query_endee, crew_llm)

    result = crew.kickoff({
        "query": query,
        "health_profile": health_profile,
        "history": st.session_state.chat_history[-2:]
    })

    st.write(result)

    # MEMORY UPDATE
    st.session_state.chat_history.append({
        "query": query,
        "response": result
    })

# -----------------------------
# SEMANTIC SEARCH
# -----------------------------
st.subheader("🔍 Semantic Search")

search = st.text_input("Search Recipes")

if search:
    results = query_endee(search, embedding_model)
    for r in results:
        st.write(r["text"])

# -----------------------------
# RECOMMENDATIONS
# -----------------------------
st.subheader("🍽️ Similar Recipes")

if query:
    results = query_endee(query, embedding_model)
    for r in results:
        st.write(r["text"])