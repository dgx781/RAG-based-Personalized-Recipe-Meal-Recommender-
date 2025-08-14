# app.py
import os
import io
import tempfile
import hashlib
import pandas as pd
import streamlit as st

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
RECIPE_GROQ_API_KEY = os.environ.get("RECIPE_GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "deepseek-r1-distill-llama-70b")

if not RECIPE_GROQ_API_KEY:
    st.error("Missing GROQ API key. Set RECIPE_GROQ_API_KEY in your environment.")
    st.stop()

st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FFFFFF; }
.stFileUploader { background-color: #1E1E1E; border: 1px solid #3A3A3A; border-radius: 5px; padding: 15px; }
h1, h2, h3 { color: #00FFAA !important; }
.block-label { color:#a8b3cf; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)


PROMPT_TEMPLATE = """
You are a personalized meal planning AI assistant.
Use the provided recipe dataset and the user's health profile to generate recommendations.

**User Health Profile:**
{health_profile}

**Query:**
{user_query}

**Relevant Recipes (top matches):**
{document_context}

Provide, in order:
1) Nutritional Analysis of the recommended meals.
2) Dietary Guidelines tailored to the user's profile.
3) Ingredient Substitutions (for allergies/restrictions/preferences).
4) 3–5 Meal Recommendations with short justifications tied to the health profile.

Be concise but specific. If data is insufficient, state assumptions explicitly.
"""

@st.cache_resource(show_spinner=False)
def get_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(
        groq_api_key=RECIPE_GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.3,
        max_tokens=1024,
    )

@st.cache_resource(show_spinner=False)
def get_vectorstore(_embedding):
    return InMemoryVectorStore(_embedding)


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@st.cache_data(show_spinner=False)
def bytes_to_tempfile(file_bytes: bytes, suffix: str) -> str:
    """Persist uploaded bytes into a temp file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        return tmp.name

@st.cache_data(show_spinner=False)
def load_as_documents(tmp_path: str, is_csv: bool, sample_rows: int | None = None) -> list[Document]:
    """
    Robust doc loader that uses pandas with encoding fallbacks and returns LangChain Documents.
    - Skips bad lines.
    - Optional sampling for very large files.
    """
    if is_csv:
        try:
            df = pd.read_csv(tmp_path, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(tmp_path, encoding="ISO-8859-1", on_bad_lines="skip")
    else:
        df = pd.read_excel(tmp_path)

    if sample_rows is not None and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=13).reset_index(drop=True)

    docs = [
        Document(
            page_content=" | ".join(map(str, row)),
            metadata={"row_index": int(idx)}
        )
        for idx, row in df.iterrows()
    ]
    return docs

def chunk_documents(documents: list[Document], chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return splitter.split_documents(documents)

def index_documents(vstore: InMemoryVectorStore, chunks: list[Document]):
    batch_size = 64
    prog = st.progress(0, text="Indexing embeddings...")
    for i in range(0, len(chunks), batch_size):
        vstore.add_documents(chunks[i:i+batch_size])
        prog.progress(min(100, int((i + batch_size) / max(1, len(chunks)) * 100)))
    prog.progress(100)

def build_context(vstore: InMemoryVectorStore, query: str, k: int = 4, per_doc_chars: int = 800) -> str:
    hits = vstore.similarity_search(query, k=k)
    trimmed = []
    for d in hits:
        txt = d.page_content
        trimmed.append(txt[:per_doc_chars])
    return "\n\n".join(trimmed)

def generate_response(llm: ChatGroq, health_profile: str, user_query: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    out = chain.invoke(
        {
            "health_profile": health_profile,
            "user_query": user_query,
            "document_context": context,
        }
    )
    return getattr(out, "content", out)

if "health_profile" not in st.session_state:
    st.session_state.health_profile = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

embedding_model = get_embedder()
lang_model = get_llm()
vector_store = get_vectorstore(embedding_model)

st.title("🥗 NutriGenie AI — Personalized Meal Planner")
st.markdown("### Get customized meal plans based on your health profile and recipe dataset")
st.markdown("---")

with st.form("health_form"):
    st.subheader("👤 Health Profile")

    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    height = st.number_input("Height (cm)", min_value=0.0, step=0.1)
    weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
    activity = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
    allergies = st.text_area("Allergies (comma separated)")
    restrictions = st.text_area("Dietary Restrictions (comma separated)")
    health_conditions = st.text_area("Health Conditions (comma separated)")

    st.markdown("<span class='block-label'>Upload your recipe dataset (CSV or Excel)</span>", unsafe_allow_html=True)
    recipe_file = st.file_uploader(" ", type=["csv", "xlsx"], label_visibility="collapsed")

    col1, col2, col3 = st.columns(3)
    save_profile_btn = col1.form_submit_button("💾 Save Profile")
    process_dataset_btn = col2.form_submit_button("📤 Process Dataset")
    generate_btn = col3.form_submit_button("🤖 Generate Plan")


if save_profile_btn:
    hp = f"""Name: {name}
        Age: {age}
        Sex: {sex}
        Height: {height} cm
        Weight: {weight} kg
        Activity Level: {activity}
        Allergies: {allergies}
        Dietary Restrictions: {restrictions}
        Health Conditions: {health_conditions}
        """
    st.session_state.health_profile = hp
    st.success("✅ Health profile saved!")

if process_dataset_btn:
    if not recipe_file:
        st.error("Please upload a recipe dataset first.")
    else:
        file_bytes = recipe_file.getvalue()
        st.session_state.file_hash = _hash_bytes(file_bytes)
        suffix = os.path.splitext(recipe_file.name)[1] or ".csv"

        with st.spinner("Saving uploaded file and parsing..."):
            tmp_path = bytes_to_tempfile(file_bytes, suffix)
            is_csv = recipe_file.name.lower().endswith(".csv")

        SAMPLE_ROWS = None 

        with st.spinner("Converting rows to documents..."):
            documents = load_as_documents(tmp_path, is_csv=is_csv, sample_rows=SAMPLE_ROWS)

        with st.spinner("Chunking documents..."):
            chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)

        with st.spinner("Embedding & indexing (batched)..."):
            index_documents(vector_store, chunks)

        st.session_state.dataset_loaded = True
        st.session_state.vector_ready = True
        st.success(f"✅ Dataset processed and indexed! Chunks: {len(chunks)}")


if generate_btn:
    if not st.session_state.health_profile:
        st.error("Please save your health profile first.")
    elif not st.session_state.vector_ready:
        st.error("Please process your recipe dataset first.")
    else:
        st.markdown("---")
        st.subheader("💬 Ask for Personalized Meal Recommendations")
        default_query = "Plan meals for the next 3 days considering my allergies and health conditions."
        user_query = st.text_input("Your query", value=default_query)

        if user_query.strip():
            with st.spinner("Retrieving context..."):
                context = build_context(vector_store, user_query, k=4, per_doc_chars=800)

            with st.spinner("Generating recommendations..."):
                result = generate_response(
                    lang_model, st.session_state.health_profile, user_query, context
                )

            st.markdown("### 🤖 NutriGenie Recommendations")
            st.write(result)


with st.expander("⚡ Tips to keep it fast"):
    st.markdown(
        "- Use CSVs with fewer than ~50k rows while prototyping.\n"
        "- Consider setting `SAMPLE_ROWS` to cap rows for very large files.\n"
        "- Keep queries focused; the retriever feeds only the top matches to the LLM.\n"
        "- You can re-upload a dataset and click **Process Dataset** to rebuild the index."
    )
