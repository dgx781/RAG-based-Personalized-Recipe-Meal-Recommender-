import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
import tempfile
import pandas as pd
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

RECIPE_GROQ_API_KEY = os.environ["RECIPE_GROQ_API_KEY"]
GROQ_MODEL = os.environ.get("GROQ_MODEL", "deepseek-r1-distill-llama-70b")

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

prompt_template = """
You are a personalized meal planning AI assistant.
Use the provided recipe dataset and the user's health profile to generate recommendations.

**User Health Profile:**
{health_profile}

**Query:**
{user_query}

**Relevant Recipes:**
{document_context}

For your answer, provide:
1. **Nutritional Analysis** of the recommended meals.
2. **Dietary Guidelines** specific to the user.
3. **Ingredient Substitutions** for allergies, dietary restrictions, or preferences.
4. **Meal Recommendations** with clear justification linked to the user's health profile.
"""

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
document_vector_db = InMemoryVectorStore(embedding_model)
lang_model = ChatGroq(
    groq_api_key=RECIPE_GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0.3,
    max_tokens=1024,
)

def load_recipe_dataset(file):
    if not file:
        return None

    # Save uploaded file to temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    if file.name.endswith(".csv"):
        try:
            df = pd.read_csv(tmp_path, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(tmp_path, encoding="ISO-8859-1", on_bad_lines="skip")

        documents = [
            Document(page_content=" | ".join(map(str, row)), metadata={"row": idx})
            for idx, row in df.iterrows()
        ]
        return documents

    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(tmp_path)
        documents = [
            Document(page_content=" | ".join(map(str, row)), metadata={"row": idx})
            for idx, row in df.iterrows()
        ]
        return documents

    else:
        st.error("Unsupported file format. Please upload CSV or Excel.")
        return None

def chunk_documents(documents):
    """
    Split documents into smaller chunks for better processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return text_splitter.split_documents(documents)


def index_documents(document_chunks):
    """Index the recipe dataset."""
    document_vector_db.add_documents(document_chunks, batch_size = 32)

def find_related_documents(query):
    """Find similar recipes for the query."""
    return document_vector_db.similarity_search(query, k=3)

def generate_response(query, related_docs, health_profile):
    """Generate a personalized meal recommendation."""
    context = "\n\n".join([doc.page_content for doc in related_docs])
    prompt = ChatPromptTemplate.from_template(prompt_template)
    response_chain = prompt | lang_model
    return response_chain.invoke({
        "user_query": query,
        "document_context": context,
        "health_profile": health_profile
    })

st.title("🥗 NutriGenie AI - Personalized Meal Planner")
st.markdown("### Get customized meal recommendations based on your health profile")
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
    
    recipe_file = st.file_uploader(
        "Upload Recipe Dataset (CSV or Excel)",
        type=["csv", "xlsx"],
        help="Provide your recipe dataset for recommendations."
    )

    submitted = st.form_submit_button("💾 Save Profile")

if submitted:
    if not recipe_file:
        st.error("Please upload a recipe dataset.")
    else:
        with st.spinner("Processing your health profile and dataset..."):
            df = load_recipe_dataset(recipe_file)
            if df is not None:
                chunks = chunk_documents(df)
                index_documents(chunks)
                
                health_profile = f"""
                Name: {name}
                Age: {age}
                Sex: {sex}
                Height: {height} cm
                Weight: {weight} kg
                Activity Level: {activity}
                Allergies: {allergies}
                Dietary Restrictions: {restrictions}
                Health Conditions: {health_conditions}
                """
                
                st.session_state["health_profile"] = health_profile
                st.success("✅ Profile saved and recipes indexed successfully!")
                st.progress(100)

if "health_profile" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Ask for Personalized Meal Recommendations")
    user_input = st.chat_input("What meals do you recommend for me?")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Generating recommendations..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_response(user_input, relevant_docs, st.session_state["health_profile"])
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response.content)