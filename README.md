# 🥗 NutriGenie AI - Personalized Meal Planner

**NutriGenie AI** is an intelligent, personalized meal recommendation system built with **[Streamlit](https://streamlit.io/)** and **LangChain**, powered by **Groq's LLM API**.  
It takes a user's health profile and a custom recipe dataset, then generates **tailored meal recommendations** with nutritional analysis, dietary guidelines, ingredient substitutions, and justifications.

---

## 📌 Features
- **User Health Profile Form** — Collects personal health data and dietary needs.
- **Recipe Dataset Upload** — Supports `.csv` and `.xlsx` recipe files.
- **Document Loading & Processing** — Reads recipes into LangChain `Document` objects.
- **Text Chunking & Splitting** — Breaks large documents into manageable chunks using `RecursiveCharacterTextSplitter`.
- **Vector Embeddings** — Generates embeddings using `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`).
- **In-Memory Vector Store** — Stores and indexes recipe chunks for similarity search.
- **Similarity Search** — Retrieves the top `k` most relevant recipes for a given query.
- **Prompt Engineering** — Uses `ChatPromptTemplate` to feed health profile, query, and relevant recipes to the model.
- **LLM Integration** — Connects to Groq's API (`deepseek-r1-distill-llama-70b` by default).
- **Interactive Chat UI** — Uses Streamlit's `chat_input` and `chat_message` components.

---

## 📂 Project Structure
```
NutriGenie/
│
├── app.py # 🎯 Main Streamlit application (core logic & UI)
├── requirements.txt # 📦 Python dependencies for the project
├── README.md # 📄 Project documentation
└── .env # 🔐 Environment variables (API keys, model config)

---
```

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2️⃣ Create a Virtual Environment
``` bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```
### 3️⃣ Install Dependencies
``` bash
pip install -r requirements.txt
```
requirements.txt should include:
``` txt
streamlit
langchain
langchain-community
langchain-groq
langchain-huggingface
pandas
python-dotenv
openpyxl
```

### 4️⃣ Set Up Environment Variables

``` env
RECIPE_GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=deepseek-r1-distill-llama-70b
```

## 🚀 Running the App Locally
``` bash
streamlit run app.py
```
Your app will run at: **(http://localhost:8501)**

## 📜 Detailed Code Walkthrough

## 

Below is a **line-by-line breakdown** of the application, referencing relevant **official documentation**.

* * *

### 1️⃣ Environment Setup

## 

```
from dotenv import load_dotenv 
load_dotenv()  
RECIPE_GROQ_API_KEY = os.environ["RECIPE_GROQ_API_KEY"] 
GROQ_MODEL = os.environ.get("GROQ_MODEL", "deepseek-r1-distill-llama-70b")
```  
*   **`dotenv`**: Loads environment variables from `.env`.  
    📄 Docs: [python-dotenv](https://pypi.org/project/python-dotenv/)
    
*   Retrieves the **Groq API key** and **model name** for LLM calls.
    

* * *

### 2️⃣ Streamlit UI Styling

##            
```
                st.markdown("""
                <style>     
                .stApp { background-color: #0E1117; color: #FFFFFF; }     
                .stFileUploader { background-color: #1E1E1E; border: 1px solid #3A3A3A; }     
                h1, h2, h3 { color: #00FFAA !important; }
                </style> """
                , unsafe_allow_html=True)
```

Custom CSS to set a **dark theme**.
    
*   Streamlit allows HTML/CSS injection with `unsafe_allow_html=True`.
    

**`streamlit`**: Framework for building interactive data apps.  
📄 Docs: [Customizing Streamlit](https://docs.streamlit.io/library/advanced-features/configuration)

* * *

### 3️⃣ Prompt Template

## 

```
prompt_template =

""" You are a personalized meal planning AI assistant. ...
**Meal Recommendations** with clear justification linked to the user's health profile. """
```

*   Provides **instructions** and **structured output**.
    
*   Uses `{placeholders}` for **dynamic variables**.
    

**`langchain`**: Framework for developing applications powered by language models.  
📄 Docs: [LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates)

* * *

### 4️⃣ Embedding & Vector Store Initialization

## 

```
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
document_vector_db = InMemoryVectorStore(embedding_model)
```

*   **HuggingFaceEmbeddings**: Generates semantic vector representations of text.  
    **`langchain-huggingface`**: Integration of HuggingFace embeddings into LangChain workflows.  
    📄 Docs: [HuggingFace Embeddings in LangChain](https://python.langchain.com/docs/integrations/text_embedding/huggingface)


    
*   **InMemoryVectorStore**: Stores embeddings in memory for **fast retrieval**.  
  **`vectorstores`**: Storage and retrieval of vector embeddings for similarity search.  
  📄 Docs: [Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores)

    

* * *

### 5️⃣ LLM Model Initialization

## 

```
lang_model = ChatGroq(groq_api_key=RECIPE_GROQ_API_KEY,
                      model_name=GROQ_MODEL,
                      temperature=0.3, 
                      max_tokens=1024)
```

*   **ChatGroq**: LangChain wrapper for Groq’s LLM API.
    
*   `temperature`: Controls randomness (low = deterministic).
    
*   `max_tokens`: Caps output length.
    

**`langchain-groq`**: LangChain integration for Groq LLM inference.  
📄 Docs: [LangChain Groq](https://python.langchain.com/docs/integrations/llms/groq)

* * *

### 6️⃣ Dataset Loading

## 

```
def load_recipe_dataset(file):
  if file.name.endswith(".csv"):
    df = pd.read_csv(...)
  elif file.name.endswith(".xlsx"):
    df = pd.read_excel(...)
  documents = [Document(page_content=" | ".join(map(str, row)), metadata={"row": idx}) ...]
  return documents
```

*   Reads CSV/Excel into **Pandas DataFrame**.
    
*   Converts each row into a LangChain `Document`.
    

**`langchain`**: Document handling in LangChain.  
📄 Docs: [LangChain Document](https://python.langchain.com/docs/modules/data_connection/document_loaders)

* * *

### 7️⃣ Text Chunking

## 

```
def chunk_documents(documents):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  return text_splitter.split_documents(documents)
```

*   Splits large texts into **overlapping chunks** to preserve context.
    
*   Overlap = prevents breaking sentences mid-meaning.
    

**`langchain`**: Text splitting utility for chunking large documents.  
📄 Docs: [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter)


* * *

### 8️⃣ Indexing

## 

``` 
def index_documents(document_chunks):
  document_vector_db.add_documents(document_chunks, batch_size=32)
```

*   Embeds chunks and stores them in **vector DB**.
    
*   `batch_size=32` → Efficient bulk embedding.
    

* * *

### 9️⃣ Similarity Search

## 

```
def find_related_documents(query):
  return document_vector_db.similarity_search(query, k=3)
```

*   Finds top `k` documents most similar to the query vector.
    

**`langchain`**: Find relevant documents using embedding-based similarity.  
📄 Docs: [Similarity Search](https://python.langchain.com/docs/modules/data_connection/retrievers/similarity)

* * *

### 🔟 Response Generation

## 

```
def generate_response(query, related_docs, health_profile):
  context = "\n\n".join([doc.page_content for doc in related_docs])
  prompt = ChatPromptTemplate.from_template(prompt_template)
  response_chain = prompt | lang_model
  return response_chain.invoke({ ... })
```

*   **Chains** the prompt template with the LLM.
    
*   Injects **health profile**, **query**, and **recipe context** into the prompt.
    

**`langchain`**: Composable pipelines for LLM operations.  
📄 Docs: [LangChain Chains](https://python.langchain.com/docs/modules/chains)

* * *

### 1️⃣1️⃣ User Interaction

## 

*   **Health Profile Form** — Saves user data and indexes recipes.
    
*   **Chat Interface** — Asks for meal recommendations and displays AI responses.
    

**`streamlit`**: Form elements for structured user input.  
📄 Docs: [Streamlit Forms](https://docs.streamlit.io/library/api-reference/control-flow/st.form)
 
**`streamlit`**: Chat interface components for conversational apps.  
📄 Docs: [Streamlit Chat Elements](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)

* * *

## 🔄 Workflow Orchestration

## 

1.  **User submits health profile & recipe dataset**.
    
2.  **File is loaded** → `Document` objects.
    
3.  **Text is chunked** for better vectorization.
    
4.  **Chunks embedded & indexed** into in-memory vector store.
    
5.  **User query triggers similarity search** to find relevant recipes.
    
6.  **Prompt is constructed** with health profile + relevant recipes.
    
7.  **Groq LLM generates recommendations**.
    
8.  **Results displayed** in the Streamlit chat interface.
    

* * *

## 📊 Example Usage

## 

**Upload CSV/Excel**, enter your health details:

```
Name: Alice
Age: 30
Sex: Female
Height: 165 cm
Weight: 60 kg
Activity Level: Moderately Active 
Allergies: Peanuts
Dietary Restrictions: Vegan
Health Conditions: None
```

**Query:**

> What meals do you recommend for dinner?

**AI Output:**

*   Nutritional breakdown
    
*   Dietary guidelines
    
*   Substitution suggestions
    
*   Meal recommendations with reasoning
    

* * *

## 📜 License

## 

MIT License

* * *

## 📚 References

## 

*   Streamlit Docs
    
*   LangChain Docs
    
*   HuggingFace Embeddings
    
*   Groq API
