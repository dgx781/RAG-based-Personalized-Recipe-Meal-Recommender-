from crewai import LLM, Agent, Task, Crew, Process
from crewai.flow import Flow
import requests
from crewai import LLM
from crewai.tools import tool

# -----------------------------
# LLM WRAPPER
# -----------------------------

def get_crewai_llm(api_key, model_name):
    return LLM(
        model=model_name,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        max_tokens=1000
    )


# -----------------------------
# TOOLS
# -----------------------------
@tool("nutrition_tool")
def nutrition_tool(food_name):
    """Fetch nutrition data for a given food item using Edamam API."""
    try:
        url = f"https://api.edamam.com/api/nutrition-data?app_id=b0db4938&app_key=e5a0288d0fb850480750b1d50ed3e2ab&ingr={food_name}"
        res = requests.get(url).json()

        return {
            "calories": res.get("calories"),
            "protein": res.get("totalNutrients", {}).get("PROCNT", {}).get("quantity"),
            "fat": res.get("totalNutrients", {}).get("FAT", {}).get("quantity"),
            "carbs": res.get("totalNutrients", {}).get("CHOCDF", {}).get("quantity"),
        }
    except:
        return "Nutrition data unavailable"

@tool("web_search_tool")
def web_search_tool(query):
    """Perform a web search using SerpAPI and return results."""
    try:
        url = f"https://serpapi.com/search?q={query}&api_key=db77fe3438e79e96b24b6434c79eddb9e958c3a667feaeefd25cc25a1381f538"
        return requests.get(url).json()
    except:
        return "Search failed"


# -----------------------------
# HARD FILTER
# -----------------------------
def strict_filter(text, health_profile):
    if "peanut" in text.lower() and "peanut allergy" in health_profile.lower():
        return False
    return True


# -----------------------------
# BUILD CREW
# -----------------------------
def build_crew(embedding_model, query_fn, llm):

    from crewai.tools import tool

    @tool("vector_search")
    def vector_search(query: str):
        """
        Search internal recipe database using vector embeddings.
        """
        results = query_fn(query, embedding_model)

        # Optional: format output (important for LLM readability)
        return "\n".join([
            f"{i+1}. {r['text'][:200]} (score: {r['score']})"
            for i, r in enumerate(results[:3])
        ])

    # 🔥 OPTIMIZED AGENTS

    nutrition_agent = Agent(
    role="Clinical Nutrition Expert",
    goal="""
    Analyze the user's health profile and extract precise dietary constraints.

    Responsibilities:
    1. Identify all health conditions (e.g., asthma, bronchitis, jaundice)
    2. Detect allergies and strict food restrictions
    3. Identify foods to completely avoid
    4. Determine nutritional requirements (low-fat, high-protein, etc.)
    5. Output structured and medically relevant insights
    """,
    backstory="Certified clinical dietician with expertise in disease-specific nutrition planning",
    llm=llm,
    verbose=True
    )

    retrieval_agent = Agent(
    role="Semantic Retrieval Specialist",
    goal="""
    Retrieve diverse, relevant recipes using:
    1. Internal vector database (PRIMARY)
    2. Web search (SECONDARY fallback)

    Always use vector_search first before web search.
    Avoid redundancy.
    Ensure diversity in retrieved recipes.
    """,
    backstory="Expert in semantic search, vector databases, and information retrieval systems",
    llm=llm,
    tools=[vector_search, web_search_tool],
    verbose=True
    )

    reasoning_agent = Agent(
    role="Nutritional Reasoning Engine",
    goal="""
    Filter and rank recipes based on health constraints and nutritional suitability.

    Responsibilities:
    1. Remove recipes that violate allergies or restrictions
    2. Eliminate foods harmful to medical conditions
    3. Use nutrition_tool to evaluate nutritional values when required
    4. Rank recipes based on:
        • Safety (highest priority)
        • Nutritional value
        • Relevance to query
    5. Produce a clean, prioritized list of safe recipes
    """,
    backstory="Experienced nutritionist specializing in dietary filtering, risk assessment, and health-safe meal selection",
    llm=llm,
    tools=[nutrition_tool],
    verbose=True
    )

    recommendation_agent = Agent(
    role="Meal Planner",
    goal="""
    Generate a structured and personalized meal plan using filtered recipes.

    Responsibilities:
    1. Create a 3–5 meal plan including:
        • Breakfast
        • Lunch
        • Dinner
        • Optional snacks
    2. Ensure all meals comply with health constraints
    3. Maintain variety across meals
    4. Keep output structured and readable
    5. Use past history if available for personalization
    """,
    backstory="Expert meal planner specializing in personalized nutrition and balanced diet design",
    llm=llm,
    verbose=True
    )

    explanation_agent = Agent(
    role="Explainability Agent",
    goal="""
    Provide clear explanations for each recommended meal.

    Responsibilities:
    1. Explain why each meal is suitable for the user's health conditions
    2. Highlight nutritional benefits
    3. Explain avoided risks (e.g., low fat for jaundice)
    4. Keep explanations simple, medically accurate, and easy to understand
    """,
    backstory="Nutrition educator skilled in translating complex dietary reasoning into simple explanations",
    llm=llm,
    verbose=True
    )
    # -----------------------------
    # TASKS (WITH MEMORY)
    # -----------------------------
    # -----------------------------
# TASKS (STRUCTURED + TOOL-AWARE)
# -----------------------------

    task1 = Task(
    description="""
    Analyze the user's health profile: {health_profile}

    Instructions:
    - Extract all medical conditions (e.g., asthma, bronchitis, jaundice)
    - Identify food restrictions and allergies
    - Identify foods to strictly avoid
    - Determine nutritional requirements (low-fat, high-protein, etc.)
    - Structure the output clearly
    """,
    agent=nutrition_agent,
    expected_output="""
    A structured summary including:
    - Health conditions
    - खाद्य restrictions and allergies
    - Foods to avoid
    - Nutritional requirements
    """
    )

    task2 = Task(
    description="""
    Retrieve recipes for the query: {query}

    Instructions:
    - FIRST call `vector_search`
    - If results are insufficient, THEN call `web_search_tool`
    - Combine results from both sources
    - Remove duplicate or highly similar recipes
    - Ensure diversity in meal types
    """,
    agent=retrieval_agent,
    expected_output="""
    A high-quality, diverse list of recipes retrieved using vector and web search,
    with minimal redundancy and good coverage of meal types
    """
    )

    task3 = Task(
    description="""
    Filter and rank the retrieved recipes

    Instructions:
    - Use health constraints from Task 1
    - Remove recipes containing allergens or harmful ingredients
    - Call `nutrition_tool` to evaluate nutritional values if needed
    - Rank recipes based on:
        • Safety (highest priority)
        • Nutritional suitability
        • Relevance to query
    - Keep only the best candidates
    """,
    agent=reasoning_agent,
    expected_output="""
    A filtered and ranked list of safe, nutritionally appropriate recipes
    tailored to the user's health profile
    """
    )


    task4 = Task(
    description="""
    Generate a personalized meal plan

    Inputs:
    - Query: {query}
    - Health profile: {health_profile}
    - History: {history}

    Instructions:
    - Use only filtered recipes from previous step
    - Create a structured 3–5 meal plan including:
        • Breakfast
        • Lunch
        • Dinner
        • (Optional snacks)
    - Ensure variety across meals
    - Ensure all meals comply with health constraints
    - Keep explanations concise
    """,
    agent=recommendation_agent,
    expected_output="""
    A structured meal plan including breakfast, lunch, and dinner options,
    each aligned with the user’s health conditions and dietary restrictions
    """
    )

    task5 = Task(
    description="""
    Explain the generated meal plan

    Instructions:
    - For each meal, explain:
        • Why it is suitable for the user's health condition
        • Nutritional benefits
        • Any avoided risks (e.g., low fat for jaundice)
    - Keep explanation simple and medically relevant
    - Ensure clarity and readability
    """,
    agent=explanation_agent,
    expected_output="Clear, concise explanations for each meal recommendation, highlighting health benefits and suitability"
    )

    crew = Crew(
        agents=[
            nutrition_agent,
            retrieval_agent,
            reasoning_agent,
            recommendation_agent,
            explanation_agent
        ],
        tasks=[task1, task2, task3, task4, task5],
        process=Process.sequential,
        verbose=True
    )

    return crew