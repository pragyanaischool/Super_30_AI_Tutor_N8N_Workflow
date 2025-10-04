import streamlit as st
import pandas as pd
import requests
import base64
import os
from groq import Groq
from io import StringIO

# --- Page Configuration ---
st.set_page_config(
    page_title="Galileo AI Data Science Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & API Initialization ---
N8N_WEBHOOK_URL = "YOUR_N8N_PRODUCTION_WEBHOOK_URL_HERE" 

try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("Groq API key not found. Please add it to your Streamlit secrets.")
    st.stop()

# --- Helper Functions ---
def call_groq(prompt, model="llama-3.3-70b-versatile"):
    """Function to call the GROQ API and get a response."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model, temperature=0.7)
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the Groq API: {e}")
        return None

def run_code_in_n8n(code_to_run, df=None):
    """Sends code and optional dataframe to n8n for execution."""
    payload = {"code": code_to_run}
    if df is not None:
        payload['df_json'] = df.to_json(orient='split')
    try:
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to n8n execution engine: {e}"}

# --- Initialize Session State ---
def init_session_state():
    # This function is to avoid cluttering the main script area
    state_vars = {
        'code': "# Your Python code will appear here", 'result': {}, 'df': None,
        'problem_statement': "", 'enhanced_problem_statement': "", 'target_variable': None,
        'suggested_tasks': [], 'current_task': None, 'explanation': "", 'plan_generated_for': None
    }
    for var, default_val in state_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_val

init_session_state()

# --- UI Layout ---
st.sidebar.title("🎓 Galileo AI Tutor")
st.sidebar.markdown("Your personal guide through the Data Science lifecycle.")

lifecycle_step = st.sidebar.radio("Choose a learning module:",
    ("1. Define Problem & Load Data", "2. Understand Your Data", "3. Guided Data Analysis"))

st.title(f"Module: {lifecycle_step}")

# --- Module 1: Define Problem & Load Data ---
if lifecycle_step == "1. Define Problem & Load Data":
    st.header("Step 1: Define Your Goal")

    problem_source = st.radio(
        "How would you like to start?",
        ("Define my own problem statement", "Select a pre-defined project"),
        key="problem_source",
        horizontal=True
    )

    # --- BLOCK FOR PRE-DEFINED PROJECTS ---
    if problem_source == "Select a pre-defined project":
        
        @st.cache_data
        def get_project_categories():
            """Reads the sheet names from the Google Sheet, which are the categories."""
            try:
                #https://docs.google.com/spreadsheets/d/1V7Vsi3nIvyyjAsHB428axgDrIFFq-VSczoNz9I0XF8Y
                sheet_url = "https://docs.google.com/spreadsheets/d/1V7Vsi3nIvyyjAsHB428axgDrIFFq-VSczoNz9I0XF8Y/export?format=xlsx"
                excel_file = pd.ExcelFile(sheet_url, engine='openpyxl')
                return excel_file.sheet_names
            except Exception as e:
                st.error(f"Could not load project categories from Google Sheets. Error: {e}")
                return []

        @st.cache_data
        def load_project_sheet(category):
            """Loads a specific sheet (category) from the Google Sheet into a DataFrame."""
            try:
                sheet_url = "https://docs.google.com/spreadsheets/d/1V7Vsi3nIvyyjAsHB428axgDrIFFq-VSczoNzI0XF8Y/export?format=xlsx"
                projects_df = pd.read_excel(sheet_url, sheet_name=category, engine='openpyxl')
                projects_df.columns = projects_df.columns.str.strip()
                return projects_df
            except Exception as e:
                st.error(f"Could not load the project list for the '{category}' category. Error: {e}")
                return None

        categories = get_project_categories()
        
        if categories:
            selected_category = st.selectbox("Select a Project Category:", options=categories)

            if selected_category:
                projects_df = load_project_sheet(selected_category)
                
                if projects_df is not None:
                    problem_statements = ["-"] + projects_df['Problem Statement'].tolist()
                    selected_problem = st.selectbox("Select a Problem Statement:", options=problem_statements)

                    if selected_problem and selected_problem != "-":
                        project_details = projects_df[projects_df['Problem Statement'] == selected_problem].iloc[0]
                        
                        st.session_state.problem_statement = project_details.get('Problem Statement', '')
                        
                        # Only generate the plan if it's a new project selection
                        if st.session_state.plan_generated_for != st.session_state.problem_statement:
                            # Fetch all the detailed planning columns
                            refined_statement = project_details.get('Refined Problem Statement', 'Not specified.')
                            key_questions = project_details.get('Key Questions for Exploration', 'Not specified.')
                            key_analytics = project_details.get('Key Analytics & Statistics', 'Not specified.')
                            viz_ideas = project_details.get('Data Visualization Ideas', 'Not specified.')
                            potential_insights = project_details.get('Potential Data Insights', 'Not specified.')
                            
                            # Build the prompt for the AI to generate a comprehensive plan
                            generation_prompt = f"""
                            As a data science project manager, synthesize the following points for a project titled "{st.session_state.problem_statement}" into a comprehensive 'Detailed Project Plan & Goals'.
                            The output should be well-structured using markdown headings for clarity.

                            - **Base Problem Statement:** {refined_statement}
                            - **Key Questions to Explore:** {key_questions}
                            - **Core Analytics & Statistics to Use:** {key_analytics}
                            - **Potential Visualization Ideas:** {viz_ideas}
                            - **Expected Potential Insights:** {potential_insights}

                            Combine these points into a cohesive and actionable project plan.
                            """
                            with st.spinner("AI is generating a detailed project plan..."):
                                st.session_state.enhanced_problem_statement = call_groq(generation_prompt)
                                st.session_state.plan_generated_for = st.session_state.problem_statement # Mark as generated for this project

                        key_steps_raw = project_details.get('Key Analytics Steps', '')
                        st.session_state.suggested_tasks = [step.strip() for step in (key_steps_raw.split('\n') if key_steps_raw and isinstance(key_steps_raw, str) else []) if step.strip()]
                        dataset_url = project_details['Dataset URL']
                        
                        st.info(f"**Selected Project:** {st.session_state.problem_statement}")
                        st.markdown(f"**Dataset Source:** [Link]({dataset_url})")

                        try:
                            with st.spinner("Loading dataset..."):
                                st.session_state.df = pd.read_csv(dataset_url)
                        except Exception as e:
                            st.error(f"Failed to load data from URL: {e}")
                            st.session_state.df = None

    # --- BLOCK FOR USER-DEFINED PROJECTS ---
    else:
        st.text_area("Start with a basic problem statement or goal:", key="problem_statement", height=100)
        
        if st.button("Enhance with AI"):
            if st.session_state.problem_statement:
                with st.spinner("AI is refining your problem statement..."):
                    prompt = f"""
                    A user has provided the following basic problem statement: '{st.session_state.problem_statement}'
                    Enhance this into a clear, concise, and well-defined problem statement.
                    """
                    st.session_state.enhanced_problem_statement = call_groq(prompt)
            else:
                st.warning("Please provide a basic problem statement first.")

        st.header("Step 2: Provide Your Data")
        data_source = st.radio("Choose data source:", ("Upload a CSV file", "Provide a URL"))
        if data_source == "Upload a CSV file":
            uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
            if uploaded_file:
                st.session_state.df = pd.read_csv(uploaded_file)
        else:
            url = st.text_input("Enter the URL of a raw CSV file:")
            if url:
                try:
                    st.session_state.df = pd.read_csv(url)
                except Exception as e:
                    st.error(f"Failed to load data from URL: {e}")

    # --- COMMON DISPLAY BLOCK (AFTER DATA IS LOADED) ---
    if st.session_state.df is not None:
        st.header("Step 3: Review Loaded Data & Plan")
        st.success("Data loaded successfully!")
        st.dataframe(st.session_state.df.head())
        st.info(f"Dataset has **{st.session_state.df.shape[0]}** rows and **{st.session_state.df.shape[1]}** columns.")
        
        st.subheader("Project Plan")
        st.markdown("Review and edit the project plan. This will guide the AI in the next modules.")
        
        st.text_area(
            "**Detailed Project Plan & Goals:**", 
            key="enhanced_problem_statement", 
            height=300
        )
        
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            if st.button("Refine with AI"):
                if st.session_state.enhanced_problem_statement:
                    with st.spinner("AI is refining the plan..."):
                        refine_prompt = f"Refine the following data science project plan to be more detailed, structured, and actionable:\n\n---\n{st.session_state.enhanced_problem_statement}\n---"
                        refined_plan = call_groq(refine_prompt)
                        if refined_plan:
                            st.session_state.enhanced_problem_statement = refined_plan
                        st.rerun()
                else:
                    st.warning("Project plan is empty. Cannot refine.")

        with col2:
            if st.button("Save & Continue", type="primary"):
                st.success("Project plan saved! Please proceed to the next module from the sidebar.")

        tasks_text = "\n".join(st.session_state.suggested_tasks)
        edited_tasks_text = st.text_area(
            "**Key Analytics Steps:** (One task per line)", 
            value=tasks_text, 
            height=200,
            placeholder="List the analysis steps you want to perform, one per line. Or click 'Suggest Analysis Steps' in Module 3."
        )
        st.session_state.suggested_tasks = [step.strip() for step in edited_tasks_text.split('\n') if step.strip()]

# --- Module 2: Understand Your Data ---
elif lifecycle_step == "2. Understand Your Data":
    if st.session_state.df is None:
        st.warning("Please load a dataset in Module 1 first.")
    else:
        st.header("Exploratory Data Overview")
        st.dataframe(st.session_state.df.describe())
        
        buffer = StringIO()
        st.session_state.df.info(buf=buffer)
        st.text_area("Dataframe Info (Data Types & Non-Null Counts):", buffer.getvalue(), height=250)
        
        st.header("Identify Target Variable")
        st.markdown("Select the column your project aims to predict or analyze most deeply.")
        st.session_state.target_variable = st.selectbox("Target Variable:", options=[None] + list(st.session_state.df.columns))
        if st.session_state.target_variable:
            st.success(f"Target variable set to: **{st.session_state.target_variable}**")

# --- Module 3: Guided Data Analysis (The Iterative Loop) ---
elif lifecycle_step == "3. Guided Data Analysis":
    if st.session_state.df is None:
        st.warning("Please load a dataset in Module 1 first.")
    else:
        st.header("Iterative Analysis")
        
        if not st.session_state.suggested_tasks:
            if st.button("Suggest Analysis Steps", type="primary"):
                with st.spinner("AI is planning your analysis..."):
                    prompt = f"""
                    Given a dataset with columns {list(st.session_state.df.columns)} 
                    and the project plan '{st.session_state.enhanced_problem_statement or st.session_state.problem_statement}',
                    suggest a list of 5-7 numbered steps for exploratory data analysis and preprocessing.
                    """
                    tasks = call_groq(prompt)
                    if tasks:
                        st.session_state.suggested_tasks = [t.strip() for t in tasks.split('\n') if t.strip()]

        if st.session_state.suggested_tasks:
            st.session_state.current_task = st.radio("Select a task to perform:", options=st.session_state.suggested_tasks)

        st.text_input("Or, define a custom task:", key="current_task")

        if st.session_state.current_task:
            st.info(f"**Current Task:** {st.session_state.current_task}")

            col1, col2 = st.columns([1, 1], gap="large")

            # --- Code Generation & Execution ---
            with col1:
                st.subheader("🤖 AI Code Generation")
                if st.button("Generate Code"):
                    with st.spinner("Galileo is writing code..."):
                        prompt = f"""
                        You are a data science assistant. For a pandas DataFrame 'df' and the task '{st.session_state.current_task}',
                        write a Python script using seaborn and matplotlib.
                        DataFrame columns: {list(st.session_state.df.columns)}.
                        Rules:
                        - Provide ONLY raw Python code. No explanations or markdown.
                        - Assume 'df' is loaded.
                        - Assume 'matplotlib.pyplot as plt' and 'seaborn as sns' are imported.
                        - Add a title to the plot.
                        """
                        code = call_groq(prompt)
                        if code:
                            st.session_state.code = code.strip().strip("```python").strip("```")

                st.text_area("Edit Code:", value=st.session_state.code, height=350, key="code_editor")
                
                if st.button("▶️ Execute Code"):
                    with st.spinner("Sending code to secure execution engine..."):
                        st.session_state.result = run_code_in_n8n(st.session_state.code, st.session_state.df)
                        
                        if "error" not in st.session_state.result:
                             with st.spinner("Generating explanation..."):
                                prompt = f"""
                                A data science task was: '{st.session_state.current_task}'.
                                The following Python code was executed:
                                ```python
                                {st.session_state.code}
                                ```
                                Explain what this code does and why it's useful for the task.
                                """
                                st.session_state.explanation = call_groq(prompt, model="llama-3.3-70b-versatile")

            # --- Output and Explanation ---
            with col2:
                st.subheader("📊 Output & Explanation")
                result = st.session_state.result
                if not result:
                    st.info("Output will be displayed here.")
                
                if "image_data" in result:
                    st.image(base64.b4decode(result["image_data"]), caption="Generated Plot")
                
                if st.session_state.explanation:
                    st.markdown("---")
                    st.subheader("👨‍🏫 Tutor's Explanation")
                    st.markdown(st.session_state.explanation)

                if "error" in result:
                    st.error("An error occurred:")
                    st.code(result["error"], language='bash')

                    if st.button("🤔 Ask Tutor to Debug"):
                        prompt = f"""
                        A student's goal was: "{st.session_state.current_task}".
                        They wrote this code: {st.session_state.code}
                        It produced this error: {result['error']}
                        Explain the error simply and provide the corrected code block.
                        """
                        with st.spinner("Debugging..."):
                            fix = call_groq(prompt)
                            if fix:
                                st.markdown(fix)
