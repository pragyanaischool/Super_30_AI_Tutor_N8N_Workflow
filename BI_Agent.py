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
def call_groq(prompt, model="llama3-70b-8192"):
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
        'suggested_tasks': [], 'current_task': None, 'explanation': ""
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

    # Option 2: Select from a pre-defined list
    if problem_source == "Select a pre-defined project":
        
        @st.cache_data
        def load_project_list():
            try:
                # Convert the Google Sheet share URL to a CSV export URL
                sheet_url = "https://docs.google.com/spreadsheets/d/1V7Vsi3nIvyyjAsHB428axgDrIFFq-VSczoNz9I0XF8Y/export?format=csv"
                projects_df = pd.read_csv(sheet_url)
                # FIX: Strip any leading/trailing whitespace from column names to prevent KeyErrors
                projects_df.columns = projects_df.columns.str.strip()
                return projects_df
            except Exception as e:
                st.error(f"Could not load project list from Google Sheets. Please ensure the link is public. Error: {e}")
                return None

        projects_df = load_project_list()

        if projects_df is not None:
            categories = projects_df['Category'].unique()
            selected_category = st.selectbox("Select a Project Category:", options=categories)

            if selected_category:
                filtered_projects = projects_df[projects_df['Category'] == selected_category]
                problem_statements = ["-"] + filtered_projects['Problem Statement'].tolist()
                
                selected_problem = st.selectbox("Select a Problem Statement:", options=problem_statements)

                if selected_problem and selected_problem != "-":
                    project_details = filtered_projects[filtered_projects['Problem Statement'] == selected_problem].iloc[0]
                    st.session_state.problem_statement = project_details['Problem Statement']
                    dataset_url = project_details['Dataset URL']
                    
                    st.info(f"**Selected Problem:** {st.session_state.problem_statement}")
                    st.markdown(f"**Dataset Source:** [Link]({dataset_url})")

                    try:
                        with st.spinner("Loading dataset for the selected project..."):
                            st.session_state.df = pd.read_csv(dataset_url)
                    except Exception as e:
                        st.error(f"Failed to load data from URL: {e}")
                        st.session_state.df = None

    # Option 1: Define your own problem statement
    else:
        st.text_area("Start with a basic problem statement or goal:", key="problem_statement")
        
        if st.button("Enhance with AI"):
            if st.session_state.problem_statement:
                with st.spinner("AI is refining your problem statement..."):
                    prompt = f"""
                    A user has provided the following basic problem statement for a data science project:
                    '{st.session_state.problem_statement}'
                    Enhance this into a clear, concise, and well-defined problem statement. 
                    Focus on the potential objectives and success metrics.
                    """
                    st.session_state.enhanced_problem_statement = call_groq(prompt)
            else:
                st.warning("Please provide a basic problem statement first.")

        if st.session_state.enhanced_problem_statement:
            st.text_area("Edit and confirm the final problem statement:", key="enhanced_problem_statement", height=150)

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

    # This part is common to both options and will display the dataframe once loaded
    if st.session_state.df is not None:
        st.success("Data loaded successfully!")
        st.dataframe(st.session_state.df.head())
        st.info(f"Dataset has **{st.session_state.df.shape[0]}** rows and **{st.session_state.df.shape[1]}** columns.")

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
        
        # --- Task Suggestion ---
        if st.button("Suggest Analysis Steps", type="primary"):
            with st.spinner("AI is planning your analysis..."):
                prompt = f"""
                Given a dataset with columns {list(st.session_state.df.columns)} 
                and the problem statement '{st.session_state.enhanced_problem_statement or st.session_state.problem_statement}',
                suggest a list of 5-7 numbered steps for exploratory data analysis and preprocessing.
                Be specific. For example: '1. Analyze the distribution of the 'Age' column.'
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
                        
                        # After successful run, get an explanation
                        if "error" not in st.session_state.result:
                             with st.spinner("Generating explanation..."):
                                prompt = f"""
                                A data science task was: '{st.session_state.current_task}'.
                                The following Python code was successfully executed to complete it:
                                ```python
                                {st.session_state.code}
                                ```
                                Explain what this code does, why it's useful for the given task, and what the key takeaways from its potential output might be.
                                """
                                st.session_state.explanation = call_groq(prompt, model="llama3-8b-8192")

            # --- Output and Explanation ---
            with col2:
                st.subheader("📊 Output & Explanation")
                result = st.session_state.result
                if not result:
                    st.info("Output will be displayed here.")
                
                if "image_data" in result:
                    st.image(base64.b64decode(result["image_data"]), caption="Generated Plot")
                
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
                        They wrote this code:
                        --- CODE ---
                        {st.session_state.code}
                        --- END CODE ---
                        It produced this error:
                        --- ERROR ---
                        {result['error']}
                        --- END ERROR ---
                        Explain the error simply and provide the corrected code block.
                        """
                        with st.spinner("Debugging..."):
                            fix = call_groq(prompt)
                            if fix:
                                st.markdown(fix)
