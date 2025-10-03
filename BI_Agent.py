import streamlit as st
import pandas as pd
import requests
import base64
import os
from groq import Groq

# --- Page Configuration ---
st.set_page_config(
    page_title="Galileo AI Data Science Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & API Initialization ---

# IMPORTANT: Paste your n8n Webhook URL here
# Use the "Production URL" from your n8n Webhook node
N8N_WEBHOOK_URL = "YOUR_N8N_PRODUCTION_WEBHOOK_URL_HERE" 

# Initialize the Groq client using the API key from Streamlit's secrets
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
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the Groq API: {e}")
        return None

def run_code_in_n8n(code_to_run, df=None):
    """Sends code and optional dataframe to n8n for execution."""
    payload = {"code": code_to_run}
    if df is not None:
        # Serialize dataframe to JSON to send it over HTTP
        payload['df_json'] = df.to_json(orient='split')
        
    try:
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to n8n execution engine: {e}"}

# --- Streamlit App UI ---

# Initialize session state variables
if 'code' not in st.session_state:
    st.session_state.code = "# Your Python code will appear here"
if 'result' not in st.session_state:
    st.session_state.result = {}

st.sidebar.title("🎓 Galileo AI Tutor")
st.sidebar.markdown("Your personal guide through the Data Science lifecycle.")

# Sidebar navigation
lifecycle_step = st.sidebar.radio(
    "Choose a learning module:",
    (
        "Introduction",
        "Data Acquisition & Loading",
        "Exploratory Data Analysis (EDA)",
        # "Feature Engineering", # Future steps
        # "Model Building", # Future steps
    ),
    key="lifecycle_step"
)

# --- Main Content Area ---
st.title(f"Module: {lifecycle_step}")

# Introduction Page
if lifecycle_step == "Introduction":
    st.markdown("""
    ### Welcome to Galileo, your AI-powered Data Science Tutor!
    
    This interactive application is designed to guide you step-by-step through the typical data science workflow.
    
    **How to use this tool:**
    1.  **Navigate Modules:** Use the sidebar on the left to select a module.
    2.  **Upload Data:** In the 'Data Acquisition' module, you can upload your own CSV file. A sample Titanic dataset is also available.
    3.  **Generate & Run Code:** In the 'EDA' module, describe the analysis or plot you want in plain English. The AI will generate the Python code.
    4.  **Experiment:** You can edit the generated code and re-run it to see how your changes affect the output.
    
    Let's get started! Select the **Data Acquisition & Loading** module from the sidebar.
    """)

# Data Loading Page
elif lifecycle_step == "Data Acquisition & Loading":
    st.header("Step 1: Get Your Data")
    st.markdown("""
    Every data science project begins with data. Here, you can either upload your own dataset in CSV format or use a sample dataset to get started quickly.
    """)

    # Option to use a sample dataset
    if st.button("Load Sample Titanic Dataset"):
        # Using a well-known, accessible URL for the Titanic dataset
        df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
        st.session_state.df = df
        st.success("Sample Titanic dataset loaded successfully!")
    
    uploaded_file = st.file_uploader("Or upload your own CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df # Save dataframe to session state
        st.success("Your file was uploaded successfully!")

    if "df" in st.session_state:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head())
        st.info(f"Dataset loaded with **{st.session_state.df.shape[0]}** rows and **{st.session_state.df.shape[1]}** columns.")
        st.markdown("--- \n Great! Now, let's move to **Exploratory Data Analysis (EDA)** using the sidebar.")

# Exploratory Data Analysis (EDA) Page
elif lifecycle_step == "Exploratory Data Analysis (EDA)":
    if "df" not in st.session_state:
        st.warning("Please load a dataset first in the 'Data Acquisition & Loading' module.")
        st.stop()

    df = st.session_state.df
    st.header("Step 2: Explore Your Data")
    st.markdown("""
    Now that we have data, let's explore it! EDA is about understanding your dataset's main characteristics, often with visual methods.
    
    **Describe the plot you want to create below.** For example:
    - *`a histogram of the 'Age' column`*
    - *`a bar chart showing the count of passengers in each 'Pclass'`*
    - *`a scatter plot of 'Age' vs 'Fare', colored by 'Survived'`*
    """)

    plot_request = st.text_input(
        "What would you like to visualize?",
        "a bar chart showing survival count by gender ('Sex' column)"
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("🤖 AI Code Generation")
        if st.button("Generate Code", type="primary"):
            with st.spinner("Galileo is thinking..."):
                prompt = f"""
                You are a data science assistant. Given a pandas DataFrame named 'df',
                write a Python script using seaborn and matplotlib to: {plot_request}.
                The dataframe 'df' has the following columns: {list(df.columns)}.
                
                Important Rules:
                - The dataframe is already loaded in a variable named `df`.
                - Assume `import matplotlib.pyplot as plt` and `import seaborn as sns` are done.
                - Provide ONLY the Python code, without any explanation, comments, or markdown formatting.
                - Use seaborn for plotting if possible. Add a title to the plot.
                """
                generated_code = call_groq(prompt)
                if generated_code:
                    st.session_state.code = generated_code.strip().strip("```python").strip("```")

        # Display the code in an editable text area
        st.session_state.code = st.text_area(
            "You can edit the code here:",
            value=st.session_state.code,
            height=350,
            key="code_editor"
        )
        
        if st.button("▶️ Run Code"):
            with st.spinner("Sending code to secure execution engine..."):
                result = run_code_in_n8n(st.session_state.code, df)
                st.session_state.result = result

    with col2:
        st.subheader("📊 Output")
        result = st.session_state.result
        if not result:
            st.info("The output of your code will be displayed here.")
        
        if "image_data" in result:
            st.image(base64.b64decode(result["image_data"]), caption="Generated Plot")
        
        if "error" in result:
            st.error("An error occurred during execution:")
            st.code(result["error"], language='bash')

            # Add a button for code evaluation
            if st.button("🤔 Ask Tutor for Help"):
                eval_prompt = f"""
                You are a friendly Python tutor. A student is learning data visualization.
                Their goal was to: "{plot_request}".
                
                They wrote the following code:
                --- CODE ---
                {st.session_state.code}
                --- END CODE ---
                
                But it produced this error:
                --- ERROR ---
                {result['error']}
                --- END ERROR ---
                
                Please explain the error in simple, encouraging terms. Then, provide the corrected, complete Python code block.
                Structure your response with "Explanation:" and "Corrected Code:".
                """
                with st.spinner("Analyzing the error..."):
                    explanation = call_groq(eval_prompt, model="llama3-8b-8192")
                    if explanation:
                        st.markdown(explanation)
