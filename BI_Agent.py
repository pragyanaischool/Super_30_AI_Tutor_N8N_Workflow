import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="GROQ & LangChain AI Data Science Tutor",
    page_icon="🧪",
    layout="wide"
)

# --- App Title and Description ---
st.title("🧪 GROQ & LangChain AI Data Science Tutor")
st.markdown("""
Welcome to your advanced AI Data Science assistant! This tool uses the speed of GROQ and the power of LangChain to guide you through the data science process.
1.  **Upload a CSV** file to begin.
2.  **Provide your n8n webhook URL** to connect to the LangChain agent.
3.  **Select a Data Science step** you want to work on.
4.  **Describe your goal in plain English** (e.g., 'Clean the missing values in the age column').
5.  The AI will **Build Code**. You can then **Debug Code**, ask for an **Explanation**, and **Run Code** to see the results.
""")

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""
if 'explanation' not in st.session_state:
    st.session_state.explanation = ""

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("1. Upload your CSV file", type=["csv"])
    n8n_webhook_url = st.text_input("2. Enter your n8n Webhook URL", placeholder="https://your-n8n-instance/webhook/...")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("File uploaded successfully!")
            with st.expander("Data Preview", expanded=False):
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.df = None
    elif st.session_state.df is not None:
        st.write("Using previously uploaded data.")

# --- Main Application Layout ---
if st.session_state.df is not None:
    st.header("Data Science Workflow")

    # 3. Select Data Science Process Step
    ds_process_step = st.selectbox(
        "3. Select the Data Science process step",
        [
            "Data Loading & Inspection",
            "Data Cleaning",
            "Exploratory Data Analysis (Visualization)",
            "Feature Engineering",
            "Data Analysis & Interpretation"
        ]
    )

    # 4. User Query Input
    user_query = st.text_area(
        "4. Describe what you want to achieve in this step",
        placeholder="e.g., 'Plot a histogram of the age column to see the distribution' or 'Remove duplicate rows and fill missing values in the sales column with the mean'"
    )

    # --- API Call Function ---
    def call_n8n_agent(task, code_to_process=None):
        if not n8n_webhook_url:
            st.warning("Please enter your n8n webhook URL in the sidebar.")
            return None
        if not user_query and task == "generate":
            st.warning("Please describe what you want to achieve.")
            return None

        with st.spinner(f"Agent is working on: {task}..."):
            try:
                payload = {
                    "task": task,
                    "query": user_query,
                    "columns": st.session_state.df.columns.tolist(),
                    "step": ds_process_step,
                    "code": code_to_process,
                    "df_head": st.session_state.df.head().to_json(orient='split')
                }
                response = requests.post(n8n_webhook_url, json=payload, timeout=180)

                if response.status_code == 200:
                    return response.json()
                else:
                    st.error(f"Failed to get response from n8n. Status: {response.status_code}")
                    st.error(f"Response: {response.text}")
                    return None
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred while contacting n8n: {e}")
                return None

    # --- Action Buttons ---
    buttons_col = st.columns(3)
    if buttons_col[0].button("🤖 Build Code", use_container_width=True, type="primary"):
        response_data = call_n8n_agent("generate")
        if response_data and "code" in response_data:
            st.session_state.generated_code = response_data["code"]
            st.session_state.explanation = response_data.get("explanation", "")
            st.success("AI generated the code!")

    if buttons_col[1].button("🐞 Debug Code", use_container_width=True):
        current_code = st.session_state.get('generated_code', '')
        response_data = call_n8n_agent("debug", current_code)
        if response_data and "code" in response_data:
            st.session_state.generated_code = response_data["code"]
            st.session_state.explanation = response_data.get("explanation", "")
            st.success("AI has debugged the code!")

    if buttons_col[2].button("🧑‍🏫 Explain Code", use_container_width=True):
        current_code = st.session_state.get('generated_code', '')
        response_data = call_n8n_agent("explain", current_code)
        if response_data and "explanation" in response_data:
            st.session_state.explanation = response_data["explanation"]
            st.success("AI has explained the code!")

    # --- Code Editor and Explanation ---
    editor_col, output_col = st.columns(2)

    with editor_col:
        st.subheader("Code Editor")
        st.session_state.generated_code = st.text_area(
            "Modify the code here if needed",
            value=st.session_state.generated_code,
            height=500,
            key="code_editor"
        )
        if st.session_state.explanation:
            with st.expander("Code Explanation", expanded=True):
                st.markdown(st.session_state.explanation)

    with output_col:
        st.subheader("Output")
        output_display = st.empty()
        output_display.info("Your chart or analysis output will appear here after you run the code.")

    if st.button("▶️ Run Code", use_container_width=True):
        code_to_run = st.session_state.generated_code
        if not code_to_run.strip():
            st.warning("Code editor is empty.")
        else:
            with st.spinner("Executing code..."):
                try:
                    # Redirect stdout to capture print statements
                    old_stdout = st.io.sys.stdout
                    redirected_output = st.io.StringIO()
                    st.io.sys.stdout = redirected_output

                    # Use a fresh figure for each run
                    plt.close('all')
                    
                    # Make a copy of the dataframe to avoid modification issues
                    df_copy = st.session_state.df.copy()

                    exec_globals = {'df': df_copy, 'plt': plt}
                    exec(code_to_run, exec_globals)
                    
                    # Restore stdout
                    st.io.sys.stdout = old_stdout
                    
                    # Check for plots
                    if plt.get_fignums():
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png", bbox_inches='tight')
                        buf.seek(0)
                        output_display.image(buf, caption="Generated Chart")
                    
                    # Check for printed output
                    printed_output = redirected_output.getvalue()
                    if printed_output:
                        if plt.get_fignums(): # If there's also a plot, put text below
                             st.code(printed_output)
                        else: # If only text, put it in the main output area
                            output_display.code(printed_output)
                    
                    if not plt.get_fignums() and not printed_output:
                        output_display.success("Code executed successfully with no visual output.")

                except Exception as e:
                    st.io.sys.stdout = old_stdout
                    st.error(f"An error occurred while executing the code:")
                    st.exception(e)
                    output_display.empty()

    st.warning("""
        **Security Warning:** This app executes Python code. Only run code you trust.
    """, icon="⚠️")
else:
    st.info("Please upload a CSV file in the sidebar to get started.")
