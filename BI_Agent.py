import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Data Visualization Tutor",
    page_icon="📊",
    layout="wide"
)

# --- App Title and Description ---
st.title("📊 AI Data Visualization Tutor")
st.markdown("""
Welcome to the AI Data Viz Tutor! This tool helps you learn how to create visualizations in Python.
1.  **Upload a CSV file** with your data.
2.  **Provide your n8n webhook URL** to connect to the AI agent.
3.  **Write a plain English query** describing the chart you want to create.
4.  The AI will generate the Python code for you.
5.  You can then **run the code, modify it, and see the results instantly!**
""")

# --- Session State Initialization ---
# This is used to store variables across reruns
if 'df' not in st.session_state:
    st.session_state.df = None
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Setup")

    # 1. File Uploader
    uploaded_file = st.file_uploader("1. Upload your CSV file", type=["csv"])

    # 2. n8n Webhook URL
    n8n_webhook_url = st.text_input("2. Enter your n8n Webhook URL", placeholder="https://your-n8n-instance/webhook/...")

    # Process the uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("File uploaded successfully!")
            st.write("Data Preview:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.df = None
    elif st.session_state.df is not None:
        st.write("Using previously uploaded data.")
        st.dataframe(st.session_state.df.head())


# --- Main Application Layout ---
if st.session_state.df is not None:
    st.header("Create Your Visualization")

    # 3. User Query Input
    user_query = st.text_input(
        "3. Describe the chart you want to create",
        placeholder="e.g., 'Create a bar chart of sales by region' or 'Plot a histogram for the age column'"
    )

    if st.button("🤖 Generate Code", use_container_width=True):
        if not n8n_webhook_url:
            st.warning("Please enter your n8n webhook URL in the sidebar.")
        elif not user_query:
            st.warning("Please describe the chart you want to create.")
        else:
            with st.spinner("Asking the AI Tutor for help..."):
                try:
                    # Prepare data to send to n8n
                    # We send column names to give context to the LLM
                    payload = {
                        "query": user_query,
                        "columns": st.session_state.df.columns.tolist()
                    }

                    # Make the POST request to the n8n webhook
                    response = requests.post(n8n_webhook_url, json=payload, timeout=120)

                    if response.status_code == 200:
                        response_data = response.json()
                        st.session_state.generated_code = response_data.get("code", "# AI response was empty.")
                        st.success("AI generated the code successfully!")
                    else:
                        st.error(f"Failed to get response from n8n. Status: {response.status_code}")
                        st.error(f"Response: {response.text}")
                        st.session_state.generated_code = f"# Error: Failed to contact n8n webhook.\n# Status: {response.status_code}"

                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred while contacting n8n: {e}")
                    st.session_state.generated_code = f"# Error: Could not connect to the webhook URL.\n# Please check the URL and your n8n instance."


    # --- Two-Column Layout for Code and Chart ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Code Editor")
        # Use a key for the text_area to allow programmatic updates
        code_input = st.text_area(
            "Modify the code here if needed",
            value=st.session_state.generated_code,
            height=500,
            key="code_editor"
        )

    with col2:
        st.subheader("Chart Output")
        # Placeholder for the chart
        chart_display = st.empty()
        chart_display.info("The generated chart will appear here after you run the code.")

    if st.button("▶️ Run Code", use_container_width=True):
        if not code_input.strip():
            st.warning("Code editor is empty. Please generate or write some code first.")
        else:
            with st.spinner("Executing code..."):
                try:
                    # Use a fresh figure for each run
                    plt.close('all')
                    fig, ax = plt.subplots()

                    # Create a dictionary for the exec environment
                    # Pass the dataframe 'df' and matplotlib 'plt' to the executed code
                    exec_globals = {
                        'df': st.session_state.df,
                        'plt': plt
                    }

                    # Execute the user's code
                    exec(code_input, exec_globals)

                    # Capture the plot from matplotlib into a buffer
                    buf = io.BytesIO()
                    # Check if a plot was created before saving
                    if plt.get_fignums():
                        plt.savefig(buf, format="png", bbox_inches='tight')
                        buf.seek(0)
                        chart_display.image(buf, caption="Generated Chart")
                    else:
                        chart_display.warning("The code ran, but it didn't generate a plot with Matplotlib.")

                except Exception as e:
                    st.error(f"An error occurred while executing the code:")
                    st.exception(e)
                    chart_display.empty() # Clear previous image on error

    st.warning("""
        **Security Warning:** The 'Run Code' button executes Python code entered in the text area.
        Only run code that you trust. Malicious code could potentially harm your environment.
    """, icon="⚠️")

else:
    st.info("Please upload a CSV file in the sidebar to get started.")
