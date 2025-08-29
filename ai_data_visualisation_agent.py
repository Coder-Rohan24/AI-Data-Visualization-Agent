import os
import json
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
from PIL import Image
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
# from together import Together  # Removed Together
from e2b_code_interpreter import Sandbox
import openai
import cohere

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    with st.spinner('Executing code in E2B sandbox...'):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec = e2b_code_interpreter.run_code(code)

        if stderr_capture.getvalue():
            print("[Code Interpreter Warnings/Errors]", file=sys.stderr)
            print(stderr_capture.getvalue(), file=sys.stderr)

        if stdout_capture.getvalue():
            print("[Code Interpreter Output]", file=sys.stdout)
            print(stdout_capture.getvalue(), file=sys.stdout)

        if exec.error:
            print(f"[Code Interpreter ERROR] {exec.error}", file=sys.stderr)
            return None
        return exec.results

def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        return code
    return ""

def chat_with_llm(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    # Update system prompt to include dataset path information
    system_prompt = f"""You're a Python data scientist and data visualization expert. You are given a dataset at path '{dataset_path}' and also the user's query.
You need to analyze the dataset and answer the user's query with a response and you run Python code to solve them.
IMPORTANT: Always use the dataset path variable '{dataset_path}' in your code when reading the CSV file."""

    provider = st.session_state.llm_provider
    model_name = st.session_state.model_name
    api_key = st.session_state.llm_api_key

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    with st.spinner(f'Getting response from {provider} LLM model...'):
        if provider == "OpenAI":
            openai.api_key = api_key
            response = openai.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            response_message = response.choices[0].message
            python_code = match_code_blocks(response_message.content)
            llm_content = response_message.content
        elif provider == "Cohere":
            co = cohere.Client(api_key)
            prompt = system_prompt + "\nUser: " + user_message
            response = co.chat(
                model=model_name,
                message=prompt,
            )
            llm_content = response.text
            python_code = match_code_blocks(llm_content)
        else:
            st.error("Unsupported LLM provider selected.")
            return None, ""

        if python_code:
            code_interpreter_results = code_interpret(e2b_code_interpreter, python_code)
            return code_interpreter_results, llm_content
        else:
            st.warning(f"Failed to match any Python code in model's response")
            return None, llm_content

def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"
    
    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise error


def main():
    """Main Streamlit application."""
    st.title("📊 AI Data Visualization Agent")
    st.write("Upload your dataset and ask questions about it!")

    # Initialize session state variables
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = 'OpenAI'
    if 'llm_api_key' not in st.session_state:
        st.session_state.llm_api_key = ''
    if 'e2b_api_key' not in st.session_state:
        st.session_state.e2b_api_key = ''
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''

    with st.sidebar:
        st.header("API Keys and Model Configuration")
        st.session_state.llm_provider = st.selectbox("Select LLM Provider", ["OpenAI", "Cohere"])
        if st.session_state.llm_provider == "OpenAI":
            st.session_state.llm_api_key = st.text_input("OpenAI API Key", type="password")
            st.sidebar.markdown("[Get OpenAI API Key](https://platform.openai.com/signup)")
            model_options = {
                "GPT-3.5 Turbo": "gpt-3.5-turbo",
                "GPT-4": "gpt-4",
            }
        elif st.session_state.llm_provider == "Cohere":
            st.session_state.llm_api_key = st.text_input("Cohere API Key", type="password")
            st.sidebar.markdown("[Get Cohere API Key](https://dashboard.cohere.com/welcome)")
            model_options = {
                "Command R": "command-r",
                "Command R Plus": "command-r-plus",
            }
        st.session_state.model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0
        )
        st.session_state.model_name = model_options[st.session_state.model_name]

        st.session_state.e2b_api_key = st.text_input("Enter E2B API Key", type="password")
        st.sidebar.markdown("[Get E2B API Key](https://e2b.dev/docs/legacy/getting-started/api-key)")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Display dataset with toggle
        df = pd.read_csv(uploaded_file)
        st.write("Dataset:")
        show_full = st.checkbox("Show full dataset")
        if show_full:
            st.dataframe(df)
        else:
            st.write("Preview (first 5 rows):")
            st.dataframe(df.head())
        # Query input
        query = st.text_area("What would you like to know about your data?",
                            "Can you compare the average cost for two people between different categories?")

        if st.button("Analyze"):
            if not st.session_state.llm_api_key or not st.session_state.e2b_api_key:
                st.error("Please enter both API keys in the sidebar.")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    # Upload the dataset
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)

                    # Pass dataset_path to chat_with_llm
                    code_results, llm_response = chat_with_llm(code_interpreter, query, dataset_path)

                    # Display LLM's text response
                    st.write("AI Response:")
                    st.write(llm_response)

                    # Display results/visualizations
                    if code_results:
                        for result in code_results:
                            if hasattr(result, 'png') and result.png:  # Check if PNG data is available
                                # Decode the base64-encoded PNG data
                                png_data = base64.b64decode(result.png)

                                # Convert PNG data to an image and display it
                                image = Image.open(BytesIO(png_data))
                                st.image(image, caption="Generated Visualization", use_container_width=False)
                            elif hasattr(result, 'figure'):  # For matplotlib figures
                                fig = result.figure  # Extract the matplotlib figure
                                st.pyplot(fig)  # Display using st.pyplot
                            elif hasattr(result, 'show'):  # For plotly figures
                                st.plotly_chart(result)
                            elif isinstance(result, (pd.DataFrame, pd.Series)):
                                st.dataframe(result)
                            else:
                                st.write(result)

if __name__ == "__main__":
    main()
