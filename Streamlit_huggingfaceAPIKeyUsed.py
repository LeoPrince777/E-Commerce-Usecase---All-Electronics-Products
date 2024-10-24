import streamlit as st
import pandas as pd
import json
import requests

# layout part
st.set_page_config(layout="wide")
st.title("E-Commerce JSON Data Insight Analysis Application using Hugging Face API")  

# Sidebar part
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter Hugging Face API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload your JSON file", type=["json"])

# JSON file load part
if uploaded_file:
    try:
        data = json.load(uploaded_file)
        st.write(f"### Preview of {uploaded_file.name}")
        st.json(data)  # Display the JSON content

        # Convert JSON data to pandas DataFrame part
        if isinstance(data, list):
            df = pd.json_normalize(data)
            st.write("### Data Overview")
            st.dataframe(df)
        else:
            df = pd.DataFrame([data])

        # Prompt part
        st.write("### Ask a question based on the uploaded JSON data:")
        user_question = st.text_input("Enter your question here:", "")

        if user_question and st.button("Get Answer"):
            # Check if API key is provided
            if not api_key:
                st.error("Please provide a valid Hugging Face API key!")
            else:
                # Prepare the context for the model (convert JSON data to string format)
                context = json.dumps(data)

                # Hugging Face API request part
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                # Payload for Hugging Face Inference API part
                payload = {
                    "inputs": {
                        "question": user_question,
                        "context": context
                    }
                }
                
                # Send the request to Hugging Face's Question-Answering Model API
                response = requests.post(
                    "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2",
                    headers=headers,
                    json=payload
                )

                # response from Hugging Face
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "No answer found.")
                    #score = result.get("score", 0)
                    
                    st.write(f"**Answer:** {answer}")
                    #st.write(f"**Confidence Score:** {score:.2f}")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
else:
    st.info("Please upload a JSON file to begin.")
