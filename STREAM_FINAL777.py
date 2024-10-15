import streamlit as st
import pandas as pd
from transformers import pipeline
import json

#layout creation part
st.set_page_config(layout="wide")
st.title("E-Commerce JSON Data Insight Analysis")

# Sidebar creation part
st.sidebar.header("Upload JSON File")
uploaded_file = st.sidebar.file_uploader("Upload your JSON file", type=["json"])

if uploaded_file is not None:
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
        
        # Hugging Face QA model assign part
        model = pipeline('question-answering', model="deepset/roberta-base-squad2")

        # Prompt part
        st.write("### Ask a question based on the uploaded JSON data:")
        user_question = st.text_input("Enter your question here:", "")
        
        if user_question and st.button("Get Answer"):
        
            context = json.dumps(data)  # Convert JSON to string

            # Hugging Face model Query part
            result = model(question=user_question, context=context)
            
            # print result part
            st.write(f"**Answer:** {result['answer']}")
            #st.write(f"**Confidence Score:** {result['score']:.2f}")

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")

else:
    st.info("Please upload a JSON file to proceed.")
