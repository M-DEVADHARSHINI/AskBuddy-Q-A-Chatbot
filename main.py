import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db
import os
from dotenv import load_dotenv

# Load environment variables (needed if running streamlit directly)
load_dotenv()

st.title("AskBuddy Q&A 🌱")
btn = st.button("Create Knowledgebase")
if btn:
    with st.spinner("Creating knowledgebase..."):
        create_vector_db()
    st.success("Knowledgebase created!")

question = st.text_input("Question: ")

if question:
    try:
        chain = get_qa_chain()
        # The chain must be called with a dictionary where the key matches 'input_key' ('query')
        response = chain({"query": question}) 

        st.header("Answer")
        st.write(response["result"])
    
    except FileNotFoundError:
        st.error("Knowledgebase not found. Please click 'Create Knowledgebase' first.")
    except Exception as e:
        # Catch other errors, like connection issues or invalid API key
        st.error(f"An error occurred during query: {e}")
