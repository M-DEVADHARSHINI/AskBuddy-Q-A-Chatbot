import os
import platform
import sys
from dotenv import load_dotenv

# --- CRITICAL FIX FOR WINDOWS 'pwd' ModuleNotFoundError ---
if platform.system() == "Windows":
    sys.modules['pwd'] = object()

# Now perform all other imports
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


load_dotenv()

# --- Model Configuration ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
instructor_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectordb_file_path = "faiss_index"

def create_vector_db():
    """
    Loads data from the CSV, creates a vector database using FAISS and Google Embeddings,
    and saves it locally.
    """
    print("Loading data...")
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
    data = loader.load()
    print(f"Loaded {len(data)} documents.")

    print("Creating vector database...")
    vectordb = FAISS.from_documents(
        documents=data,
        embedding=instructor_embeddings
    )
    vectordb.save_local(vectordb_file_path)
    print(f"Vector database saved successfully to: {vectordb_file_path}")


def get_qa_chain():
    """
    Loads the vector database and creates a RetrievalQA chain for answering questions.
    """
    if not os.path.exists(vectordb_file_path):
        raise FileNotFoundError(f"Vector DB not found at {vectordb_file_path}. Please click 'Create Knowledgebase' first.")

    # Fix for DeserializationError: Added allow_dangerous_deserialization=True
    vectordb = FAISS.load_local(
        vectordb_file_path, 
        instructor_embeddings,
        allow_dangerous_deserialization=True 
    )

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={'score_threshold': 0.5, 'k': 3} 
    )

    # =========================================================================
    # *** CORE FIX HERE: Modified Prompt Template for Conversational Output ***
    # =========================================================================
    prompt_template = """You are a helpful and friendly assistant. Your goal is to answer the user's question using the provided context, while maintaining a smooth, natural, and conversational tone, like a human interacting with a customer.

    Instructions:
    1. **Prioritize Conversational Flow**: Use contractions (e.g., "we're" instead of "we are"), gentle transitions (e.g., "Actually," "I checked," "That's a great question"), and avoid overly rigid phrasing like "Based on the context..."
    2. **Integrate Facts Smoothly**: If you retrieve multiple facts (like 'we have internships' and 'no EMI info'), weave them into one or two seamless sentences instead of listing them separately.
    3. **Address Missing Information Naturally**: If a part of the question cannot be answered by the context, acknowledge it politely and smoothly. **Do not use the exact phrase "I cannot find any information."** Instead, try phrases like "I don't have that detail right here," or "I'm not seeing any info on that."
    4. **Generate a Best-Effort Answer**: If the context is relevant but doesn't directly answer the question, use your general knowledge to generate a helpful and relevant answer based on the retrieved information.

    CONTEXT: {context}

    QUESTION: {question}"""
    # =========================================================================

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query", 
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain

if __name__ == "__main__":
    # Example usage for testing outside of Streamlit
    # create_vector_db() 
    # chain = get_qa_chain()
    # print(chain.invoke({"query": "Do you have javascript course?"}))
    pass