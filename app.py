# app.py
import streamlit as st
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Constants and API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-pro"
EMBEDDING_MODEL_NAME = "models/embedding-001"
TEMPERATURE = 0.2
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Function to load and split the PDF document
def load_and_split_pdf(uploaded_file):
    """
    Loads a PDF document from an uploaded file and splits it into pages.
    Args:
    uploaded_file (UploadedFile): Streamlit UploadedFile object.
    Returns:
    list: List of pages from the PDF document.
    """
    if uploaded_file is not None:
        with BytesIO(uploaded_file.getbuffer()) as pdf_file:
            pdf_loader = PyPDFLoader(pdf_file)
            return pdf_loader.load_and_split()
    return None

def split_text_into_chunks(pages, chunk_size, chunk_overlap):
    """
    Splits text into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

def setup_gemini_model(model_name, api_key, temperature):
    """
    Sets up the Gemini model for text generation.
    """
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature, convert_system_message_to_human=True)

def create_embeddings_and_index(texts, model_name, api_key):
    """
    Creates embeddings for texts and builds a vector index for retrieval.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
    return vector_index

def create_rag_qa_chain(model, vector_index):
    """
    Creates a Retrieval-Augmented Generation QA chain.
    """
    return RetrievalQA.from_chain_type(model, retriever=vector_index, return_source_documents=True)

# Streamlit App
def main():
    st.title("Financial Report Analysis with RAG")

    uploaded_file = st.file_uploader("Upload Nvidia Financial Report", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner('Processing the document...'):
            pages = load_and_split_pdf(uploaded_file)
            documents = split_text_into_chunks(pages, CHUNK_SIZE, CHUNK_OVERLAP)
            embeddings = create_embeddings_and_index(documents, EMBEDDING_MODEL_NAME, GOOGLE_API_KEY)
            gemini_model = setup_gemini_model(GEMINI_MODEL_NAME, GOOGLE_API_KEY, TEMPERATURE)
            vector_index = create_embeddings_and_index(documents, EMBEDDING_MODEL_NAME, GOOGLE_API_KEY)
            qa_chain = create_rag_qa_chain(gemini_model, vector_index)
            st.success("Document processed successfully!")

    question = st.text_input("Enter your question about Nvidia's financial report:")
    if question and uploaded_file:
        with st.spinner('Generating answer...'):
            try:
                result = qa_chain({"query": question})
                st.write("Answer:", result["result"])
            except Exception as e:
                st.error(f"Error processing the question: {e}")

if __name__ == '__main__':
    main()
