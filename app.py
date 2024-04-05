# app.py
import streamlit as st
from io import BytesIO
from langchain_community.document_loaders import TextLoader  # Ajuste para importação correta
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import tempfile
import urllib.request

load_dotenv()

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
# Streamlit App
def main():
    st.title('Detran + Gemini 💬 CTB')
    image_url = 'https://raw.githubusercontent.com/pedrosale/falcon_test/af8a20607bae402a893817be0a766ec55a9bcec3/RAG2.jpg'
    st.image(image_url, caption='Arquitetura atual: GitHub + Streamlit')
    st.markdown('**Esta versão contém:**  \nA) Gemini ⌘ [gemini-pro](https://blog.google/intl/pt-br/novidades/nosso-modelo-de-proxima-geracao-gemini-15/);  \nB) Conjunto de dados pré-carregados do CTB.')

    # Carrega os textos diretamente dos arquivos
    context_texts = []
    for file_path in ["https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt", 
                      "https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt"]:
        with urllib.request.urlopen(file_path) as response:
            context_texts.append(response.read().decode('utf-8'))
    
    context = "\n\n".join(context_texts)
    
    # Split Texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_text(context)
    
    # Process and index the documents
    embeddings = create_embeddings_and_index(texts, EMBEDDING_MODEL_NAME, GOOGLE_API_KEY)
    gemini_model = setup_gemini_model(GEMINI_MODEL_NAME, GOOGLE_API_KEY, TEMPERATURE)
    qa_chain = create_rag_qa_chain(gemini_model, embeddings)

    question = st.text_input("Enter your question:")
    if question:
        with st.spinner('Generating answer...'):
            try:
                result = qa_chain({"query": question})
                st.write("Answer:", result["result"])
            except Exception as e:
                st.error(f"Error processing the question: {e}")

if __name__ == '__main__':
    main()
