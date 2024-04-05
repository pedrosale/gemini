__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from io import BytesIO
import os
from dotenv import load_dotenv
import urllib.request
import tempfile
from langchain.llms import GoogleGenerativeAI
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Carrega variáveis de ambiente
load_dotenv()

# Constantes e Chaves de API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-pro"
EMBEDDING_MODEL_NAME = "embedding-001"  # Atualize conforme necessário
TEMPERATURE = 0.75
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500

def load_and_process_context():
    # URLs dos textos de contexto
    file_paths = [
        "https://raw.githubusercontent.com/your_repo/main/context_file1.txt",
        "https://raw.githubusercontent.com/your_repo/main/context_file2.txt"
    ]

    context_texts = []
    for file_path in file_paths:
        with urllib.request.urlopen(file_path) as response:
            context_texts.append(response.read().decode('utf-8'))

    # Combina todos os textos em um único grande contexto
    combined_context = "\n\n".join(context_texts)

    # Divide os textos para processamento
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_text(combined_context)

    # Processa e indexa os documentos
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
    
    return vector_index

def main():
    st.title('Seu Chatbot com Gemini e RAG')
    image_url = 'https://yourimageurl.com/image.png'
    st.image(image_url, caption='Chatbot usando Gemini e RAG')
    st.markdown('**Este chatbot utiliza:**\n- Modelo Gemini para geração de respostas;\n- Um sistema de Retrieval Augmented Generation (RAG) para enriquecer as respostas com base em documentos carregados.')

    vector_index = load_and_process_context()
    
    # Configura o modelo Gemini
    gemini_model = setup_gemini_model(GEMINI_MODEL_NAME, GOOGLE_API_KEY, TEMPERATURE)
    
    # Cria a cadeia de RAG
    qa_chain = RetrievalQA(generative_model=gemini_model, retriever=vector_index, return_source_documents=False)
    
    # Inicializa o histórico de conversas
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    question = st.text_input("Qual é a sua pergunta?", key="user_query")
    
    if st.button("Perguntar"):
        if question:
            with st.spinner('Pensando...'):
                history_context = "\n".join([f"Q: {pair['question']}\nA: {pair['answer']}" for pair in st.session_state['chat_history']])
                full_query = f"{history_context}\nQ: {question}"
                
                result = qa_chain.ask(full_query)
                answer = result.get("answer", "Não foi possível gerar uma resposta.")
                
                st.session_state['chat_history'].append({"question": question, "answer": answer})
                
                for interaction in st.session_state['chat_history']:
                    st.text(f"P: {interaction['question']}")
                    st.text_area("", value=f"R: {interaction['answer']}", height=100)
                    st.write("---")

if __name__ == '__main__':
    main()
