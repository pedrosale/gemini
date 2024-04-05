__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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

# Carrega variáveis de ambiente
load_dotenv()

# Constantes e Chaves de API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-pro"
EMBEDDING_MODEL_NAME = "models/embedding-001"
TEMPERATURE = 0.75
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500

def load_and_process_context():
    # URLs dos textos de contexto
    file_paths = [
        "https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt",
        "https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt"
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
    st.title('Detran + Gemini 💬 CTB')
    image_url = 'https://raw.githubusercontent.com/pedrosale/falcon_test/af8a20607bae402a893817be0a766ec55a9bcec3/RAG2.jpg'
    st.image(image_url, caption='Arquitetura atual: GitHub + Streamlit')
    st.markdown('**Esta versão contém:**  \nA) Gemini ⌘ [gemini-pro](https://blog.google/intl/pt-br/novidades/nosso-modelo-de-proxima-geracao-gemini-15/);   \nB) Conjunto de dados pré-carregados do CTB [1. Arquivo de Contexto](https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt) e [2. Reforço de Contexto](https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt);  \nC) ["Retrieval Augmented Generation"](https://python.langchain.com/docs/use_cases/question_answering/) a partir dos dados carregados (em B.) com Langchain.')

     
    # Carrega e processa o contexto
    vector_index = load_and_process_context()
    
    # Configura o modelo Gemini
    gemini_model = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=TEMPERATURE, convert_system_message_to_human=True)
    
    # Cria a cadeia de RAG
    qa_chain = RetrievalQA.from_chain_type(gemini_model, retriever=vector_index, return_source_documents=True)
    
    # Inicializa o histórico de conversas
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    question = st.text_input("Qual é a sua pergunta?", key="user_query")
    
    if st.button("Perguntar"):
        if question:
            with st.spinner('Pensando...'):
                history_context = "\n".join([f"Q: {pair['question']}\nA: {pair['answer']}" for pair in st.session_state['chat_history']])
                full_query = f"{history_context}\nQ: {question}"
                
                result = qa_chain({"query": full_query})
                answer = result.get("result", "Não foi possível gerar uma resposta.")
                
                st.session_state['chat_history'].append({"question": question, "answer": answer})
                
                for interaction in st.session_state['chat_history']:
                    st.text(f"P: {interaction['question']}")
                    st.text_area("", value=f"R: {interaction['answer']}", height=100)
                    st.write("---")

if __name__ == '__main__':
    main()
