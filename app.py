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

load_dotenv()

# Constants and API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-pro"
EMBEDDING_MODEL_NAME = "models/embedding-001"
TEMPERATURE = 0.75
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500

def load_and_process_context(EMBEDDING_MODEL_NAME, GOOGLE_API_KEY):
    context_texts = []
    processed_texts = {}  # Dicionário para mapear IDs de documentos para textos
    doc_id = 0  # Inicializa um contador para servir como ID do documento
    
    for file_path in ["https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt", 
                      "https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt"]:
        with urllib.request.urlopen(file_path) as response:
            text = response.read().decode('utf-8')
            context_texts.append(text)
            processed_texts[doc_id] = text  # Armazena o texto com seu ID correspondente
            doc_id += 1  # Incrementa o ID para o próximo documento
    
    # Concatena todos os textos para criar um grande contexto (se necessário)
    context = "\n\n".join(context_texts)
    
    # Split Texts - Isso pode ser ajustado dependendo de como você quer processar os textos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_text(context)
    
    # Process and index the documents
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
    
    # Retorna tanto o índice de vetor quanto os textos processados
    return vector_index, processed_texts


def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'vector_index' not in st.session_state:
        st.session_state['vector_index'], st.session_state['processed_texts'] = load_and_process_context(EMBEDDING_MODEL_NAME,GOOGLE_API_KEY)


def conversation_with_gemini(query, model, chat_history, vector_index, processed_texts):
    """
    Updates conversation with Gemini considering the document context and chat history.
    """
    # Use vector index to find relevant document snippets based on the current query
    relevant_docs = vector_index.search(query, k=3)  # Adjust k as necessary
    relevant_texts = " ".join([processed_texts[doc_id] for doc_id, _ in relevant_docs])
    
    # Prepare query with history and relevant document context
    history_context = "\n".join([f"Q: {pair['question']}\nA: {pair['answer']}" for pair in chat_history])
    full_query = f"{history_context}\n\n{relevant_texts}\nQ: {query}\nA:"
    
    qa_chain  = RetrievalQA.from_chain_type(model, retriever=vector_index, return_source_documents=True)

    response = qa_chain({"query": full_query})
    chat_history.append({"question": query, "answer": response["result"]})
    return response["result"]

# Streamlit App
def main():
    st.title('Detran + Gemini 💬 CTB')
    image_url = 'https://raw.githubusercontent.com/pedrosale/falcon_test/af8a20607bae402a893817be0a766ec55a9bcec3/RAG2.jpg'
    st.image(image_url, caption='Arquitetura atual: GitHub + Streamlit')
    st.markdown('**Esta versão contém:**  \nA) Gemini ⌘ [gemini-pro](https://blog.google/intl/pt-br/novidades/nosso-modelo-de-proxima-geracao-gemini-15/);   \nB) Conjunto de dados pré-carregados do CTB [1. Arquivo de Contexto](https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt) e [2. Reforço de Contexto](https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt);  \nC) ["Retrieval Augmented Generation"](https://python.langchain.com/docs/use_cases/question_answering/) a partir dos dados carregados (em B.) com Langchain.')
    initialize_session_state()
    gemini_model = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=TEMPERATURE, convert_system_message_to_human=True)

    user_query = st.text_input("What's your question?", key="user_query")
    if st.button("Ask"):
        if user_query:
            with st.spinner('Thinking...'):
                answer = conversation_with_gemini(user_query, gemini_model, st.session_state['chat_history'], st.session_state['vector_index'], st.session_state['processed_texts'])
                st.session_state['chat_history'].append({"question": user_query, "answer": answer})
                for interaction in st.session_state['chat_history']:
                    st.text(interaction["question"])
                    st.text_area("", value=interaction["answer"], height=100)

if __name__ == '__main__':
    main()
