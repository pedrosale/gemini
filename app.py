__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from streamlit_chat import message
import os
from dotenv import load_dotenv
import tempfile
import urllib.request

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if google_api_key is None:
    st.warning("API key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Como posso te ajudar?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Olá, sou seu assistente."]

def conversation_chat(query, chain, history):
    prompt = """
    Você é um assistente que só conversa no idioma português do Brasil (você nunca, jamais conversa em outro idioma que não seja o português do Brasil).
    Você responde as perguntas do usuário com base nos arquivos carregados.
    Vamos pensar passo a passo para responder.
    """
    result = chain({"question": query, "chat_history": history}, return_only_outputs=True)
    history.append((query, result["output_text"]))
    return result["output_text"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Pergunta:", placeholder="Me pergunte sobre o(s) conjunto(s) de dados pré-carregados", key='input')
            submit_button = st.form_submit_button(label='Enviar')

        if submit_button and user_input:
            with st.spinner('Gerando resposta...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                logo_url = 'https://your-logo-url-here.png'  # Substitua pela URL do seu logotipo
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', logo=logo_url)
                message(st.session_state["generated"][i], key=str(i), logo=logo_url)

def create_conversational_chain():
    load_dotenv()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=google_api_key)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)

    # Carrega e prepara os arquivos de texto como feito originalmente
    file_paths = ["https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt", "https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt"]
    texts = []
    for file_path in file_paths:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(urllib.request.urlopen(file_path).read())
            temp_file_path = temp_file.name

        with open(temp_file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())

        os.remove(temp_file_path)

    texts = text_splitter.split_text("\n\n".join(texts))

    vector_store = Chroma.from_texts(texts, embeddings).as_retriever()

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=google_api_key)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=model, chain_type='stuff',
                                                  retriever=vector_store.as_retriever(),
                                                  memory=memory)
    return chain

def main():
    initialize_session_state()
    st.title('Chatbot Assistente com Gemini')
    # URL direta para a imagem hospedada no GitHub
    image_url = 'https://raw.githubusercontent.com/pedrosale/falcon_test/af8a20607bae402a893817be0a766ec55a9bcec3/RAG2.jpg'
    # Exibir a imagem usando a URL direta
    st.image(image_url, caption='Arquitetura atual: GitHub + Streamlit')
    st.markdown('**Esta versão contém:**  \nA) Gemini ⌘ [gemini-pro](https://blog.google/intl/pt-br/novidades/nosso-modelo-de-proxima-geracao-gemini-15/);  \nB) Conjunto de dados pré-carregados do CTB [1. Arquivo de Contexto](https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt) e [2. Reforço de Contexto](https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt);  \nC) ["Retrieval Augmented Generation"](https://python.langchain.com/docs/use_cases/question_answering/) a partir dos dados carregados (em B.).')
    
    chain = create_conversational_chain()

    display_chat_history(chain)

if __name__ == "__main__":
    main()
