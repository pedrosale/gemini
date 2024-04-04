# Import necessary libraries
import streamlit as st
from streamlit_chat import message
import google.generativeai as genai
from dotenv import load_dotenv
import os
import tempfile
import urllib.request
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Get the Google API key from the environment variables
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Google Generative AI with the API key
genai.configure(api_key=api_key)

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Como posso te ajudar?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Olá, sou seu assistente."]

def conversation_chat(query, history):
    prompt = """
    Você é um assistente que só conversa no idioma português do Brasil (você nunca, jamais conversa em outro idioma que não seja o português do Brasil).
    Você responde as perguntas do usuário com base nos arquivos carregados.
    Vamos pensar passo a passo para responder.
    """
    query_with_prompt = prompt + query
    model = genai.GenerativeModel("gemini-pro", generation_config={"temperature": 0.7, "max_output_tokens": 512})
    result = model.generate_content(query_with_prompt)
    history.append((query, result.text))
    return result.text

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("question:", placeholder="Me pergunte sobre o(s) conjunto(s) de dados pré-carregados", key='input')
            submit_button = st.form_submit_button(label='Enviar')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                logo_url = 'https://your_logo_url_here.png'
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', logo=logo_url)
                message(st.session_state["generated"][i], key=str(i), logo=logo_url)

def main():
    # Initialize session state
    initialize_session_state()
    st.title('🦅💬 Chatbot Assistente com Gemini.')
    
    # Display the chat history and the input form for new questions
    display_chat_history()

if __name__ == "__main__":
    main()
