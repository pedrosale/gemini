import streamlit as st
from streamlit_chat import message
import google.generativeai as genai
from dotenv import load_dotenv
import os
import tempfile
import urllib.request

# Carrega as variáveis de ambiente
load_dotenv()

# Configura a API key do Google Gemini
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Como posso te ajudar?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Olá, sou seu assistente."]

# Função para carregar e processar dados de URLs específicas
def load_and_process_data(file_urls):
    all_text = []
    for file_url in file_urls:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(urllib.request.urlopen(file_url).read())
            temp_file_path = temp_file.name
        
        with open(temp_file_path, 'r', encoding='utf-8') as file:
            all_text.extend(file.readlines())
        os.remove(temp_file_path)

    # Simplificação: junta os textos e retorna
    return " ".join(all_text)[:500]  # Limita a 500 caracteres para simplificar

def conversation_chat(query, history, processed_data):
    prompt = f"""
    Você é um assistente que só conversa no idioma português do Brasil. Baseado nas informações que temos: {processed_data}.
    Pergunta do usuário: {query}
    """
    model = genai.GenerativeModel("gemini-pro", generation_config={"temperature": 0.7, "max_output_tokens": 512})
    result = model.generate_content(prompt)
    history.append((query, result.text))
    return result.text

def display_chat_history(processed_data):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Pergunta:", placeholder="Me pergunte algo", key='input')
            submit_button = st.form_submit_button(label='Enviar')

        if submit_button and user_input:
            with st.spinner('Gerando resposta...'):
                output = conversation_chat(user_input, st.session_state['history'], processed_data)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                logo_url = 'https://raw.githubusercontent.com/pedrosale/falcon_test/a7248c8951827efd997b927d7a4d4c4c200c1996/logo_det3.png'  # Substitua pelo seu logo
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', logo=logo_url)
                message(st.session_state["generated"][i], key=str(i), logo=logo_url)

def main():
    initialize_session_state()
    st.title('Chatbot Assistente com Gemini')
    # URL direta para a imagem hospedada no GitHub
    st.markdown('**Esta versão contém:**  \nA) Gemini ⌘ [gemini-pro](https://blog.google/intl/pt-br/novidades/nosso-modelo-de-proxima-geracao-gemini-15/);  \nB) Conjunto de dados pré-carregados do CTB [1. Arquivo de Contexto](https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt) e [2. Reforço de Contexto](https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt);  \nC) ["Retrieval Augmented Generation"](https://python.langchain.com/docs/use_cases/question_answering/) a partir dos dados carregados (em B.) com Langchain.')
    # Carrega o arquivo diretamente (substitua o caminho do arquivo conforme necessário)

    # URLs dos arquivos de texto para carregar
    file_urls = [
        "https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt",
        "https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt"
    ]
    processed_data = load_and_process_data(file_urls)

    display_chat_history(processed_data)

if __name__ == "__main__":
    main()
