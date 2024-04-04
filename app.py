import streamlit as st
import google.generativeai as genai
from PIL import Image

def config_model(temperature=0.9, top_p=0.1, top_k=50, max_output_tokens=2048, gemini='models/gemini-1.0-pro-latest'):
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_output_tokens,
        }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]
    api_key = ''
    genai.configure(api_key=api_key) #https://aistudio.google.com/app/apikey
    model = genai.GenerativeModel(gemini,
                                generation_config=generation_config,
                                safety_settings=safety_settings
                                )
    return model

def sidebar():
    input = st.sidebar.text_area("Digite o que deseja:")
    uploaded_file = st.sidebar.file_uploader("Choose a Photo to Upload (Opcional)", type=['jpg', 'jpeg', 'png'])
    temperature = st.sidebar.slider('Temperature?', 0, 100, 90)
    top_p = st.sidebar.slider('Top P?', 0, 100, 10)
    top_k = st.sidebar.slider('Top K?', 0, 100, 50)
    max_output_tokens = st.sidebar.number_input("max_output_tokens", value=None, placeholder="Type a integer number...")
    if max_output_tokens == None:
        max_output_tokens = 2048
    st.sidebar.write('The current number is ', max_output_tokens)
    if st.sidebar.button("Enviar"):
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            model = config_model(temperature/100, top_p/100, top_k, max_output_tokens, 'models/gemini-1.0-pro-vision-latest')
            response = model.generate_content([input, image])
            return response.text, model, image
        else:  
            model = config_model(temperature/100, top_p/100, top_k, max_output_tokens)
            response = model.generate_content(input)
            return response.text, model, ''
    return '', None, ''

content, model, image = sidebar()

def body(content, image):
    st.title("Streamlit: Gemini answer")
    st.write("Desenvolvido por Victor Pugliese")
    st.write(content)
    st.write(image)

body(content, image)
