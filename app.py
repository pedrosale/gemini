import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import io
import urllib.request

st.title('Detran + Gemini 💬 CTB')
image_url = 'https://raw.githubusercontent.com/pedrosale/falcon_test/af8a20607bae402a893817be0a766ec55a9bcec3/RAG2.jpg'
st.image(image_url, caption='Arquitetura atual: GitHub + Streamlit')
st.markdown('**Esta versão contém:**  \nA) Gemini ⌘ [gemini-pro](https://blog.google/intl/pt-br/novidades/nosso-modelo-de-proxima-geracao-gemini-15/);  \nB) Conjunto de dados pré-carregados do CTB [1. Arquivo de Contexto](https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt) e [2. Reforço de Contexto](https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt);  \nC) ["Retrieval Augmented Generation"](https://python.langchain.com/docs/use_cases/question_answering/) a partir dos dados carregados (em B.).')
    
# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is available
if google_api_key is None:
    st.warning("API key not found. Please set the google_api_key environment variable.")
    st.stop()

# Este bloco deve estar fora do escopo de verificação da chave da API.
context_texts = []
for file_path in ["https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt", 
                  "https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt"]:
    with urllib.request.urlopen(file_path) as response:
        context_texts.append(response.read().decode('utf-8'))

context = "\n\n".join(context_texts)

# Split Texts
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
texts = text_splitter.split_text(context)

# Chroma Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=google_api_key)
vector_index = Chroma.from_texts(texts, embeddings)

# Get User Question
user_question = st.text_input("Ask a Question:")

if st.button("Get Answer") and user_question:
    # Get Relevant Documents
    docs = vector_index.get_relevant_documents(user_question)

    # Define Prompt Template
    prompt_template = """
    Answer the question as detailed as possible from the provided context,
    make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context",
    don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    # Create Prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    # Load QA Chain
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=google_api_key)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    # Get Response
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Display Answer
    st.subheader("Answer:")
    st.write(response['output_text'])
else:
    st.warning("Please enter a question.")
