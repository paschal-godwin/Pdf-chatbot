#!/usr/bin/env python
# coding: utf-8


import os
import streamlit as st
import fitz
from dotenv import load_dotenv
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from IPython.display import Markdown, display, update_display


load_dotenv(override = True)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI()


pdf_folder = r"C:\Users\User\Documents\Projects\llm_engineering\PDF-chatbot\books"


uploaded_files = st.file_uploader("upload PDF files", type="pdf", accept_multiple_files=True)


def extract_text():
    all_text = []
    for filename in os.listdir(r"C:\Users\User\Documents\Projects\llm_engineering\PDF-chatbot\books"):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            doc = fitz.open(pdf_path)
            text = ''
            for page in doc:
                content = page.get_text('text')
                text += content
            all_text.append(text)
    if uploaded_files:
        for file in uploaded_files:
            text = ""
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text("text")
            all_text.append(text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size =500, chunk_overlap=100)
    docs = text_splitter.create_documents(all_text)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(docs, embeddings)   
    return vector_store


vector_store = extract_text()
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')


def bot(query):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    llms = ChatOpenAI(model = 'gpt-3.5-turbo')
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llms, retriever =vector_store.as_retriever(), memory=memory, return_source_documents=True)
    talk = qa_chain.invoke(query)
    response = talk['answer']
    sources = talk.get("source_documents",[])
    st.session_state.chat_history.append((query,response))
    total = [response,sources]
    return total

def system_prompt_func(response):
    system_prompt = 'you are an AI assitant who is very  knowlegeable and smart \n'
    system_prompt += f'You are to compare the user input and the response. this is the response:  {response}\n'
    system_prompt += "if the response does not fit the user's needs, find a better answer. If the response is good, respond with just the same answer"
    return system_prompt


def user_prompt_func(query,response):
    user_prompt = "compare the query and the response and confirm if the response is suitable. \n"
    user_prompt = 'If it is not suitable provide a better answer'
    user_prompt+= f'query : {query}, response: {response}'
    return user_prompt


st.title("Multi-PDF Chatbot for Exam Prep")
st.markdown("Ask anything from the loaded PDFs or infact, anything else...")

query = st.text_input('what do you need help with?')



def assistant():
    response,sources = bot(query)
    chat_bot = openai.chat.completions.create(
         model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": system_prompt_func(response)},
            {"role": "user", "content": user_prompt_func(query,response)}
      ],
    )
    result = chat_bot.choices[0].message.content
    if sources:
        st.markdown("**Sources:**")
        for i, doc in enumerate(sources):
            st.markdown(f"**Source {i+1}:**")
            st.write(doc.page_content[:500])
    
    st.write(result)
    

assistant()

