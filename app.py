import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdffReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationRetrievalChain


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdffReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=700,
        overlap=200,
        length_function=len
    )
    chunks=text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm=ChatOpenAI()
    memory= ConversationBufferMemory(memory_key="chat-history",return_messages=True)
    conversation_chain= ConversationRetrievalChain(
        llm=llm,
        retriever=vectorstore,
        memory=memory
    )
    return  conversation_chain

def handle_userinput(user_question):
    response=st.session_state.conversation({"question":"user_question"})
    st.session_state.chat_history=response['chat-history']
    for i,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace(
                "{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}",message.content),unsafe_allow_html=True
            )
        

def main():
    pass