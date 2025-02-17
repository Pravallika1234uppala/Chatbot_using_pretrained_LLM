import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI



#creating streamlit app:
st.header("Chatbot-Train it")

#uploading specfic domian pdf:
with st.sidebar:
    st.title("Domian Specific Documents:")
    file = st.file_uploader("Upload a pdf file to train chatbot:", type="pdf")

#string intialization:
text = ""

if file is not None:
    pdf_reader = PdfReader(file)

    for page in pdf_reader.pages:
        text += page.extract_text()
        #st.write(text)

    #breaking text into chunks:
    text_split = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )

    #len as it is string text

    chunks = text_split.split_text(text)

    #st.write(chunks)

    #Generating Embeddings
    #unique api key

    if chunks:


        embeddings = OpenAIEmbeddings(openai_api_key="#Give open ai api key")

        #this generates emmedings

        #Creating vector stores

        vector_store = FAISS.from_texts(chunks, embeddings)

        #storing chunks and its corresponding embeddings


        #Now user part:

        #getting user question

        user_question=st.text_input("Type your question?")

        #finding similar answers in db

        if user_question:
            match=vector_store.similarity_search(user_question)
            #st.write(match)

            #llm(to generate answers from matches)
            llm = ChatOpenAI(
                openai_api_key="#Give open ai api key",
                temperature=0,
                max_tokens=100,
                model_name="gpt-3.5-turbo"

            )

            #chaining:
            chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vector_store.as_retriever())
            #bucket type of text sent into llm

            response=chain.invoke(user_question)
            #giving match of similar found and a question asked with assigned model and type.

            st.write(response["result"])



