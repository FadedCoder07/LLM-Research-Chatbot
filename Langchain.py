import os
import streamlit as st
import time
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv() 

st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
num_urls = st.sidebar.number_input("Number of URLs", min_value=1, max_value=10, value=3, step=1)
for i in range(num_urls):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...")
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()

    try:
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...")
        time.sleep(2)
    except Exception as e:
        st.error(f"Error while creating vector store: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    if 'vectorstore_openai' not in locals():
        st.error("Vector store not initialized. Please process URLs first.")
    else:
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore_openai.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
  
        st.header("Answer")
        st.write(result["answer"])
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  
            for source in sources_list:
                st.write(source)
