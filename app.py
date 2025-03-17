import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import time
from dotenv import load_dotenv

load_dotenv()

## Load the NVIDIA API key
if not os.getenv("NVIDIA_API_KEY"):
    st.error("NVIDIA API key is missing. Check your .env file.")
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

def save_uploaded_file(uploaded_file):
    os.makedirs("uploaded_pdfs", exist_ok=True)
    file_path = os.path.join("uploaded_pdfs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def vector_embedding(uploaded_file):
    if "vectors" not in st.session_state:
        file_path = save_uploaded_file(uploaded_file)
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFLoader(file_path)
        st.session_state.docs = st.session_state.loader.load()
        
        if not st.session_state.docs:
            st.error("No documents loaded. Ensure the PDF contains text.")
            return
        
        st.write("Sample document content:", st.session_state.docs[0].page_content[:500])
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        if not st.session_state.final_documents:
            st.error("Text splitting failed. Check document structure.")
            return
        
        st.write(f"Total split documents: {len(st.session_state.final_documents)}")
        
        texts = [doc.page_content for doc in st.session_state.final_documents]
        
        if not texts:
            st.error("No text extracted from documents.")
            return
        
        try:
            st.session_state.vectors = FAISS.from_texts(texts, st.session_state.embeddings)
            st.success("Vector Store DB is Ready!")
        except Exception as e:
            st.error(f"Error creating FAISS index: {e}")

st.title("Nvidia NIM Demo")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file and st.button("Process PDF"):
    vector_embedding(uploaded_file)

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
)

prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1:
    if "vectors" not in st.session_state:
        st.error("Vector database is not initialized. Please upload and process a PDF first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time:", time.process_time() - start)
        
        st.write(response['answer'])
        
 # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")