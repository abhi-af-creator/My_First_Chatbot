#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tempfile
import warnings
import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="AI-Powered PDF Chatbot", layout="wide")
st.title("AI-Powered PDF Chatbot")
st.markdown("Upload up to **2 PDF files** and ask questions based on their content.")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# ------------------------------
# Function to process PDFs
def process_pdfs(files):
    all_docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            all_docs.extend(loader.load())
    return all_docs

# ------------------------------
# Main App Logic
if uploaded_files:
    st.info("Processing your documents... ")
    docs = process_pdfs(uploaded_files)

    if not docs:
        st.error("No content found in the uploaded PDFs.")
    else:
        st.success(f" Loaded {len(docs)} pages successfully!")

        # Text splitting for better embedding quality
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = splitter.split_documents(docs)

        # Embeddings using a robust model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Create FAISS vectorstore
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # Choose a stronger LLM for better answers
        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            tokenizer="google/flan-t5-large",
            max_new_tokens=400,
            temperature=0.3,
        )
        llm = HuggingFacePipeline(pipeline=generator)

        # Prompt engineering for better structure
        template = """
        You are an intelligent AI assistant. Use the extracted context from the uploaded PDFs to answer the question below.
        Be **accurate**, **concise**, and **well-structured**. If unsure, say "I could not find enough information."
        
        Context:
        {context}
        
        Question:
        {question}
        
        Helpful Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Query section
        st.markdown("###  Ask me anything about your PDFs")
        query = st.text_input("Enter your question here")

        def rag_query(question):
            retrieved_docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([d.page_content for d in retrieved_docs])
            final_prompt = prompt.format(context=context, question=question)
            answer = llm(final_prompt)
            return answer, context

        if query:
            with st.spinner("Thinking... "):
                answer, context = rag_query(query)
                st.markdown("###  Answer:")
                st.write(answer[0]['generated_text'] if isinstance(answer, list) else answer)
                
                with st.expander("🔍 View Retrieved Context"):
                    st.text(context[:1500] + "...")


# In[ ]:




