"""
rag_pipeline.py
Core RAG pipeline: PDF loading, chunking, embeddings, FAISS, retrieval, and generation.
Designed for offline use with Hugging Face sentence-transformers and a local text2text model.
"""

import os
import tempfile
from typing import List, Dict
import yaml
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Some systems may not have langchain_community; the code expects the earlier installed packages

class RAGPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.embedding_model = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.generator_model = config.get("generator_model", "google/flan-t5-base")
        self.chunk_size = config.get("chunk_size", 800)
        self.chunk_overlap = config.get("chunk_overlap", 150)
        self.top_k = config.get("top_k", 3)
        self.max_output_tokens = config.get("max_output_tokens", 300)
        self._vectorstore = None
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        # generator pipeline for text2text
        gen_pipeline = pipeline(
            "text2text-generation",
            model=self.generator_model,
            tokenizer=self.generator_model,
            max_new_tokens=self.max_output_tokens,
            truncation=True
        )
        self.llm = HuggingFacePipeline(pipeline=gen_pipeline)

    def index_documents(self, uploaded_files: List):
        """
        Build the FAISS index from uploaded PDF files (list of Streamlit UploadedFile objects).
        """
        docs = []
        # load PDFs temporarily and extract
        for file in uploaded_files:
            # write file to temp
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(file.getvalue())
            tmp.flush()
            tmp.close()
            loader = PyPDFLoader(tmp.name)
            loaded = loader.load()
            # Add metadata of file
            for d in loaded:
                # attach filename as metadata title
                d.metadata = d.metadata or {}
                d.metadata["title"] = getattr(file, "name", "uploaded_pdf")
            docs.extend(loaded)
            # do not delete temp immediate (FAISS may read)
        # split
        split_docs = self._splitter.split_documents(docs)
        # build vectorstore
        self._vectorstore = FAISS.from_documents(split_docs, self._embeddings)
        # keep split_docs as last_used maybe
        self._last_docs = split_docs

    def num_chunks(self):
        return len(self._last_docs) if hasattr(self, "_last_docs") else 0

    def ask(self, question: str) -> Dict:
        """
        Run retrieval then generation. Returns dict with answer and retrieved docs.
        """
        if not self._vectorstore:
            raise ValueError("Vector store not initialized. Call index_documents() first.")
        retriever = self._vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        retrieved = retriever.get_relevant_documents(question)
        # build context
        context = "\n\n".join([d.page_content for d in retrieved])
        # instruct model to use only context
        final_prompt = (
            "You are a helpful assistant. Use ONLY the following context to answer the question. "
            "If the answer is not in the context, reply: 'I could not find this information in the uploaded PDFs.'\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        raw = self.llm(final_prompt)
        # HuggingFacePipeline returns list or string; normalize
        if isinstance(raw, list):
            answer = raw[0].get("generated_text", "").strip()
        elif isinstance(raw, dict):
            answer = raw.get("generated_text", "").strip()
        else:
            answer = str(raw).strip()
        return {"answer": answer, "retrieved": retrieved}
