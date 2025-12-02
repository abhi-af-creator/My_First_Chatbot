"""
app.py
Streamlit UI for PDF InsightBot (offline RAG)
"""

import os
import streamlit as st
from pathlib import Path
import yaml
from rag_pipeline import RAGPipeline
from utils import preprocess_query, is_allowed_file, is_safe_query

st.set_page_config(page_title="PDF InsightBot", layout="centered")

# Load config
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# App title/description
st.title("📘 PDF InsightBot")
st.markdown(
    "Upload one or two PDF documents and ask questions. "
    "Answers are generated from the content of your PDFs using a local RAG pipeline (no API keys required)."
)

# File uploader
uploaded_files = st.file_uploader(
    "Upload up to 2 PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Only PDF files are accepted. Maximum file size is configured in config.yaml."
)

# Initialize pipeline
@st.cache_resource(show_spinner=False)
def get_pipeline():
    pipeline = RAGPipeline(config)
    return pipeline

pipeline = get_pipeline()

if uploaded_files:
    # Basic file checks
    valid_files = []
    for f in uploaded_files:
        if not is_allowed_file(f.name, config["allowed_extensions"]):
            st.error(f"File {f.name} has unsupported extension.")
        elif f.size > config.get("max_upload_bytes", 20 * 1024 * 1024):
            st.error(f"File {f.name} is too large. Max allowed {config.get('max_upload_bytes') // (1024*1024)} MB.")
        else:
            valid_files.append(f)

    if not valid_files:
        st.stop()

    # Process files and build vector store (cached)
    with st.spinner("Processing PDFs and building index (first run might take some time)..."):
        pipeline.index_documents(valid_files)
    st.success("Index ready — you can now ask questions!")

    # show small stats
    st.info(f"Indexed {pipeline.num_chunks()} chunks. Embedding model: {config['embedding_model']}")

    # Input box
    user_query = st.text_input("Ask a question about the uploaded PDFs:")
    if user_query:
        # safety checks
        if not is_safe_query(user_query):
            st.warning("Your query contains terms that are blocked by safety policy.")
        else:
            q = preprocess_query(user_query)
            with st.spinner("Retrieving and generating answer..."):
                result = pipeline.ask(q)
            # Display
            st.markdown("### 🧠 Answer")
            st.write(result["answer"])

            # Show metadata and retrieved contexts
            with st.expander("🔍 Retrieved context (top chunks)"):
                for i, doc in enumerate(result["retrieved"]):
                    title = doc.metadata.get("title", "source")
                    st.markdown(f"**Chunk {i+1} — {title}**")
                    st.text(doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""))

            # Small evaluation note
            st.caption("Note: The assistant answers based strictly on retrieved context. If the information isn't found, it will say so.")
else:
    st.info("Upload PDF files to begin. You can use the 'sample_pdfs' folder for testing.")

# Footer
st.markdown("---")
st.markdown("Project: PDF InsightBot • Offline RAG demo • No API keys required.")
