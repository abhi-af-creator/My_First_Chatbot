"""
utils.py
Helper functions: query preprocessing, safety checks, evaluation helpers.
"""

import re
from typing import List
from langchain_community.docstore.document import Document

# Simple blocked keywords (extend as needed)
BLOCKED_KEYWORDS = [
    "hack", "attack", "kill", "bomb", "terror", "illegal", "poison", "explode", "nuke"
]

def is_allowed_file(filename: str, allowed_extensions: List[str]) -> bool:
    ext = filename.lower().strip()
    return any(ext.endswith(e) for e in allowed_extensions)

def is_safe_query(query: str) -> bool:
    q = query.lower()
    for w in BLOCKED_KEYWORDS:
        if w in q:
            return False
    return True

def clean_query(query: str) -> str:
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    return query

def expand_query(query: str) -> str:
    # If too short, expand heuristically to get better retrieval
    if len(query.split()) < 4:
        return f"Provide details about: {query}"
    return query

def preprocess_query(query: str) -> str:
    q = clean_query(query)
    q = expand_query(q)
    return q

# Evaluation helpers (basic)
def recall_at_k(retriever, question: str, ground_truth_text: str, k: int = 3) -> float:
    """
    Simple recall@k: returns 1.0 if any of top-k retrieved chunks contain ground_truth_text (substring match).
    """
    docs = retriever.get_relevant_documents(question)[:k]
    for d in docs:
        if ground_truth_text.lower() in d.page_content.lower():
            return 1.0
    return 0.0
