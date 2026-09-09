import sys
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "noticerag"), str(BASE_DIR / "retrieval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from generation.rag_pipeline import answer_question

st.set_page_config(page_title="NITA Campus Intelligence Assistant", layout="centered")

st.title("NITA Campus Intelligence Assistant")

query = st.text_input("Enter your question:", placeholder="e.g., When is the next holiday or Janmashtami notice?")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Searching notices and generating answer..."):
            res = answer_question(query.strip())
            st.subheader("Answer:")
            st.write(res["answer"])
    else:
        st.warning("Please enter a question.")
