import streamlit as st
import time

from components.sidebar import show_sidebar
from services.api import ask_ai
from services.s3 import upload_document
from services.config import PROJECT_NAME


st.set_page_config(
    page_title=PROJECT_NAME,
    page_icon="📄",
    layout="wide"
)

show_sidebar()

st.title("📄 Document Intelligence Platform")
st.caption("Upload PDFs, extract text with Textract, search with OpenSearch, and answer with Bedrock Nova Lite.")

tab1, tab2 = st.tabs(["📤 Upload Document", "💬 Ask Questions"])

with tab1:
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        st.write(f"Selected file: **{uploaded_file.name}**")

        if st.button("Upload and Process"):
            with st.spinner("Uploading document to S3..."):
                upload_document(uploaded_file)

            st.success("Document uploaded successfully.")
            st.info("Please wait 30–60 seconds for Textract extraction and vector indexing.")

with tab2:
    st.subheader("Ask Questions")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask something about your uploaded document...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                start = time.time()
                result = ask_ai(question)
                elapsed = round(time.time() - start, 2)

                answer = (
                    result.get("answer")
                    or result.get("message", {}).get("content")
                    or str(result)
                )

                st.write(answer)
                st.caption(f"Response time: {elapsed}s")

        st.session_state.messages.append({"role": "assistant", "content": answer})
