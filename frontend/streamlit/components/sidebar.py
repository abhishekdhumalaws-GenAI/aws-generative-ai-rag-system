import streamlit as st
from services.config import PROJECT_NAME, MODEL_NAME, VECTOR_DB

def show_sidebar():
    st.sidebar.title(PROJECT_NAME)
    st.sidebar.success("🟢 Connected")
    st.sidebar.markdown("---")
    st.sidebar.write("Model")
    st.sidebar.info(MODEL_NAME)
    st.sidebar.write("Vector Database")
    st.sidebar.info(VECTOR_DB)
