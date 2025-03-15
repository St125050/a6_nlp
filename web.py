import streamlit as st
import os
from langchain.chains import RetrievalQA
from app import ask_chatbot

# ✅ Set up the page
st.set_page_config(page_title="📄 AI Document Assistant", layout="wide")

# ✅ Custom Styling
st.markdown(
    """
    <style>
        body {background-color: #f4f4f4; font-family: 'Arial', sans-serif;}
        .stTextInput>div>div>input {
            border-radius: 10px; 
            padding: 12px;
            font-size: 18px;
            border: 2px solid #4CAF50;
        }
        .stButton>button {
            border-radius: 10px; 
            padding: 12px 20px; 
            background: linear-gradient(135deg, #4CAF50, #2E8B57);
            color: white; 
            font-size: 18px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #2E8B57, #4CAF50);
        }
        .answer-box {
            background: #E8F5E9;
            padding: 15px;
            border-radius: 10px;
            font-size: 18px;
            color: #2E7D32;
            font-weight: bold;
        }
        .source-box {
            background: #F1F8E9;
            padding: 10px;
            border-radius: 8px;
            font-size: 16px;
            color: #558B2F;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Title & Description
st.markdown("""
    <h1 style='text-align: center;'>📄 AI-Powered Document Assistant</h1>
    <p style='text-align: center; font-size: 18px;'>Ask a question about the uploaded documents and get an AI-powered response instantly!</p>
""", unsafe_allow_html=True)

# ✅ Input Section
question = st.text_input("Enter your question:", "What is Ponkrit Kaewsawee's highest level of education?")

# ✅ Process the Question
if st.button("🔍 Get Answer"):
    with st.spinner("🤖 Thinking... Fetching the most relevant answer..."):
        answer, sources = ask_chatbot(question)
        
        # ✅ Display Answer
        st.subheader("💡 Answer:")
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
        
        # ✅ Display Sources
        if sources:
            st.subheader("📌 Sources:")
            for source in sources:
                st.markdown(f"<div class='source-box'>📜 {source.metadata.get('source', 'Unknown')}</div>", unsafe_allow_html=True)
        else:
            st.info("No sources found for this question.")
