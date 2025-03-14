import streamlit as st
from app import ask_chatbot

# Set up the page
st.set_page_config(page_title="AI Document Assistant", layout="centered", initial_sidebar_state="collapsed")

# Custom Styling
st.markdown(
    """
    <style>
        body {background-color: #1E1E1E; color: #C0C0C0; font-family: 'Verdana', sans-serif;}
        .stTextInput>div>div>input {
            border-radius: 20px;
            padding: 15px;
            font-size: 20px;
            border: 1px solid #FF4500;
            background-color: #333333;
            color: #FFFFFF;
        }
        .stButton>button {
            border-radius: 20px;
            padding: 10px 25px;
            background: #FF4500;
            color: #FFFFFF;
            font-size: 20px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: #FF6347;
        }
        .answer-box {
            background: #444444;
            padding: 25px;
            border-radius: 15px;
            font-size: 18px;
            color: #FF4500;
            font-weight: bold;
        }
        .source-box {
            background: #555555;
            padding: 20px;
            border-radius: 15px;
            font-size: 16px;
            color: #FFA07A;
        }
        .header {
            text-align: center;
            color: #FF4500;
            margin-top: 50px;
        }
        .description {
            text-align: center;
            font-size: 18px;
            margin-bottom: 30px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title & Description
st.markdown("""
    <h1 class='header'>AI Document Assistant</h1>
    <p class='description'>Ask a question about the documents and get an AI-powered response instantly!</p>
""", unsafe_allow_html=True)

# Input Section
st.markdown("<h3 style='text-align: center;'>Type your question below:</h3>", unsafe_allow_html=True)
question = st.text_input("", "")

# Process the Question
if st.button("Get Answer"):
    with st.spinner("Fetching the most relevant answer..."):
        answer, sources = ask_chatbot(question)
        
        # Display Answer
        st.markdown("<h3 style='color: #FF4500;'>Answer:</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
        
        # Display Sources
        if sources:
            st.markdown("<h3 style='color: #FF4500;'>Sources:</h3>", unsafe_allow_html=True)
            for source in sources:
                st.markdown(f"<div class='source-box'>{source.metadata.get('source', 'Unknown')}</div>", unsafe_allow_html=True)
        else:
            st.info("No sources found for this question.")
