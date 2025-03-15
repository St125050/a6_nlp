import streamlit as st
from app import ask_chatbot
import json

# Set up the page
st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="centered")

# Custom Styling
st.markdown(
    """
    <style>
        body {background-color: #282828; color: #E0E0E0; font-family: 'Helvetica', sans-serif;}
        .stTextInput>div>div>input {
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            border: 2px solid #00ADB5;
            background-color: #393E46;
            color: #E0E0E0;
        }
        .stButton>button {
            border-radius: 5px;
            padding: 10px 20px;
            background-color: #00ADB5;
            color: #FFFFFF;
            font-size: 16px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #007B7F;
        }
        .answer-box {
            background: #393E46;
            padding: 20px;
            border-radius: 10px;
            font-size: 16px;
            color: #00ADB5;
            margin-top: 20px;
        }
        .source-box {
            background: #222831;
            padding: 15px;
            border-radius: 10px;
            font-size: 14px;
            color: #EEEEEE;
            margin-top: 10px;
        }
        .header {
            text-align: center;
            color: #00ADB5;
            margin-top: 20px;
        }
        .description {
            text-align: center;
            font-size: 18px;
            margin-bottom: 30px;
            color: #EEEEEE;
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

# Initialize list to store question-answer pairs
qa_pairs = []

# Input Section
st.markdown("<h4 style='text-align: center;'>Enter your question below:</h4>", unsafe_allow_html=True)
question = st.text_input("", "")

# Process the Question
if st.button("Get Answer"):
    with st.spinner("Fetching the most relevant answer..."):
        answer, sources = ask_chatbot(question)
        
        # Store the question-answer pair
        qa_pairs.append({"question": question, "answer": answer})

        # Display Answer
        st.markdown("<h3 style='color: #00ADB5;'>Answer:</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
        
        # Display Sources
        if sources:
            st.markdown("<h3 style='color: #00ADB5;'>Sources:</h3>", unsafe_allow_html=True)
            for source in sources:
                st.markdown(f"<div class='source-box'>{source.metadata.get('source', 'Unknown')}</div>", unsafe_allow_html=True)
        else:
            st.info("No sources found for this question.")

# Provide option to download the JSON file
if qa_pairs:
    st.markdown("<h4 style='text-align: center;'>Download your question-answer pairs:</h4>", unsafe_allow_html=True)
    json_output = json.dumps(qa_pairs, indent=2)
    st.download_button(
        label="Download JSON",
        data=json_output,
        file_name="qa_pairs.json",
        mime="application/json"
    )
