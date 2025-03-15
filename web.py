import streamlit as st
import json
import os
from simple_rag import initialize_rag_system
from model_analysis import analyze_retriever_model, analyze_generator_model

# Initialize the RAG system
st.write("Initializing RAG system...")
rag_system = initialize_rag_system()
st.write("RAG system initialized and ready to use")

# Store conversation history for the demo
qa_history = []

# Run initial model analysis with a few test questions
test_questions = [
    "How old are you?",
    "What is your highest level of education?",
    "What are your core beliefs regarding technology?",
    "What programming languages do you know?",
    "Where did you work before Google?"
]

# Store analysis results
retriever_analysis = analyze_retriever_model(rag_system, test_questions, verbose=False)
generator_analysis = analyze_generator_model(rag_system, test_questions, verbose=False)

# Main application
st.title("RAG Chatbot Demo")

# Display model analysis
st.subheader("Retriever Model Analysis")
st.json(retriever_analysis)

st.subheader("Generator Model Analysis")
st.json(generator_analysis)

# Chat interface
st.subheader("Chat with the RAG System")

user_message = st.text_input("Enter your message:")

if st.button("Send"):
    if user_message:
        # Query the RAG system
        result = rag_system.query(user_message)
        
        # Add to history
        qa_pair = {
            "question": result["question"],
            "answer": result["answer"]
        }
        qa_history.append(qa_pair)
        
        # Save the updated history to a JSON file
        with open('qa_history.json', 'w') as f:
            json.dump(qa_history, f, indent=2)
        
        # Display the result
        st.write("Answer:", result["answer"])
        st.write("Sources:", result["sources"])
    else:
        st.write("Please enter a message.")

# Display question-answer history
st.subheader("Question-Answer History")
st.json(qa_history)

# Analyze specific question
st.subheader("Analyze a Specific Question")

analyze_question = st.text_input("Enter a question to analyze:")

if st.button("Analyze"):
    if analyze_question:
        # Analyze with retriever model
        retriever_results = rag_system.vector_store.similarity_search(analyze_question)
        
        # Analyze with generator model (get full response)
        generation_result = rag_system.query(analyze_question)
        
        # Display analysis results
        st.write("Retriever Results:", retriever_results)
        st.write("Generation Result:", generation_result)
    else:
        st.write("Please enter a question to analyze.")
