# a6_nlp

## Deployment

The deployment link for the Streamlit application is: [Your Deployment Link Here](https://jwko7hx7nbp2stsxpg6uee.streamlit.app/)
Here's an explanation of what we have done in the provided code snippets, along with the implementation details of each step:

### Overview
We are building an AI-powered chatbot that can answer questions about a specific individual (Aakash) based on their resume (in PDF format). The chatbot utilizes FAISS for vector storage and retrieval, Hugging Face models for language generation, and Streamlit for the web interface.

### Key Steps and Code Explanation

1. **Install Required Packages**
   ```python
   !pip install pypdf
   !pip install -U langchain-community
   !pip install streamlit
   ```

   We start by installing the necessary packages such as `pypdf`, `langchain-community`, and `streamlit` using pip.

2. **Set Up Environment Variables**
   ```python
   import os
   os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
   ```

   We set the Hugging Face API token as an environment variable to authenticate and access Hugging Face models.

3. **Load and Initialize Hugging Face Models**
   ```python
   from langchain.llms import HuggingFaceHub

   hf_llm = HuggingFaceHub(
       repo_id="google/flan-t5-large",
       huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
       model_kwargs={"temperature": 0.7, "max_length": 512}
   )

   hf_llm_alternate = HuggingFaceHub(
       repo_id="gpt2",
       huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
       model_kwargs={"temperature": 0.7, "max_length": 512}
   )
   ```

   We initialize two Hugging Face models: `flan-t5-large` and `gpt2`, which will be used to generate responses.

4. **Load and Process PDF Documents**
   ```python
   from langchain.document_loaders import PyPDFLoader
   from langchain.text_splitter import CharacterTextSplitter

   pdf_files = ["aakashresume.pdf"]

   documents = []
   for pdf_file in pdf_files:
       loader = PyPDFLoader(pdf_file)
       documents.extend(loader.load())

   text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
   text_chunks = text_splitter.split_documents(documents)
   ```

   We load the PDF file (`aakashresume.pdf`) using `PyPDFLoader` and split the documents into chunks using `CharacterTextSplitter`.

5. **Create FAISS Vector Store**
   ```python
   import faiss
   import numpy as np
   from sentence_transformers import SentenceTransformer
   from langchain.vectorstores import FAISS
   from langchain.storage import InMemoryStore
   from langchain_core.documents import Document

   embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
   embeddings = embedding_model.encode([doc.page_content for doc in text_chunks], convert_to_tensor=False)
   embedding_matrix = np.array(embeddings).astype("float32")

   index = faiss.IndexFlatL2(embedding_matrix.shape[1])
   index.add(embedding_matrix)
   faiss.write_index(index, "faiss_index.bin")

   docstore = InMemoryStore()
   index_to_docstore_id = {i: str(i) for i in range(len(text_chunks))}
   docstore.mset([(str(i), doc) for i, doc in enumerate(text_chunks)])

   vector_store = FAISS(
       embedding_function=embedding_model.encode,
       index=index,
       docstore=docstore,
       index_to_docstore_id=index_to_docstore_id
   )
   ```

   We create a FAISS vector store by encoding the text chunks using `SentenceTransformer` and storing the embeddings in a FAISS index. The index is saved to disk for future use.

6. **Set Up Retriever and QA Chain**
   ```python
   from langchain.chains import RetrievalQA

   retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

   qa_chain_hf = RetrievalQA.from_chain_type(
       llm=hf_llm,
       chain_type="stuff",
       retriever=retriever,
       return_source_documents=True
   )

   qa_chain_hf_alternate = RetrievalQA.from_chain_type(
       llm=hf_llm_alternate,
       chain_type="stuff",
       retriever=retriever,
       return_source_documents=True
   )
   ```

   We set up a retriever to fetch relevant documents based on similarity and create QA chains using the Hugging Face models to generate answers.

7. **Streamlit Application**
   ```python
   import streamlit as st

   st.title("AI Chatbot")
   st.write("Ask questions about Aakash and get responses along with relevant source documents.")

   question = st.text_input("Enter your question:")
   model = st.selectbox("Select Model", ["Hugging Face - google/flan-t5-large", "Hugging Face - GPT-2", "Groq Cloud - Llama"])

   if st.button("Ask"):
       if model == "Hugging Face - google/flan-t5-large":
           model_key = "hf"
       elif model == "Hugging Face - GPT-2":
           model_key = "hf_alternate"
       elif model == "Groq Cloud - Llama":
           model_key = "groq"

       answer, source_documents = ask_chatbot(question, model=model_key)
       st.write("Answer:", answer)
       st.write("Source Documents:")
       for doc in source_documents:
           st.write(doc.page_content)
   ```

   We create a Streamlit web application that allows users to enter questions and select a model. The application fetches answers from the QA chain and displays them along with the source documents.

### Summary
- **Installation of Packages**: Install necessary libraries.
- **Environment Variables**: Set environment variables for API tokens.
- **Model Initialization**: Load and initialize language models from Hugging Face.
- **Document Processing**: Load and split PDF documents into chunks.
- **FAISS Vector Store**: Create and save a FAISS vector store for document embeddings.
- **Retriever and QA Chain**: Set up a retriever and QA chains for question answering.
- **Streamlit Application**: Build a web interface using Streamlit to interact with the chatbot.

These steps provide a comprehensive solution for building an AI chatbot that can answer questions based on a given PDF document, utilizing vector search and language models for accurate and informative responses.
