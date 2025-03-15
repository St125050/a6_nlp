import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.schema import Document

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create vector store directory if it doesn't exist
current_dir = os.path.dirname(os.path.abspath(__file__))
vector_path = os.path.join(current_dir, 'vector-store')
if not os.path.exists(vector_path):
    os.makedirs(vector_path)
    print('Vector store directory created')

# Set the path to the PDF file
pdf_path = os.path.join(current_dir, 'aakashresume.pdf')

# 1. Document Loaders - Load personal information from PDF
def load_documents():
    """Load personal information documents from PDF"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} not found. Ensure the file exists.")
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded document from {pdf_path}")
    
    return documents

# 2. Document Transformers - Split documents into chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into manageable chunks"""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(doc_chunks)} document chunks")
    return doc_chunks

# 3. Text Embedding Models - Create embeddings for document chunks
class SentenceEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        print(f"Initialized embedding model: {model_name}")
        
    def get_embeddings(self, texts):
        """Get embeddings for a list of texts"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return np.array(embeddings)

# 4. Vector Stores - Create or load vector store
class FAISSVectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.documents = []
        
    def add_documents(self, documents):
        """Add documents to the vector store"""
        self.documents = documents
        
        # Extract text content for embedding
        texts = [doc.page_content for doc in documents]
        
        # Get embeddings
        embeddings = self.embedding_model.get_embeddings(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        
        print(f"Added {len(documents)} documents to vector store")
        
    def similarity_search(self, query, k=3):
        """Search for similar documents"""
        # Get query embedding
        query_embedding = self.embedding_model.get_embeddings([query])[0].reshape(1, -1)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Get relevant documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'unknown'),
                    'score': float(distances[0][i])
                })
        
        return results
    
    def save(self, path):
        """Save vector store to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
        
        # Save documents
        with open(os.path.join(path, 'documents.pkl'), 'wb') as f:
            pickle.dump(self.documents, f)
            
        print(f"Saved vector store to {path}")
        
    def load(self, path):
        """Load vector store from disk"""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, 'index.faiss'))
        
        # Load documents
        with open(os.path.join(path, 'documents.pkl'), 'rb') as f:
            self.documents = pickle.load(f)
            
        print(f"Loaded vector store from {path} with {len(self.documents)} documents")
        return self

# 5. LLM Setup - Initialize language model for generation
class TextGenerator:
    def __init__(self, model_id='google/flan-t5-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            device_map='auto',
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        self.pipe = pipeline(
            task="text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            model_kwargs={
                "temperature": 0.1,
                "repetition_penalty": 1.2
            }
        )
        
        print(f"Initialized language model: {model_id}")
        
    def generate(self, prompt):
        """Generate text based on prompt"""
        result = self.pipe(prompt)
        return result[0]['generated_text']

# 6. RAG System - Combine retrieval and generation
class RAGSystem:
    def __init__(self, vector_store, generator):
        self.vector_store = vector_store
        self.generator = generator
        self.chat_history = []
        
    def query(self, question):
        """Process a query through the RAG system"""
        # Retrieve relevant documents
        results = self.vector_store.similarity_search(question)
        
        # Format context for the generator
        context = "\n\n".join([f"From {r['source']}:\n{r['content']}" for r in results])
        
        # Create prompt
        prompt = f"""
        I am a helpful assistant that provides accurate information based on the provided context.
        I will answer questions about personal information, education, work experience, skills, beliefs, and other relevant details.
        I will be friendly, conversational, and informative in my responses.
        If I don't know the answer based on the provided context, I will say so honestly rather than making up information.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """.strip()
        
        # Generate answer
        answer = self.generator.generate(prompt)
        
        # Update chat history
        self.chat_history.append({
            "question": question,
            "answer": answer
        })
        
        return {
            "question": question,
            "answer": answer,
            "sources": results
        }
    
    def save_chat_history(self, filename='qa_history.json'):
        """Save chat history to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.chat_history, f, indent=2)
        print(f"Saved chat history to {filename}")

# Main function to initialize and return the RAG system
def initialize_rag_system():
    """Initialize the complete RAG system"""
    print("Initializing RAG system...")
    
    # Set up paths
    vector_store_path = os.path.join(vector_path, 'personal_info_db')
    
    # Initialize embedding model
    embedder = SentenceEmbedder()
    
    # Initialize vector store
    vector_store = FAISSVectorStore(embedder)
    
    # Check if vector store exists
    if os.path.exists(os.path.join(vector_store_path, 'index.faiss')):
        print("Loading existing vector store...")
        vector_store.load(vector_store_path)
    else:
        print("Creating new vector store...")
        # Load and process documents
        documents = load_documents()
        doc_chunks = split_documents(documents)
        
        # Add documents to vector store
        vector_store.add_documents(doc_chunks)
        
        # Save vector store
        vector_store.save(vector_store_path)
    
    # Initialize text generator
    generator = TextGenerator()
    
    # Create RAG system
    rag_system = RAGSystem(vector_store, generator)
    
    print("RAG system initialization complete")
    return rag_system

# If run directly, test the RAG system
if __name__ == "__main__":
    # Initialize the RAG system
    rag = initialize_rag_system()
    
    # Test with a few questions
    test_questions = [
        "How old are you?",
        "What is your highest level of education?",
        "What are your core beliefs regarding technology?"
    ]
    
    print("\nTesting RAG system with sample questions:")
    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = rag.query(question)
        print(f"Answer: {result['answer']}")
        print("Sources:")
        for source in result['sources']:
            print(f" - {source['source']}")
