import os
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from simple_rag import initialize_rag_system, SentenceEmbedder

def analyze_retriever_model(rag_system, test_questions, verbose=True):
    """
    Analyze the retriever model's performance in finding relevant documents.
    
    Args:
        rag_system: The initialized RAG system
        test_questions: List of test questions
        verbose: Whether to print detailed analysis
        
    Returns:
        Dictionary with analysis results
    """
    # Handle case where RAG system is not initialized
    if rag_system is None:
        return {
            "error": "RAG system not initialized",
            "average_relevance": 0,
            "irrelevant_percentage": 0,
            "recommendations": ["Initialize the RAG system before analysis"]
        }
    
    if verbose:
        print("\n=== Retriever Model Analysis ===")
        print(f"Model: sentence-transformers/all-mpnet-base-v2")
        print(f"Vector Store: FAISS (Facebook AI Similarity Search)")
    
    # Analyze retrieval performance
    retrieval_scores = []
    irrelevant_retrievals = 0
    total_retrievals = 0
    
    try:
        for question in test_questions:
            # Get retrievals without generating answer
            results = rag_system.vector_store.similarity_search(question)
            total_retrievals += len(results)
            
            # Check relevance (simple heuristic - if question keywords appear in content)
            question_keywords = set(question.lower().split())
            question_keywords = {word for word in question_keywords if len(word) > 3}  # Filter out short words
            
            for result in results:
                content = result['content'].lower()
                keyword_matches = sum(1 for keyword in question_keywords if keyword in content)
                relevance_score = keyword_matches / len(question_keywords) if question_keywords else 0
                
                retrieval_scores.append(relevance_score)
                if relevance_score < 0.3:  # Threshold for irrelevance
                    irrelevant_retrievals += 1
    except Exception as e:
        return {
            "error": str(e),
            "average_relevance": 0,
            "irrelevant_percentage": 0,
            "recommendations": ["Error during retrieval analysis"]
        }
    
    # Calculate metrics
    average_relevance = np.mean(retrieval_scores) if retrieval_scores else 0
    irrelevant_percentage = (irrelevant_retrievals / total_retrievals * 100) if total_retrievals > 0 else 0
    
    # Generate recommendations
    recommendations = []
    if average_relevance < 0.5:
        recommendations.append("Consider using a different embedding model for better semantic understanding")
    if irrelevant_percentage > 30:
        recommendations.append("Improve document chunking strategy to create more focused chunks")
    if irrelevant_percentage > 50:
        recommendations.append("Expand the knowledge base with more relevant information")
    
    if not recommendations:
        recommendations.append("Retriever model is performing well, no immediate improvements needed")
    
    # Format results
    results = {
        "average_relevance": round(float(average_relevance), 2),
        "irrelevant_percentage": round(float(irrelevant_percentage), 2),
        "recommendations": recommendations
    }
    
    if verbose:
        print(f"Average Relevance Score: {results['average_relevance']}")
        print(f"Irrelevant Retrievals: {irrelevant_percentage:.2f}%")
        print("Recommendations:")
        for rec in recommendations:
            print(f"- {rec}")
    
    return results

def analyze_generator_model(rag_system, test_questions, verbose=True):
    """
    Analyze the generator model's performance in providing accurate and relevant responses.
    
    Args:
        rag_system: The initialized RAG system
        test_questions: List of test questions
        verbose: Whether to print detailed analysis
        
    Returns:
        Dictionary with analysis results
    """
    # Handle case where RAG system is not initialized
    if rag_system is None:
        return {
            "error": "RAG system not initialized",
            "hallucination_percentage": 0,
            "recommendations": ["Initialize the RAG system before analysis"]
        }
    
    if verbose:
        print("\n=== Generator Model Analysis ===")
        print(f"Model: google/flan-t5-base")
    
    # Analyze generation performance
    hallucination_count = 0
    total_generations = len(test_questions)
    
    try:
        for question in test_questions:
            # Get full response
            result = rag_system.query(question)
            answer = result["answer"]
            sources = result["sources"]
            
            # Check for potential hallucinations (simple heuristic)
            source_content = " ".join([s["content"] for s in sources]).lower()
            
            # Extract key statements from answer (simplistic approach)
            statements = [sent.strip() for sent in answer.split('.') if sent.strip()]
            
            for statement in statements:
                # If a substantial statement doesn't
