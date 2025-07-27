#!/usr/bin/env python3
"""
Simple test script for the ColBERT RAG pipeline.
This processes a single question through the pipeline for debugging purposes.
"""

import sys
import asyncio
from pathlib import Path

# Add the parent directory to the Python path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.absolute()))

# Import necessary modules
from regulations_rag_eval.rag_implementations.paths import implementation_system_path
from regulations_rag_eval.rag_implementations.ColBERT.colbert import (
    get_weaviate_client,
    load_metadata,
    retrieve_context_with_hyde,
    call_llm,
)

async def main():
    """Run the ColBERT pipeline on a test question"""
    # Test question
    question = "Should an open-end investment fund be considered an obligor under Art. 178(1) CRR, irrespective of whether it has legal personality under a Member States' regulations on investment funds?"
    
    # Get path to index directory
    index_dir = implementation_system_path("ColBERT")
    
    # Load metadata
    metadata = load_metadata(index_dir)
    collection_name = metadata.get("collection_name", "RegulationArticles")
    
    # Connect to Weaviate
    client = get_weaviate_client()
    
    # Retrieve context with article references
    context, retrieved_articles = await retrieve_context_with_hyde(
        client, question, collection_name, k=25
    )
    
    # Generate answer
    answer = await call_llm(question, context)
    
    # Print answer and retrieved articles
    print("\nAnswer:")
    print(answer)
    
    print("\nRetrieved Articles:")
    for article in retrieved_articles:
        print(f"- {article}")
    
    # Clean up
    client.close()

if __name__ == "__main__":
    asyncio.run(main())
