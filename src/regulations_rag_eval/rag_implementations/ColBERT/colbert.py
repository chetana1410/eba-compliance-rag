import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.util import generate_uuid5
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel
from regulations_rag_eval.rag_implementation_data_types import RAGResult
from regulations_rag_eval.rag_implementations.paths import implementation_system_path

# Import local OpenRouter request function
from regulations_rag_eval.rag_implementations.ColBERT.utils.openrouter_request import open_router_request
import json

# --- Configuration ---
COLLECTION_NAME = "RegulationArticles"
WEAVIATE_METADATA_FILE = "weaviate_metadata.json"
BATCH_SIZE = 10  # Batch size for importing data

# Default number of results to retrieve
DEFAULT_RETRIEVAL_K = 25  # Changed from 100 to 25 as per requirements

# Model configuration
MODEL_NAME = "lightonai/Reason-ModernColBERT"  # Embedding model
OUTPUT_DIM = 128  # Output dimensionality of the model
HYDE_MODEL = "google/gemini-2.5-flash"  # Model for generating hypothetical answers
WINNOWING_MODEL = "google/gemini-2.5-flash"  # Model for filtering relevant article points
# WINNOWING_MODEL = "anthropic/claude-3.5-sonnet"
ANSWER_MODEL = "openai/gpt-4.1"  # Model for generating final answers

# Winnowing configuration
MAX_TOKENS_PER_GROUP = 5000  # Maximum tokens per group for winnowing

# Device detection with support for Apple MPS
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"  # Apple Silicon GPU
else:
    DEVICE = "cpu"

# Global model instance for reuse
_colbert_model = None

# Load the ModernColBERT model
def get_colbert_model(force_reload=False):
    """
    Initialize and return the ModernColBERT model using Hugging Face.
    Reuses the model if it's already loaded.
    
    Args:
        force_reload: Force reloading of the model even if already loaded
        
    Model details:
    - Model: lightonai/Reason-ModernColBERT
    - Base model: lightonai/GTE-ModernColBERT-v1
    - Document Length: 8192 tokens
    - Query Length: 128 tokens
    - Output Dimensionality: 128 tokens
    - Similarity Function: MaxSim
    
    Returns:
        An object with an encode method for generating embeddings
    """
    global _colbert_model
    if _colbert_model is not None and not force_reload:
        return _colbert_model
    
    try:
        # Try to load the HuggingFace model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        
        # Create a wrapper class with encode method to provide consistent interface
        class ModernColBERTWrapper:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer
            
            def encode(self, text, is_query=False):
                # Set different max lengths based on if it's a query or document
                max_length = 128 if is_query else 8192
                
                # Tokenize the input text
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True
                ).to(DEVICE)
                
                # Generate embeddings with no gradient computation
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Get the token-level embeddings (last hidden state)
                embeddings = outputs.last_hidden_state.cpu().numpy()
                
                # For ColBERT, we need to keep all token embeddings for the multi-vector approach
                # Return the raw embeddings as a list of lists (converting from numpy array)
                # Shape: [num_tokens, embedding_dim]
                token_embeddings = embeddings[0].tolist()  # First dimension is batch, which is always 1 here
                
                return token_embeddings
        
        _colbert_model = ModernColBERTWrapper(model, tokenizer)
        return _colbert_model
    
    except Exception as e:
        print(f"Error loading ModernColBERT model from HuggingFace: {e}")
        print("Falling back to mock model for development.")
        
        # Mock model for development/testing
        class MockColBERT:
            def encode(self, text, is_query=False):
                # For the mock model, we'll generate multiple random vectors to simulate token-level embeddings
                # Create 10 vectors for documents and 5 for queries
                num_tokens = 5 if is_query else 10
                # Return a list of random vectors
                return [np.random.rand(OUTPUT_DIM).tolist() for _ in range(num_tokens)]
        
        _colbert_model = MockColBERT()
        return _colbert_model

# Set path to the regulation data JSON file
# Get the directory of this file
current_dir = Path(__file__).parent.absolute()

# Create absolute path to the JSON file
JSON_FILE_PATH = os.path.join(current_dir, 'utils', 'crr_xsl_extract_processed.json')

# --- Helper Functions ---
def get_weaviate_client() -> weaviate.WeaviateClient:
    """
    Initialize and return a connected Weaviate client using the simplified connection method.
    
    IMPORTANT: This requires a Weaviate instance to be running locally.
    To start one, run the following Docker command:
    docker run --detach -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.30.1
    
    Returns:
        weaviate.WeaviateClient: Initialized and connected Weaviate client
    """
    try:
        # Try to connect to local Weaviate instance
        client = weaviate.connect_to_local(skip_init_checks=True)
        print("Successfully connected to local Weaviate instance")
        return client
    except Exception as e:
        print(f"Error connecting to Weaviate: {e}")
        print("\nIMPORTANT: This error typically means Weaviate is not running.")
        print("To start Weaviate, run the following Docker command:")
        print("docker run --detach -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.30.1")
        print("\nIf you don't want to use Docker, you can install Weaviate in other ways:")
        print("See https://weaviate.io/developers/weaviate/installation for installation options.")
        raise

def create_weaviate_collection(client: weaviate.WeaviateClient, collection_name: str = COLLECTION_NAME) -> None:
    """
    Create a Weaviate collection for storing regulation articles with multi-vector support.
    The schema is designed to match the structure of the crr_xsl_extract.json file.
    
    Args:
        client: Weaviate client
        collection_name: Name of the collection to create
    """
    # Check if collection exists before deleting
    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Deleting...")
        client.collections.delete(collection_name)  # THIS WILL DELETE THE SPECIFIED COLLECTION AND ALL ITS OBJECTS
    
    # Create the collection with multi-vector support
    client.collections.create(
        collection_name,
        vectorizer_config=[
            # User-provided embeddings
            Configure.NamedVectors.none(
                name="multi_vector",
                vector_index_config=Configure.VectorIndex.hnsw(
                    # Enable multi-vector index with default settings
                    multi_vector=Configure.VectorIndex.MultiVector.multi_vector()
                )
            ),
        ],
        properties=[
            # Article properties
            Property(name="article_number",
                    data_type=DataType.TEXT,
                    vectorize_property_name=False,
                    index_filterable=True,  # Allow filtering by article number
                    index_searchable=True), # Allow searching by article number
            Property(name="article_title",
                    data_type=DataType.TEXT,
                    vectorize_property_name=False,
                    index_searchable=True),
                    
            # Point properties
            Property(name="point_id",
                    data_type=DataType.TEXT,
                    vectorize_property_name=False,
                    index_filterable=True),
            Property(name="content",
                    data_type=DataType.TEXT,
                    vectorize_property_name=False,
                    index_searchable=True),  # Make content searchable
        ],
    )
    print(f"Collection '{collection_name}' created successfully with multi-vector support")

def save_metadata(index_dir: str, metadata: Dict) -> None:
    """
    Save metadata to a file.
    
    Args:
        index_dir: Directory to save metadata
        metadata: Dictionary containing metadata
    """
    os.makedirs(index_dir, exist_ok=True)
    metadata_path = os.path.join(index_dir, WEAVIATE_METADATA_FILE)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Metadata saved to {metadata_path}")

def load_metadata(index_dir: str) -> Dict:
    """
    Load metadata from a file.
    
    Args:
        index_dir: Directory containing metadata
        
    Returns:
        Dictionary containing metadata
    """
    metadata_path = os.path.join(index_dir, WEAVIATE_METADATA_FILE)
    
    if not os.path.exists(metadata_path):
        return {}
    
    with open(metadata_path, 'r') as f:
        return json.load(f)

def load_prompt_template(file_path):
    """
    Load a prompt template from a YAML file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        dict: Dictionary containing prompt template
    """
    import yaml
    
    try:
        with open(file_path, 'r') as file:
            prompt_template = yaml.safe_load(file)
        return prompt_template
    except Exception as e:
        print(f"Error loading prompt template: {e}")
        # Fallback to default prompt if file can't be loaded
        return {
            "system_prompt": "You are a regulatory expert. Answer the question based on the provided context.",
            "user_prompt": "Question: {question}\n\nContext: {context}"
        }

async def generate_hyde_answer(question: str) -> str:
    """
    Generate a hypothetical answer for the query using the Hyde technique.
    This creates a concise hypothetical answer with article references that will
    be used to enhance retrieval.
    
    Args:
        question: The user query/question
        
    Returns:
        str: The hypothetical answer to enhance query
    """
    # Get current file's directory and construct path to prompt template
    current_dir = Path(__file__).parent.absolute()
    prompt_path = os.path.join(current_dir, 'prompts', 'hyde.yaml')
    
    # Load prompt template
    prompt_template = load_prompt_template(prompt_path)
    
    # Format the user prompt with the question
    formatted_user_prompt = prompt_template.get("user_prompt", "").format(
        question=question
    )
    
    # Create the full prompt with system message
    system_prompt = prompt_template.get("system_prompt", "")
    
    # Call the LLM with the formatted prompt
    print(f"Generating hypothetical answer for query enhancement...")
    hyde_response = await open_router_request(
        message=formatted_user_prompt,
        model=HYDE_MODEL,
        system_message=system_prompt
    )
    
    try:
        # Clean up the response if it contains markdown code block formatting
        cleaned_response = hyde_response
        if "```json" in hyde_response and "```" in hyde_response:
            # Extract the content between ```json and ```
            cleaned_response = hyde_response.split("```json")[1].split("```")[0].strip()
        
        # Parse the JSON response to extract the hypothetical answer
        hyde_json = json.loads(cleaned_response)
        hypothetical_answer = hyde_json.get("hypothetical_answer", "")
        print(f"Generated hypothetical answer: {hypothetical_answer}")
        return hypothetical_answer
    except json.JSONDecodeError:
        print(f"Error parsing JSON from Hyde response. Using original query.")
        print(f"Raw response: {hyde_response}")
        # Try to extract hypothetical answer using regex as a fallback
        import re
        match = re.search(r'"hypothetical_answer":\s*"([^"]+)"', hyde_response)
        if match:
            hypothetical_answer = match.group(1)
            print(f"Extracted hypothetical answer using regex: {hypothetical_answer}")
            return hypothetical_answer
        return ""

# --- Core RAG Functions ---
def init(index_dir: str, json_file_path: str = JSON_FILE_PATH):
    """
    Initialize the RAG system by creating a Weaviate collection and storing regulation articles.
    Uses the predefined JSON file path for regulatory data.
    
    Args:
        index_dir: Directory to store metadata
        json_file_path: Path to the JSON file containing regulation data
    """
    print(f"Initializing RAG system from file: {json_file_path} using Weaviate")
    print(f"Index directory: {index_dir}")
    
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"File not found: {json_file_path}")
    
    # 1. Create index directory
    os.makedirs(index_dir, exist_ok=True)
    
    # 2. Load the JSON file
    print(f"Loading JSON file: {json_file_path}...")
    try:
        # Load the entire JSON file at once
        with open(json_file_path, 'r') as f:
            articles = json.load(f)
        
        print(f"Loaded {len(articles)} articles from JSON file.")
        
        if not articles:
            print("Warning: No articles loaded from JSON file.")
            save_metadata(index_dir, {"collection_name": COLLECTION_NAME, "json_file_path": json_file_path, "article_count": 0})
            print("Initialization complete (with empty data).")
            return
            
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        raise
    
    # 3. Connect to Weaviate and create collection
    try:
        client = get_weaviate_client()
        create_weaviate_collection(client, COLLECTION_NAME)
        
        # 4. Import articles into Weaviate
        print(f"Importing {len(articles)} articles into Weaviate collection '{COLLECTION_NAME}'...")
        collection = client.collections.get(COLLECTION_NAME)
        
        # Load the ColBERT model for embeddings
        print("Loading ModernColBERT model...")
        model = get_colbert_model()
        print("ModernColBERT model loaded successfully.")
        
        # Track total points for metadata
        total_points = 0
        
        # Process articles in batches
        with collection.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
            # Process each article
            for article_idx, article in enumerate(articles):
                # Extract article metadata
                article_number = article.get("article", "")
                article_title = article.get("title", "")
                
                # Process each point (paragraph) in the article
                points = article.get("points", {})
                
                point_counter = 0
                for point_id, content in points.items():
                    # Skip empty content
                    if not content or not content.strip():
                        continue
                    
                    try:
                        # Generate the embedding using ModernColBERT
                        embedding = model.encode(content, is_query=False)
                        
                        # Add the article point to Weaviate batch
                        batch.add_object(
                            properties={
                                # Article properties
                                "article_number": article_number,
                                "article_title": article_title,
                                
                                # Point properties
                                "point_id": point_id,
                                "content": content,
                            },
                            vector={
                                "multi_vector": embedding
                            },
                            uuid=generate_uuid5(f"article_{article_number}_point_{point_id}")
                        )
                        
                        point_counter += 1
                        total_points += 1
                    except Exception as e:
                        print(f"Error processing article {article_number}, point {point_id}: {e}")
                        continue
                
                if (article_idx + 1) % 10 == 0:
                    print(f"  Imported {article_idx+1}/{len(articles)} articles ({total_points} total points)...")
        
        # 5. Save metadata
        metadata = {
            "collection_name": COLLECTION_NAME,
            "json_file_path": json_file_path,
            "article_count": len(articles),
            "point_count": total_points
        }
        save_metadata(index_dir, metadata)
        print(f"Initialization complete. Imported {len(articles)} articles with {total_points} total points.")
        
        # 6. Close the client connection when done
        client.close()
        print("Weaviate client connection closed.")
        
    except Exception as e:
        print(f"Error during Weaviate collection creation or import: {e}")
        raise

async def apply_winnowing_filter(enhanced_prompt: str, retrieved_points: List[Dict]) -> List[Dict]:
    """
    Apply a winnowing filter to select the most relevant article points.
    This implementation preserves the original order of points and uses
    asynchronous processing for improved performance.
    
    Args:
        enhanced_prompt: The query prompt enhanced with HyDE output
        retrieved_points: List of dictionaries containing retrieved article points with metadata
        
    Returns:
        List[Dict]: Filtered list of relevant article points in original order
    """
    if not retrieved_points:
        return []
    
    print(f"Applying winnowing filter to {len(retrieved_points)} retrieved points...")
    
    # Get the prompt template for winnowing
    current_dir = Path(__file__).parent.absolute()
    prompt_path = os.path.join(current_dir, 'prompts', 'winnowing.yaml')
    prompt_template = load_prompt_template(prompt_path)
    
    # Prepare article points for the model input, organizing them by article
    # We'll keep a mapping of point_id to original index to preserve order
    article_points = {}
    point_id_to_index = {}
    
    for idx, point in enumerate(retrieved_points):
        article_number = point.get("article_number")
        article_title = point.get("article_title", "")
        point_id = point.get("point_id")
        content = point.get("content")
        
        # Store the original index of each point_id
        point_id_to_index[point_id] = idx
        
        if article_number not in article_points:
            article_points[article_number] = {
                "title": article_title,
                "points": {}
            }
        
        article_points[article_number]["points"][point_id] = content
    
    # Create groups of points for processing, ensuring each group stays under token limit
    # We'll group by articles, but be more careful about token limits
    groups = []
    current_group = []
    current_tokens = 0
    
    for article_number, article_data in article_points.items():
        # Convert this single article to JSON to estimate tokens
        article_json = json.dumps([{"title": article_data["title"], "points": article_data["points"]}])
        article_tokens = len(article_json) // 4  # Rough approximation: 1 token ≈ 4 chars
        
        # If adding this article would exceed token limit and we already have articles in the group
        if current_tokens + article_tokens > MAX_TOKENS_PER_GROUP and current_group:
            # Save current group and start a new one
            groups.append(current_group)
            current_group = [article_number]
            current_tokens = article_tokens
        else:
            # Add to current group
            current_group.append(article_number)
            current_tokens += article_tokens
    
    # Add the last group if it has any articles
    if current_group:
        groups.append(current_group)
    
    print(f"Split retrieval into {len(groups)} groups for winnowing")
    
    # Create tasks for asynchronous processing of each group
    async def process_group(group_idx, article_numbers):
        # Create a subset of articles for this group
        group_articles = []
        for article_number in article_numbers:
            group_articles.append({
                "title": article_points[article_number]["title"],
                "points": article_points[article_number]["points"]
            })
        
        group_articles_json = json.dumps(group_articles)
        
        # Format the prompt with the query+hyde context and articles
        formatted_user_prompt = prompt_template.get("user_prompt", "").format(
            query=enhanced_prompt,
            articles=group_articles_json
        )
        
        # Create the full prompt with system message
        system_prompt = prompt_template.get("system_prompt", "")
        
        # Call the LLM with the formatted prompt
        winnowing_response = await open_router_request(
            message=formatted_user_prompt,
            model=WINNOWING_MODEL,
            system_message=system_prompt
        )
        
        # Extract the list of relevant point IDs from the response
        relevant_point_ids = []
        try:
            # Extract the point IDs using a regular expression
            import re
            useful_points_match = re.search(r'Useful points:\s*(\[.*?\])', winnowing_response, re.DOTALL)
            
            if useful_points_match:
                useful_points_str = useful_points_match.group(1)
                try:
                    useful_points = json.loads(useful_points_str)
                    relevant_point_ids = useful_points
                    print(f"Found {len(useful_points)} relevant points in group {group_idx+1}")
                except json.JSONDecodeError:
                    print(f"Error parsing JSON for useful points in group {group_idx+1}")
            else:
                print(f"No useful points found in winnowing response for group {group_idx+1}")
        except Exception as e:
            print(f"Error processing winnowing response: {e}")
        
        return relevant_point_ids
    
    # Process all groups in parallel
    import asyncio
    tasks = [process_group(i, article_numbers) for i, article_numbers in enumerate(groups)]
    results = await asyncio.gather(*tasks)
    
    # Combine all relevant point IDs from all groups
    all_relevant_point_ids = [point_id for group_result in results for point_id in group_result]
    
    # Filter the original points but maintain original ordering
    # First, get the indices of points that have relevant point_ids
    relevant_indices = sorted([
        point_id_to_index[point_id] 
        for point_id in all_relevant_point_ids 
        if point_id in point_id_to_index
    ])
    
    # Use these indices to extract points in original order
    filtered_points = [retrieved_points[i] for i in relevant_indices]
    
    print(f"Winnowing completed. Reduced from {len(retrieved_points)} to {len(filtered_points)} relevant points.")
    return filtered_points

async def retrieve_context_with_hyde(client: weaviate.WeaviateClient, prompt: str, collection_name: str = COLLECTION_NAME, k: int = DEFAULT_RETRIEVAL_K) -> tuple:
    """
    Retrieve the top-k relevant article points for a prompt using HyDE technique.
    First generates a hypothetical answer to enhance the query before retrieval,
    then applies winnowing to filter the most relevant article points.
    
    Args:
        client: Weaviate client
        prompt: The query prompt
        collection_name: Name of the collection to search
        k: Number of article points to retrieve
        
    Returns:
        tuple: (
            str: Concatenated context from retrieved article points with article metadata,
            list: List of retrieved article references in format "Article X (Point Y)"
        )
    """
    print(f"\nProcessing prompt: '{prompt}'")
    
    # Generate a hypothetical answer using Hyde
    hyde_answer = await generate_hyde_answer(prompt)
    
    # Combine original query with hypothetical answer if available
    enhanced_prompt = prompt
    if hyde_answer:
        enhanced_prompt = f"{prompt}\n{hyde_answer}"
        print(f"Enhanced prompt with hypothetical answer")
    
    print(f"Retrieving context for enhanced prompt")
    
    try:
        collection = client.collections.get(collection_name)
        
        # Load the ColBERT model for query embedding
        model = get_colbert_model()
        
        # Generate query embedding for the enhanced prompt
        query_embedding = model.encode(enhanced_prompt, is_query=True)
        
        # Perform a vector search with the query embedding using near_vector for multi-vector search
        # This matches the example in the notebook that uses MaxSim approach
        results = collection.query.near_vector(
            near_vector=query_embedding,  # Just pass the token embeddings directly
            target_vector="multi_vector",  # Specify the target vector field
            limit=k,
            return_properties=[
                "article_number", 
                "article_title", 
                "point_id", 
                "content"
            ]
        )
        
        # Format Context
        if not results.objects:
            print("No relevant article points retrieved.")
            return "", []
        
        # Extract content and metadata from each object
        retrieved_points = []
        for obj in results.objects:
            properties = obj.properties
            if "content" in properties:
                retrieved_points.append({
                    "article_number": properties.get("article_number", ""),
                    "article_title": properties.get("article_title", ""),
                    "point_id": properties.get("point_id", ""),
                    "content": properties.get("content", "")
                })
        
        print(f"Retrieved {len(retrieved_points)} article points initially.")
        
        # Apply winnowing filter to select the most relevant points - only passing enhanced prompt
        filtered_points = await apply_winnowing_filter(enhanced_prompt, retrieved_points)
        
        # If no points remain after winnowing, use the top 50 points from the original retrieval
        if not filtered_points:
            print("No points remained after winnowing. Using top 50 points from original retrieval as fallback.")
            filtered_points = retrieved_points[:50]  # Take only the top 50 points
        
        # Format the filtered points for context and article references
        formatted_chunks = []
        retrieved_articles = []
        
        for point in filtered_points:
            article_number = point["article_number"]
            article_title = point["article_title"]
            point_id = point["point_id"]
            content = point["content"]
            
            # Add formatted article reference and content
            header = f"Article {article_number}"
            if article_title:
                header += f" - {article_title}"
            if point_id:
                header += f" (Point {point_id})"
            
            # Add to retrieved articles list
            article_ref = f"Article {article_number}"
            if point_id:
                article_ref += f" (Point {point_id})"
            retrieved_articles.append(article_ref)
            
            formatted_chunk = f"{header}\n\n{content}"
            formatted_chunks.append(formatted_chunk)
        
        context = "\n\n---\n\n".join(formatted_chunks)
        print(f"Final context includes {len(formatted_chunks)} filtered article points.")
        return context, retrieved_articles
        
    except Exception as e:
        print(f"Error during context retrieval: {e}")
        return "", []

async def call_llm(prompt: str, context: str) -> str:
    """
    Call a large language model using OpenRouter with custom prompt template.
    
    Args:
        prompt: The user query/prompt
        context: The retrieved context to include in the prompt
        
    Returns:
        str: The answer generated by the LLM
    """
    # Get current file's directory and construct path to prompt template
    current_dir = Path(__file__).parent.absolute()
    prompt_path = os.path.join(current_dir, 'prompts', 'answer_generation.yaml')
    
    # Load prompt template
    prompt_template = load_prompt_template(prompt_path)
    
    # Format the user prompt with question and context
    formatted_user_prompt = prompt_template.get("user_prompt", "").format(
        question=prompt,
        context=context
    )
    
    # Create the full prompt with system message
    system_prompt = prompt_template.get("system_prompt", "")
    
    # Call the LLM with the formatted prompt using our local utility
    # Make sure parameters match the function signature in openrouter_request.py
    final_answer = await open_router_request(
        message=formatted_user_prompt,
        model=ANSWER_MODEL,
        system_message=system_prompt
    )
    
    return final_answer

async def answer(questions: List[str], index_dir: str, k: int = DEFAULT_RETRIEVAL_K) -> List[RAGResult]:
    """
    Generate answers for a list of questions using Weaviate for retrieval with HyDE and LLM for generation.
    Uses Hypothetical Document Embeddings (HyDE) to improve retrieval quality.
    
    Args:
        questions: List of questions to answer
        index_dir: Directory containing metadata about the Weaviate collection
        k: Number of chunks to retrieve for each question
        
    Returns:
        List[RAGResult]: List of answers with retrieved article references
    """
    # 1. Load metadata
    metadata = load_metadata(index_dir)
    if not metadata:
        print(f"Error: No metadata found in {index_dir}. Please run init() first.")
        return [RAGResult(answer="I don't have access to the required information.", retrieved_articles=["No metadata"]) 
                for _ in questions]
    
    collection_name = metadata.get("collection_name", COLLECTION_NAME)
    
    # 2. Connect to Weaviate
    try:
        client = get_weaviate_client()
        
        # Check if collection exists
        if not client.collections.exists(collection_name):
            print(f"Error: Collection '{collection_name}' does not exist in Weaviate.")
            return [RAGResult(answer="I don't have access to the required information.", retrieved_articles=[]) 
                    for _ in questions]
        
        # Load the ColBERT model once for all questions
        print("Loading ModernColBERT model for all questions...")
        model = get_colbert_model()
        
        results = []
        for question in questions:
            # 3. Retrieve Context with article references using HyDE
            context, retrieved_articles = await retrieve_context_with_hyde(client, question, collection_name, k)
            
            # 4. Call LLM
            final_answer = await call_llm(question, context)
            
            # 5. Append to results with retrieved article references
            results.append(RAGResult(answer=final_answer, retrieved_articles=retrieved_articles))
        
        # Close the client connection when done
        client.close()
        print("Weaviate client connection closed.")
        
        return results
        
    except Exception as e:
        print(f"Error connecting to Weaviate or processing questions: {e}")
        return [RAGResult(answer=f"An error occurred: {str(e)}", retrieved_articles=["Error"]) 
                for _ in questions]

# --- Example Usage ---
if __name__ == "__main__":
    # IMPORTANT: Ensure a Weaviate instance is running before running this code!
    # Start Weaviate with Docker using:
    # docker run --detach -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.30.1
    
    index_path = implementation_system_path("ColBERT")
    
    # Initialize with Weaviate using the regulation JSON file
    print("\n--- Step 1: Initializing (Weaviate with Regulation Data) ---")
    try:
        init(index_path)
    except Exception as e:
        print(f"\nInitialization failed: {e}")
        
        # Check if it's a connection error
        if "Connection refused" in str(e) or "Failed to connect" in str(e):
            print("\nERROR: Could not connect to Weaviate. Please ensure Weaviate is running with:")
            print("docker run --detach -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.30.1")
        else:
            print("\nPlease ensure dependencies are installed:")
            print("pip install weaviate-client torch transformers numpy")
        exit(1)
