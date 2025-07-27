import httpx
import time
import logging
import asyncio
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def open_router_request(
    message: str,
    model: str,
    system_message: str = "",
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    stop_sequences: Optional[list] = None,
    additional_params: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
    retry_delay: float = 1.0 , 
    seed = 42
) -> str:
    """
    Makes a request to the OpenRouter API with customizable parameters and retry logic.
    
    Args:
        message: The message to send to the model
        model: The model identifier to use
        system_message: The system message to set context for the conversation
        temperature: Controls randomness. Higher values (e.g., 0.8) make output more random, 
                     lower values (e.g., 0.2) make it more deterministic. Default: 0.7
        top_k: Limits token selection to k most probable tokens. Default: None
        top_p: Nucleus sampling - consider the smallest set of tokens whose probability sum is at least top_p. Default: None
        max_tokens: Maximum number of tokens to generate. Default: None
        frequency_penalty: Penalizes frequent tokens. Default: None
        presence_penalty: Penalizes repeated tokens. Default: None
        stop_sequences: List of sequences where the API will stop generating further tokens. Default: None
        additional_params: Any additional parameters to pass to the API. Default: None
        
    Returns:
        The response text from the model
    """
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": message
            }
        ],
        "temperature": temperature
    }
    
    # Add optional parameters if provided
    if top_k is not None:
        data["top_k"] = top_k
    if top_p is not None:
        data["top_p"] = top_p
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    # if frequency_penalty is not None:
    #     data["frequency_penalty"] = frequency_penalty
    # if presence_penalty is not None:
    #     data["presence_penalty"] = presence_penalty
    if stop_sequences is not None:
        data["stop"] = stop_sequences
    
    # Add any additional parameters
    if additional_params:
        data.update(additional_params)

    data["seed"] = seed  # Add seed to the request data
    
    # Initialize retry counter
    retry_count = 0
    last_exception = None
    
    # Retry loop
    while retry_count <= max_retries:
        try:
            async with httpx.AsyncClient() as client:
                logger.info(f"Making request to OpenRouter API (attempt {retry_count + 1}) to {model}")
                response = await client.post(
                    url=url,
                    headers=headers,
                    json=data,
                    timeout=60.0  # Add a reasonable timeout
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                logger.info(f"OpenRouter API request successful on attempt {retry_count + 1} to {model}")
                return response_data["choices"][0]["message"]["content"]
                
        except httpx.HTTPStatusError as e:
            last_exception = e
            status_code = e.response.status_code
            
            # For certain status codes, don't retry
            if status_code in (400, 401, 403):  # Bad request, unauthorized, forbidden
                logger.error(f"OpenRouter API error (status {status_code}): {e}")
                raise
                
            logger.warning(f"OpenRouter API HTTP error (status {status_code}): {e}. Retrying ({retry_count + 1}/{max_retries + 1})")
            
        except (httpx.RequestError, httpx.TimeoutException) as e:
            last_exception = e
            logger.warning(f"OpenRouter API request error: {e}. Retrying ({retry_count + 1}/{max_retries + 1})")
            
        except Exception as e:
            last_exception = e
            logger.warning(f"Unexpected error during OpenRouter API request: {e}. Retrying ({retry_count + 1}/{max_retries + 1})")
        
        # If we've reached the maximum number of retries, raise the last exception
        if retry_count >= max_retries:
            logger.error(f"OpenRouter API request failed after {max_retries + 1} attempts")
            raise last_exception
        
        # Increase retry delay with each attempt (exponential backoff)
        current_delay = retry_delay * (2 ** retry_count)
        logger.info(f"Waiting {current_delay:.2f} seconds before retrying...")
        await asyncio.sleep(current_delay)
        
        retry_count += 1
    
    # This should never be reached, but just in case
    raise RuntimeError(f"OpenRouter API request failed after {max_retries + 1} attempts")
