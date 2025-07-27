import os
import httpx
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


async def open_router_request(message: str | List[Dict], model: str, additional_params: Optional[Dict] = None) -> str:
    """
    Makes a request to the OpenRouter API.
    
    Args:
        message: The message to send to the model
        model: The model identifier to use
        additional_params: Optional dictionary of additional parameters to include in the request
        
    Returns:
        The response text from the model
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ]
    }
    
    # Merge additional parameters if provided
    if additional_params:
        data.update(additional_params)
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            url=url,
            headers=headers,
            json=data
        )
        
        response.raise_for_status()
        response_data = response.json()
        
        return response_data["choices"][0]["message"]["content"]