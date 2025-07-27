import importlib
import os
import sys
import json
from typing import Callable, List, Tuple, Dict, Any, Awaitable
from ..rag_implementations import RAGResult

project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_rag_impl_entry_point(implementation_name: str) -> Tuple[Callable[[List[str], str, Dict[str, Any]], Awaitable[List[RAGResult]]], Dict[str, Any]]:
    # Initialize options if not provided
    options = {}
        
    # Check for configuration mapping
    config_path = os.path.join("config", "rag_implementation_mappings.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                
            # If implementation name exists in config, update the name and merge options
            if implementation_name in config:
                config_entry = config[implementation_name]
                if isinstance(config_entry, dict):
                    if "implementation" in config_entry:
                        implementation_name = config_entry["implementation"]
                    if "options" in config_entry:
                        # Merge options, with passed options taking precedence
                        options = config_entry["options"]
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load configuration: {e}")
    
    try:
        # Dynamically import the 'runner' module from the specific implementation package
        module_path = f"regulations_rag_eval.rag_implementations.{implementation_name}.generate_answers"
        implementation_module = importlib.import_module(module_path)

        # Assume the inner function is always named 'run_rag_query'
        if not hasattr(implementation_module, 'generate_answers'):
            raise AttributeError(f"'generate_answers' function not found in {module_path}")

        inner_evaluate_func = implementation_module.generate_answers    

    except ModuleNotFoundError:
        print(f"Error: Implementation '{implementation_name}' not found or missing 'get_rag_impl_entry_point.py'.")
        sys.exit(1)
    except AttributeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    return inner_evaluate_func, options

