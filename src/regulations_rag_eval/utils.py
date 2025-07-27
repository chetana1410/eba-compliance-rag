
import json
from typing import Any, Set, Optional


def prompt_escape(s: str) -> str:
    return repr(s)[1:-1].replace('{', '{{').replace('}', '}}') 


def load_questions_filter(questions_filter_file: Optional[str]) -> Optional[Set[str]]:
    """
    Load question hashes from a filter file.
    
    Args:
        questions_filter_file: Path to file containing question hashes (one per line)
        
    Returns:
        Set of question hashes if file provided and readable, None otherwise
    """
    if not questions_filter_file:
        return None
        
    try:
        with open(questions_filter_file, "r", encoding="utf-8") as f:
            filter_hashes = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(filter_hashes)} question hashes from filter file")
        return filter_hashes
    except (IOError, OSError) as e:
        print(f"Warning: Could not read questions filter file: {e}")
        return None
    

def load_questions_json(questions_file: Optional[str]) -> Any:
    """
    Load questions from a JSON file.
    
    Args:
        questions_file: Path to the JSON file containing questions
        
    Returns:
        Dictionary of questions if file provided and readable, None otherwise
    """
    if not questions_file:
        return None

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    questions = [q for q in questions if not q.get("retired", False)]
    return questions
