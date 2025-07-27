
from typing import NamedTuple, List

class RAGResult(NamedTuple):
    """
    Named tuple to represent a result of RAG generation.
    
    Attributes:
        answer (str): The generated answer.
        retrieved_articles (List[str]): List of article paths retrieved for generating the answer.
    """
    answer: str
    retrieved_articles: List[str]





