from typing import List, Dict, Any
from ...rag_implementation_data_types import RAGResult
from .colbert import answer
from ..paths import implementation_system_path

async def generate_answers(questions: List[str], implementation_name: str, params: Dict[str, Any]) -> List[RAGResult]:
    """
    Implementation of Reason ColBERT to answer questions using only CRR 13 as the data source.

    :param questions: List of questions to answwrt
    :param implementation_name: Name of the implementation
    :param params: Any other params to pass to the model
    :return: List of RAGResult types which contains the final answer along with citations.
    """

    return await answer(questions, implementation_system_path(implementation_name))