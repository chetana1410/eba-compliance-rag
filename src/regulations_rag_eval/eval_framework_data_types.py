from typing import TypedDict, List, Dict, NamedTuple, Literal, Union

Hash12 = str # 12 character hash
Origin = Literal['AI', 'HUMAN'] # origin of the question
QuestionCategory = Literal['POINT_FACT', 'SYNTHESIS'] # category of the question

class RegulationQuestion(TypedDict):
    question_hash: Hash12 
    question_text: str  
    regulation: str
    category: str
    origin: str
    article_paths: List[str]

class GoldenAnswer(TypedDict):
    question_hash: Hash12
    answer_text: str
    article_paths: List[str]
    origin: Origin 

class GeneratedAnswer(TypedDict):
    question_hash: Hash12 
    answer_hash: Hash12 
    answer_text: str
    retrieved_article_paths: List[str]
    rag_implementation_code: str
    run_id: str
    run_timestamp: str

class DetailedMetrics(TypedDict):
    clarity_score: int
    completeness_score: int
    correctness_score: int
    overall_score: int
    relevance_rationale: str
    clarity_rationale: str
    completeness_rationale: str
    correctness_rationale: str
    overall_rationale: str

class AnswerEvaluation(TypedDict):
    question_hash: Hash12 
    answer_hash: Hash12 
    evaluation: int
    evaluation_timestamp: str
    evaluation_llm_model: str
    detailed_metrics: DetailedMetrics

class RetrievalEvaluation(TypedDict):
    question_hash: Hash12 
    retrieved_article_paths: List[str]
    coverage_score: float
    relevance_score: float
    evaluation_timestamp: str

class SampledAnswer(NamedTuple):
    """
    Named tuple to represent a sampled answer.
    
    Attributes:
    """
    question: str
    golden_answer: str
    answer: str

class AnswerEvaluationResult(NamedTuple):
    """
    Named tuple to represent the result of an answer evaluation.
    
    Attributes:
        grade (int): The overall grade of the evaluation (1-4 scale)
        clarity_score (int): Score for clarity (0-15 scale)
        completeness_score (int): Score for completeness (0-35 scale)
        correctness_score (int): Score for correctness (0-50 scale)
        overall_score (int): Sum of all scores (0-100 scale)
        relevance_rationale (str): Rationale for relevance assessment
        clarity_rationale (str): Rationale for clarity assessment
        completeness_rationale (str): Rationale for completeness assessment
        correctness_rationale (str): Rationale for correctness assessment
        overall_rationale (str): Overall evaluation rationale
    """
    grade: int
    clarity_score: int
    completeness_score: int
    correctness_score: int
    overall_score: int
    relevance_rationale: str
    clarity_rationale: str
    completeness_rationale: str
    correctness_rationale: str
    overall_rationale: str
