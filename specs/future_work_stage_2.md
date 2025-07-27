#FOR FURURE WORK NOT NOW


# signature for the evaluation function
async def evaluate_answers(List[SampledAnswer]) -> List[AnswerEvaluation]:
    """
    Evaluate generated answers against golden answers.
    
    Args:
        sampled_answers (List[SampledAnswer]): List of sampled answers to be evaluated
    
    Returns:
        List[AnswerEvaluationResult]: List of evaluation results.
    """
    pass 

  
# signature for the retrieval evaluation function
async def evaluate_retrieval(questions: List[RegulationQuestion], generated_answers: List[GeneratedAnswer]) -> List[RetrievalEvaluation]:
    """
    Evaluate the retrieval process for generated answers.
    
    Args:
        questions (List[RegulationQuestion]): List of questions to be evaluated.
        generated_answers (List[GeneratedAnswer]): List of generated answers to be evaluated.
    
    Returns:
        List[RetrievalEvaluation]: List of retrieval evaluation results.
    """
    pass



