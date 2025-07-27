from ..utils import prompt_escape


def llm_as_judge_prompt(
    question: str,
    golden_answer: str,
    answer: str,
) -> str:
    """
    Generates a prompt for the LLM to evaluate an answer based on a question, reference, and feedback.
    This includes detailed metrics for clarity, completeness, correctness, and overall quality.

    Returns:
        str: The formatted prompt for the LLM.
    """
    escaped_question = prompt_escape(question)
    escaped_golden_answer = prompt_escape(golden_answer)
    escaped_answer = prompt_escape(answer)
    return f"""
I will provide you with two answers to a question. One is the Ground Truth Answer, which serves as the benchmark. The other is the Generated Answer, which needs to be evaluated against the Ground Truth Answer.

Your task is to evaluate the Generated Answer based on several dimensions and provide a JSON output with detailed scores and rationales.

Evaluate using these criteria:
1. CLARITY (15 points): How clear, well-structured, and easily understood is the Generated Answer?
   - Score from 0-15 points
   - Consider organization, wording clarity, lack of ambiguity, and logical flow

2. COMPLETENESS (35 points): How completely does the Generated Answer cover the information in the Ground Truth Answer?
   - Score from 0-35 points
   - The Ground Truth Answer should be split into components which can be checked for their presence in the Generated Answer
   - Score higher when more components from the Ground Truth are present in the Generated Answer

3. CORRECTNESS (50 points): How accurate is the Generated Answer compared to the Ground Truth Answer?
   - Score from 0-50 points
   - Consider factual accuracy, absence of contradictions with the Ground Truth, and logical consistency
   - Deduct points for statements that contradict the Ground Truth

4. OVERALL SCORE: Sum of the above scores (out of 100 points)

5. Traditional evaluation grade on a 1-4 scale:
   - RATING 1: The Generated Answer is completely incorrect and incomplete compared to the Ground Truth Answer
   - RATING 2: The Generated Answer is incorrect but either complete or partially complete. It contains some useful information found in the Ground Truth Answer but the main statement is incorrect
   - RATING 3: The Generated Answer is correct but only partially complete. The main statement matches the Ground Truth Answer, but some information from the Ground Truth Answer is missing
   - RATING 4: The Generated Answer is fully correct and complete. It is essentially a rephrased version of the Ground Truth Answer with no significant differences

Question:
{escaped_question}

Ground Truth Answer:
{escaped_golden_answer}

Generated Answer:
{escaped_answer}

Output your evaluation as a JSON object with the following structure:
{{
  "traditional_grade": 1-4,
  "clarity_score": 0-15,
  "completeness_score": 0-35,
  "correctness_score": 0-50,
  "overall_score": 0-100,
  "relevance_rationale": "Brief explanation of the answer's relevance to the question",
  "clarity_rationale": "Brief explanation of clarity score",
  "completeness_rationale": "Brief explanation of completeness score, including which components from the Ground Truth are missing",
  "correctness_rationale": "Brief explanation of correctness score, noting any inaccuracies",
  "overall_rationale": "Brief summary of overall evaluation, highlighting key strengths and weaknesses"
}}

Remember to be thorough and fair in your assessment.
"""
