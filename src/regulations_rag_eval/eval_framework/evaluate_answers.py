import os
from typing import List, Dict
import json
from datetime import datetime
import re

from ..eval_framework_data_types import (
    SampledAnswer, 
    GeneratedAnswer, 
    RegulationQuestion, 
    GoldenAnswer,
    AnswerEvaluationResult,
    AnswerEvaluation
)
from .llm_as_judge_prompt import llm_as_judge_prompt
from ..open_router_request import open_router_request
from ..calculate_report import calculate_and_update_report
from ..utils import load_questions_filter, load_questions_json


def load_existing_evaluations(evaluation_output_file_path: str) -> Dict[str, AnswerEvaluation]:
    """Load existing evaluations and return a mapping of answer_hash -> AnswerEvaluation"""
    if not os.path.exists(evaluation_output_file_path):
        return {}
    
    try:
        with open(evaluation_output_file_path, "r", encoding="utf-8") as f:
            existing_evaluations = json.load(f)
        return {evaluation["answer_hash"]: evaluation for evaluation in existing_evaluations}
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load existing evaluations: {e}")
        return {}


def filter_unevaluated_answers(
    generated_answers: List[GeneratedAnswer], 
    existing_evaluations: Dict[str, AnswerEvaluation]
) -> tuple[List[GeneratedAnswer], List[GeneratedAnswer]]:
    """
    Separate answers into evaluated and unevaluated.
    Returns: (unevaluated_answers, evaluated_answers)
    """
    unevaluated = []
    evaluated = []
    
    for answer in generated_answers:
        if answer["answer_hash"] in existing_evaluations:
            evaluated.append(answer)
        else:
            unevaluated.append(answer)
    
    return unevaluated, evaluated


def merge_evaluations(
    new_evaluations: List[AnswerEvaluation], 
    existing_evaluations: Dict[str, AnswerEvaluation],
    regulation_questions: List[RegulationQuestion]
) -> List[AnswerEvaluation]:
    """Merge new evaluations with existing ones, preserving existing evaluations"""
    merged = list(existing_evaluations.values())  # Start with existing evaluations
    merged.extend(new_evaluations)  # Add new evaluations
    
    # Create mapping of question_hash to original question order
    question_order = {q["question_hash"]: i for i, q in enumerate(regulation_questions)}
    
    # Sort by original question order
    merged.sort(key=lambda x: question_order.get(x["question_hash"], float('inf')))
    return merged


async def evaluate_single_answer(answer: SampledAnswer, model: str) -> AnswerEvaluationResult:
    """
    Evaluates a single answer using the LLM as a judge with detailed metrics.
    
    Args:
        answer: A SampledAnswer object containing question, golden answer, and generated answer
        model: The model identifier to use for evaluation
        
    Returns:
        An AnswerEvaluationResult object containing the evaluation grades and detailed metrics
    """
    try:
        prompt = llm_as_judge_prompt(
            question=answer.question,
            golden_answer=answer.golden_answer,
            answer=answer.answer
        )
        
        response = await open_router_request(message=prompt, model=model)
        
        # Extract JSON from response
        try:
            # Find JSON object in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                evaluation_data = json.loads(json_str)
                
                # Extract metrics from JSON
                grade = int(evaluation_data.get("traditional_grade", 0))
                clarity_score = int(evaluation_data.get("clarity_score", 0))
                completeness_score = int(evaluation_data.get("completeness_score", 0))
                correctness_score = int(evaluation_data.get("correctness_score", 0))
                overall_score = int(evaluation_data.get("overall_score", 0))
                
                # Extract rationales
                relevance_rationale = evaluation_data.get("relevance_rationale", "")
                clarity_rationale = evaluation_data.get("clarity_rationale", "")
                completeness_rationale = evaluation_data.get("completeness_rationale", "")
                correctness_rationale = evaluation_data.get("correctness_rationale", "")
                overall_rationale = evaluation_data.get("overall_rationale", "")
                
                # Create result with detailed metrics
                return AnswerEvaluationResult(
                    grade=grade,
                    clarity_score=clarity_score,
                    completeness_score=completeness_score,
                    correctness_score=correctness_score,
                    overall_score=overall_score,
                    relevance_rationale=relevance_rationale,
                    clarity_rationale=clarity_rationale,
                    completeness_rationale=completeness_rationale,
                    correctness_rationale=correctness_rationale,
                    overall_rationale=overall_rationale
                )
            else:
                # Fallback to basic extraction if JSON is not found
                print(f"Warning: Could not find JSON in response. Using fallback method.")
                match = re.search(r'\d', response[:15])
                grade = int(match.group()) if match else 0
                
                # Create result with default values
                return AnswerEvaluationResult(
                    grade=grade,
                    clarity_score=0,
                    completeness_score=0,
                    correctness_score=0,
                    overall_score=0,
                    relevance_rationale="Error extracting detailed metrics",
                    clarity_rationale="Error extracting detailed metrics",
                    completeness_rationale="Error extracting detailed metrics",
                    correctness_rationale="Error extracting detailed metrics",
                    overall_rationale="Error extracting detailed metrics"
                )
        except Exception as e:
            print(f"Error parsing evaluation response: {e}")
            print(f"Response: {response[:100]}...")
            
            # Try fallback method if JSON parsing fails
            try:
                match = re.search(r'\d', response[:15])
                grade = int(match.group()) if match else 0
            except (ValueError, IndexError):
                # Even more basic fallback
                if " 1" in response:
                    grade = 1
                elif " 2" in response:
                    grade = 2
                elif " 3" in response:
                    grade = 3
                elif " 4" in response:
                    grade = 4
                else:
                    grade = 0  # Indicate error in parsing
            
            # Create result with error information
            return AnswerEvaluationResult(
                grade=grade,
                clarity_score=0,
                completeness_score=0,
                correctness_score=0,
                overall_score=0,
                relevance_rationale=f"Error parsing response: {str(e)[:100]}",
                clarity_rationale="Error extracting detailed metrics",
                completeness_rationale="Error extracting detailed metrics",
                correctness_rationale="Error extracting detailed metrics",
                overall_rationale="Error extracting detailed metrics"
            )
    except Exception as e:
        print(f"Error evaluating answer: {e}")
        # Return error result
        return AnswerEvaluationResult(
            grade=0,
            clarity_score=0,
            completeness_score=0,
            correctness_score=0,
            overall_score=0,
            relevance_rationale=f"Error evaluating answer: {str(e)[:100]}",
            clarity_rationale="Error evaluating answer",
            completeness_rationale="Error evaluating answer",
            correctness_rationale="Error evaluating answer",
            overall_rationale="Error evaluating answer"
        )


async def evaluate_answers(answers: List[SampledAnswer], model: str) -> List[AnswerEvaluationResult]:
    """
    Evaluates a list of answers using the LLM as a judge with detailed metrics.
    Processes answers in parallel batches of 10 for improved performance.
    
    Args:
        answers: A list of SampledAnswer objects containing questions, golden answers, and generated answers.
        model: The model identifier to use for evaluation
        
    Returns:
        A list of AnswerEvaluationResult objects containing the evaluation grades and detailed metrics.
    """
    import asyncio
    
    results = []
    batch_size = 10
    
    print(f"Evaluating {len(answers)} answers in batches of {batch_size}...")
    
    # Process answers in batches of 10
    for i in range(0, len(answers), batch_size):
        batch = answers[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(answers) + batch_size - 1)//batch_size} ({len(batch)} answers)...")
        
        # Create tasks for this batch
        tasks = [evaluate_single_answer(answer, model) for answer in batch]
        
        # Execute batch in parallel
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions and add results
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"Error in batch evaluation: {result}")
                # Create error result
                error_result = AnswerEvaluationResult(
                    grade=0,
                    clarity_score=0,
                    completeness_score=0,
                    correctness_score=0,
                    overall_score=0,
                    relevance_rationale=f"Batch evaluation error: {str(result)[:100]}",
                    clarity_rationale="Batch evaluation error",
                    completeness_rationale="Batch evaluation error",
                    correctness_rationale="Batch evaluation error",
                    overall_rationale="Batch evaluation error"
                )
                results.append(error_result)
            else:
                results.append(result)
    
    print(f"Completed evaluation of {len(results)} answers.")
    return results


# Keep the old sequential version for reference/fallback
async def evaluate_answers_sequential(answers: List[SampledAnswer], model: str) -> List[AnswerEvaluationResult]:
    """
    Sequential version of evaluate_answers for fallback if needed.
    """
    results = []
    
    for answer in answers:
        result = await evaluate_single_answer(answer, model)
        results.append(result)
    
    return results


async def evaluate(
    generated_answers: List[GeneratedAnswer], 
    regulation_questions: List[RegulationQuestion], 
    golden_answers: List[GoldenAnswer],
    model: str = "openai/gpt-4o"
) -> List[AnswerEvaluation]:
    """
    Evaluates a list of generated answers against golden answers.
    
    Args:
        generated_answers: List of generated answers to evaluate
        regulation_questions: List of regulation questions
        golden_answers: List of golden answers
        model: The model identifier to use for evaluation
        
    Returns:
        A list of AnswerEvaluation objects with evaluation results
    """
    # Create lookup dictionaries for questions and golden answers by hash
    question_lookup: Dict[str, str] = {q["question_hash"]: q["question_text"] for q in regulation_questions}
    golden_answer_lookup: Dict[str, str] = {g["question_hash"]: g["answer_text"] for g in golden_answers}
    
    # Create SampledAnswer objects for each generated answer
    sampled_answers = []
    generated_answer_map = []  # To keep track of which generated answer corresponds to which sampled answer
    
    for generated_answer in generated_answers:
        question_hash = generated_answer["question_hash"]
        
        # Skip if we don't have matching question or golden answer
        if question_hash not in question_lookup or question_hash not in golden_answer_lookup:
            continue
            
        sampled_answers.append(
            SampledAnswer(
                question=question_lookup[question_hash],
                golden_answer=golden_answer_lookup[question_hash],
                answer=generated_answer["answer_text"]
            )
        )
        generated_answer_map.append(generated_answer)
    
    # Evaluate the sampled answers
    evaluation_results = await evaluate_answers(sampled_answers, model)
    
    # Create AnswerEvaluation objects from the results
    current_timestamp = datetime.now().isoformat()
    
    answer_evaluations = []
    for i, result in enumerate(evaluation_results):
        # Get the corresponding generated answer
        generated_answer = generated_answer_map[i]
        
        answer_evaluations.append({
            "question_hash": generated_answer["question_hash"],
            "answer_hash": generated_answer["answer_hash"],
            "evaluation": result.grade,
            "evaluation_timestamp": current_timestamp,
            "evaluation_llm_model": model,
            "detailed_metrics": {
                "clarity_score": result.clarity_score,
                "completeness_score": result.completeness_score,
                "correctness_score": result.correctness_score,
                "overall_score": result.overall_score,
                "relevance_rationale": result.relevance_rationale,
                "clarity_rationale": result.clarity_rationale,
                "completeness_rationale": result.completeness_rationale,
                "correctness_rationale": result.correctness_rationale,
                "overall_rationale": result.overall_rationale
            }
        })
    
    return answer_evaluations


async def generate_evaluate_with_files(
    answers_file_path: str, 
    questions_file_path: str | None = None, 
    golden_answer_file_path: str | None = None, 
    evaluation_output_file_path: str | None = None,
    model: str = "openai/gpt-4o",
    incremental: bool = False,
    valid: bool = False,
    questions_filter_file: str | None = None
) -> None:
    """
    Loads data from input files, evaluates answers, and writes evaluation results to an output file.
    
    Args:
        answers_file_path: Path to the JSON file containing generated answers
        questions_file_path: Path to the JSON file containing regulation questions
        golden_answer_file_path: Path to the JSON file containing golden answers
        evaluation_output_file_path: Path to write the evaluation results
        model: The model identifier to use for evaluation
        incremental: Skip answers that already have evaluations
        valid: Only evaluate answers with non-empty retrieved_article_paths
        questions_filter_file: Path to file containing question hashes to filter (one per line)
        
    Returns:
        None
    """
    questions_file_path = questions_file_path or os.getenv("DATA_PATH") + "/qa_datasets/regulation_questions.json" # type: ignore
    golden_answer_file_path = golden_answer_file_path or os.getenv("DATA_PATH") + "/qa_datasets/golden_answers.json" # type: ignore


    # Ensure answers_file_path is absolute
    if not os.path.isabs(answers_file_path):
        answers_file_path = os.path.join(os.getenv("DATA_PATH"), answers_file_path) # type: ignore

    # TODO base filename on filename in answers_file_path, to make more unique 
    if not evaluation_output_file_path:
        evaluation_output_file_path = os.path.join(
            os.path.dirname(answers_file_path), "answer_evaluations.json"
        )

    # Load input files
    with open(answers_file_path, 'r', encoding='utf-8') as f:
        all_generated_answers = json.load(f)
    
    # Apply question filter if provided
    filter_hashes = load_questions_filter(questions_filter_file)
    if filter_hashes:
        all_generated_answers = [
            answer for answer in all_generated_answers 
            if answer["question_hash"] in filter_hashes
        ]
        print(f"Filtered to {len(all_generated_answers)} answers based on filter file")
    
    # Filter for valid answers if valid flag is set
    if valid:
        all_generated_answers = [
            answer for answer in all_generated_answers 
            if answer.get("retrieved_article_paths") and len(answer["retrieved_article_paths"]) > 0
        ]
        print(f"Filtered to {len(all_generated_answers)} answers with valid retrieved article paths")
    
    regulation_questions = load_questions_json(questions_file_path) 
    
    with open(golden_answer_file_path, 'r', encoding='utf-8') as f:
        golden_answers = json.load(f)
    
    # Handle incremental processing
    if incremental:
        existing_evaluations = load_existing_evaluations(evaluation_output_file_path)
        answers_to_evaluate, already_evaluated = filter_unevaluated_answers(all_generated_answers, existing_evaluations)
        
        print(f"Total answers: {len(all_generated_answers)}")
        print(f"Already evaluated: {len(already_evaluated)}")
        print(f"Answers to evaluate: {len(answers_to_evaluate)}")
        
        if not answers_to_evaluate:
            print("All answers already have evaluations. Use without --incremental to re-evaluate.")
            return
        
        generated_answers = answers_to_evaluate
    else:
        existing_evaluations = {}
        generated_answers = all_generated_answers
    
    # Evaluate the answers
    evaluation_results = await evaluate(
        generated_answers=generated_answers,
        regulation_questions=regulation_questions,
        golden_answers=golden_answers,
        model=model
    )
    
    # Merge with existing evaluations if incremental
    if incremental and existing_evaluations:
        all_evaluation_results = merge_evaluations(evaluation_results, existing_evaluations, regulation_questions)
    else:
        all_evaluation_results = evaluation_results
    
    # Write evaluation results to output file
    with open(evaluation_output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_evaluation_results, f, indent=2, ensure_ascii=False)
    
    # Extract implementation name from the path
    implementation_name = os.path.basename(os.path.dirname(evaluation_output_file_path))
    
    # Calculate and update average scores
    data_path = os.environ.get("DATA_PATH", "data")
    calculate_and_update_report(implementation_name, data_path)
    
    print(f"Evaluation complete. Results written to {evaluation_output_file_path}")
    if incremental and existing_evaluations:
        print(f"Evaluated {len(evaluation_results)} new answers")
        print(f"Total answers in file: {len(all_evaluation_results)}")
    else:
        print(f"Evaluated {len(evaluation_results)} answers")
    print(f"Updated average scores in {os.path.join(data_path, 'report.csv')}")


async def report(common: bool = False, offset: int = 0, truncate: int = 0, valid: bool = False) -> None:
    """
    Generate evaluation report for all implementations.
    
    Args:
        common: Only include questions that are answered by all implementations
        offset: Start from question N (0-indexed)
        truncate: Process only this many questions from offset (0 means all)
    """
    data_path = os.environ.get("DATA_PATH", "data")
    
    # Get implementations that are already in the CSV file
    from ..calculate_report import get_implementation_names
    csv_path = os.path.join(data_path, "report.csv")
    implementation_dirs = get_implementation_names(csv_path)
    
    if not implementation_dirs:
        print("No implementations found in CSV file. Run evaluations first.")
        return
    
    print(f"Generating report for implementations: {implementation_dirs}")
    
    # Load regulation questions to get the subset based on offset/truncate
    questions_path = os.path.join(data_path, "qa_datasets", "regulation_questions.json")
    all_questions = load_questions_json(questions_path) 
    
    # Apply offset and truncate to get question subset
    start_idx = offset
    end_idx = offset + truncate if truncate > 0 else len(all_questions)
    question_subset = all_questions[start_idx:end_idx]
    subset_question_hashes = [q["question_hash"] for q in question_subset]
    
    print(f"Processing questions {start_idx} to {end_idx-1} ({len(subset_question_hashes)} questions)")
    
    # Get common questions and intersect with subset if using common mode
    if common:
        from ..calculate_report import get_common_question_hashes
        all_common_questions = get_common_question_hashes(data_path, valid)
        # Intersect common questions with our subset
        common_questions = [q for q in subset_question_hashes if q in all_common_questions]
        print(f"Found {len(common_questions)} common questions in the specified range")
        filter_questions = common_questions
    else:
        filter_questions = subset_question_hashes
    
    # Calculate and update scores for each implementation using filtered questions
    from ..calculate_report import calculate_report_scores, update_or_append_implementation
    csv_path = os.path.join(data_path, "report.csv")
    
    for implementation in implementation_dirs:
        evaluation_file = os.path.join(data_path, implementation, "answer_evaluations.json")
        if os.path.exists(evaluation_file):
            print(f"Processing {implementation}...")
            scores = calculate_report_scores(implementation, data_path, filter_questions)
            if scores:
                update_or_append_implementation(scores, csv_path)
                print(f"Updated scores for {implementation}")
        else:
            print(f"Warning: No evaluation file found for {implementation}, skipping...")
    
    print(f"Report generation complete. Updated scores in {os.path.join(data_path, 'report.csv')}")
