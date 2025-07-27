import asyncio
import json
import os
import time
import csv
from datetime import datetime
from typing import List, Dict
from ..short_hash import generate_short_hash

from ..eval_framework_data_types import RegulationQuestion, GeneratedAnswer
from ..rag_implementation_data_types import RAGResult
from .get_rag_impl_entry_point import get_rag_impl_entry_point
from ..utils import load_questions_filter, load_questions_json

def load_existing_answers(implementation_name: str) -> Dict[str, GeneratedAnswer]:
    """Load existing answers and return a mapping of question_hash -> GeneratedAnswer"""
    data_dir = os.getenv("DATA_PATH", "data")
    answers_path = os.path.join(data_dir, implementation_name, "generated_answers.json")
    
    if not os.path.exists(answers_path):
        return {}
    
    try:
        with open(answers_path, "r", encoding="utf-8") as f:
            existing_answers = json.load(f)
        return {answer["question_hash"]: answer for answer in existing_answers}
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load existing answers: {e}")
        return {}

def filter_unanswered_questions(
    questions: List[RegulationQuestion], 
    existing_answers: Dict[str, GeneratedAnswer]
) -> tuple[List[RegulationQuestion], List[RegulationQuestion]]:
    """
    Separate questions into answered and unanswered.
    Returns: (unanswered_questions, answered_questions)
    """
    unanswered = []
    answered = []
    
    for question in questions:
        if question["question_hash"] in existing_answers:
            answered.append(question)
        else:
            unanswered.append(question)
    
    return unanswered, answered

def merge_answers(
    new_answers: List[GeneratedAnswer], 
    existing_answers: Dict[str, GeneratedAnswer],
    all_questions: List[RegulationQuestion]
) -> List[GeneratedAnswer]:
    """Merge new answers with existing ones, preserving existing answers"""
    merged = existing_answers.copy()  # Start with existing answers
    update = {answer["question_hash"]: answer for answer in new_answers}
    merged.update(update)  # Update with new answers
    merged_list = list(merged.values()) 
    
    # Create mapping of question_hash to original question order
    question_order = {q["question_hash"]: i for i, q in enumerate(all_questions)}
    
    # Sort by original question order
    merged_list.sort(key=lambda x: question_order.get(x["question_hash"], float('inf')))
    return merged_list

async def rag_eval(rag_implementation: str, truncate_n: int = 0, offset: int = 0, incremental: bool = False, questions_filter_file: str = None, force_update: bool = False) -> None:
    """
    Main evaluation function for the RAG implementation.
    
    Args:
        rag_implementation: Name of the RAG implementation to evaluate
        truncate_n: Truncate the list of questions to this length if not zero
        offset: Start from question N (0-indexed), defaults to 0
        incremental: Skip questions that already have answers, defaults to False
        questions_filter_file: Path to file containing question hashes to filter (one per line)
    """
    # Start timer for tracking execution time
    start_time = time.time()
    
    if not rag_implementation:
        raise ValueError("RAG implementation name must be provided.")
    # Load questions
    questions_path = os.path.join("data", "qa_datasets", "regulation_questions.json")
    questions_data = load_questions_json(questions_path)
    
    # Parse to RegulationQuestion objects
    questions: List[RegulationQuestion] = questions_data
 
     # Apply offset and truncate first
    if offset > 0 or truncate_n > 0:
        start_idx = offset
        end_idx = offset + truncate_n if truncate_n > 0 else len(questions)
        questions = questions[start_idx:end_idx]
    
    # Apply question filter if provided
    filter_hashes = load_questions_filter(questions_filter_file)
    if filter_hashes:
        questions = [q for q in questions if q["question_hash"] in filter_hashes]
        print(f"Filtered to {len(questions)} questions based on filter file")
    
   # Handle incremental processing after offset/truncate
    if incremental:
        existing_answers = load_existing_answers(rag_implementation)
        if force_update:
            questions_to_process = questions
            already_answered = []
        else:
            questions_to_process, already_answered = filter_unanswered_questions(questions, existing_answers)
        
        print(f"Questions in range: {len(questions)}")
        print(f"Already answered: {len(already_answered)}")
        print(f"Questions to process: {len(questions_to_process)}")
        
        if not questions_to_process:
            print("All questions in range already have answers. Use without --incremental to regenerate.")
            return
        
        questions = questions_to_process
    else:
        existing_answers = {}
    
    # Extract just the question text
    question_texts = [q["question_text"] for q in questions]
    
    # Get the entry point for the RAG implementation and options
    rag_impl_func, implementation_params = get_rag_impl_entry_point(rag_implementation)
    
    # Call the RAG implementation
    results: List[RAGResult] = await rag_impl_func(question_texts, rag_implementation, implementation_params)
    
    # Create output directory if it doesn't exist
    data_dir = os.getenv("DATA_PATH", "data")
    output_dir = os.path.join(data_dir, rag_implementation)
    os.makedirs(output_dir, exist_ok=True)
    
    # Current timestamp for run ID
    timestamp = datetime.now().isoformat()
    run_id = f"{rag_implementation}_{timestamp}"
    
    # Convert RAGResults to GeneratedAnswers
    generated_answers: List[GeneratedAnswer] = []
    
    for i, result in enumerate(results):
        # Generate a hash for the answer
        answer_hash = generate_short_hash(result.answer)
        # Create a unique identifier for the answer 
        
        generated_answer: GeneratedAnswer = {
            "question_hash": questions[i]["question_hash"],
            "answer_hash": answer_hash,
            "answer_text": result.answer,
            "retrieved_article_paths": result.retrieved_articles,
            "rag_implementation_code": rag_implementation,
            "run_id": run_id,
            "run_timestamp": timestamp
        }
        
        generated_answers.append(generated_answer)
    
    # Merge with existing answers if incremental
    if incremental and existing_answers:
        all_generated_answers = merge_answers(generated_answers, existing_answers, questions_data)
    else:
        all_generated_answers = generated_answers
    
    # Save generated answers to JSON
    output_path = os.path.join(output_dir, "generated_answers.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_generated_answers, f, indent=2, ensure_ascii=False)
    
    # End timer and calculate total execution time
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    
    # Number of answers generated
    num_answers = len(generated_answers)
    
    # Update the CSV file with execution metrics
    csv_path = os.path.join(data_dir, "time_scores.csv")
    
    try:
        # Read existing data to ensure proper formatting
        existing_data = []
        header = ["implementation_name", "time_taken_in_seconds", "no_of_answers_generated"]
        
        if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
            with open(csv_path, 'r', newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                for i, row in enumerate(csv_reader):
                    if i == 0 and row[0] == "implementation_name":
                        # This is the header row
                        header = row
                    else:
                        # Ensure the row has 3 elements to prevent malformed data
                        if len(row) >= 3:
                            existing_data.append(row[:3])
                        elif len(row) == 1 and "," in row[0]:
                            # Handle malformed rows with comma in the data
                            parts = row[0].split(',')
                            if len(parts) >= 3:
                                existing_data.append(parts[:3])
        
        # Write all data to file with proper formatting
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write the header
            csv_writer.writerow(header)
            # Write existing data
            for row in existing_data:
                csv_writer.writerow(row)
            # Write new data
            csv_writer.writerow([rag_implementation, execution_time, num_answers])
    
    except Exception as e:
        print(f"Error writing to time_scores.csv: {e}")
        # Fallback: Create a new file if there's an error
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["implementation_name", "time_taken_in_seconds", "no_of_answers_generated"])
            csv_writer.writerow([rag_implementation, execution_time, num_answers])
    
    print(f"Evaluation complete. Generated answers saved to {output_path}")
    if incremental and existing_answers:
        print(f"Time taken: {execution_time} seconds for {num_answers} new answers")
        print(f"Total answers in file: {len(all_generated_answers)}")
    else:
        print(f"Time taken: {execution_time} seconds for {num_answers} answers")
    print(f"Execution metrics saved to {csv_path}")
