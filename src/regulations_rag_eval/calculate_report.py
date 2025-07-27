import os
import json
import csv
import argparse
from typing import Dict, List, Optional, Any, Tuple

# Avoid import dependency on dotenv which isn't installed in the test environment


def get_implementation_names(csv_path: str) -> List[str]:
    """
    Get a list of all implementation names from the CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of implementation names from the first column
    """
    implementations = []
    
    if not os.path.exists(csv_path):
        return implementations
    
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                impl_name = row.get("implementation_name", "").strip()
                if impl_name and impl_name not in implementations:
                    implementations.append(impl_name)
    except (csv.Error, IOError) as e:
        print(f"Warning: Could not read CSV file {csv_path}: {e}")
    
    return implementations


def calculate_report_scores(implementation_name: str, data_path: str, filter_question_hashes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate average scores for a specific implementation.
    
    Args:
        implementation_name: Name of the implementation
        data_path: Path to the data directory
        filter_question_hashes: List of question hashes to filter by (if None, use all questions)
        
    Returns:
        Dictionary containing average scores
    """
    evaluations_path = os.path.join(data_path, implementation_name, "answer_evaluations.json")
    
    # Check if evaluations exist
    if not os.path.exists(evaluations_path):
        print(f"No evaluations found for {implementation_name}")
        return {}
    
    # Load evaluations
    with open(evaluations_path, "r", encoding="utf-8") as f:
        all_evaluations = json.load(f)
    
    if not all_evaluations:
        print(f"Empty evaluations file for {implementation_name}")
        return {}
    
    # Filter evaluations by common questions if provided
    if filter_question_hashes is not None:
        evaluations = [eval_item for eval_item in all_evaluations 
                      if eval_item.get("question_hash") in filter_question_hashes]
        if not evaluations:
            print(f"No common question evaluations found for {implementation_name}")
            return {}
    else:
        evaluations = all_evaluations
    
    # Initialize counters and sums
    total_evaluation = 0
    total_relevance = 0
    total_clarity = 0
    total_completeness = 0
    total_correctness = 0
    total_overall = 0
    
    eval_count = len(evaluations)
    detailed_count = 0
    
    # Calculate sums
    for eval_item in evaluations:
        # Basic evaluation score (always present)
        if "evaluation" in eval_item:
            grade = eval_item["evaluation"]
            if grade == 1:
                pass
            elif grade == 2:
                total_evaluation += 15 
            elif grade == 3:
                total_evaluation += 100 
            elif grade == 4:
                total_evaluation += 120 
            # total_evaluation += eval_item["evaluation"]
        
        # Detailed metrics (may not be present in all implementations)
        if "detailed_metrics" in eval_item:
            detailed_count += 1
            metrics = eval_item["detailed_metrics"]
            
            if "relevance_score" in metrics:
                total_relevance += metrics["relevance_score"]
            
            if "clarity_score" in metrics:
                total_clarity += metrics["clarity_score"]
            
            if "completeness_score" in metrics:
                total_completeness += metrics["completeness_score"]
            
            if "correctness_score" in metrics:
                total_correctness += metrics["correctness_score"]
            
            if "overall_score" in metrics:
                total_overall += metrics["overall_score"]
    
    # Calculate averages
    avg_evaluation = total_evaluation / eval_count if eval_count > 0 else 0
    
    # Calculate detailed averages only if we have detailed data
    avg_relevance = total_relevance / detailed_count if detailed_count > 0 else None
    avg_clarity = total_clarity / detailed_count if detailed_count > 0 else None
    avg_completeness = total_completeness / detailed_count if detailed_count > 0 else None
    avg_correctness = total_correctness / detailed_count if detailed_count > 0 else None
    avg_overall = total_overall / detailed_count if detailed_count > 0 else None

    report_clarity = (avg_clarity -5) * (15.0/10.0) if avg_clarity is not None else 0 
    report_completeness = (avg_completeness -10) * (35.0/25.0) if avg_completeness is not None else 0 
    report_correctness = (avg_correctness -15) * (80.0/35.0) if avg_correctness is not None else 0 
    report_overall = report_clarity + report_completeness + report_correctness 
    
    # Return results
    return {
        "implementation_name": implementation_name,
        "evaluation": round(avg_evaluation, 2),
        "clarity_score": round(report_clarity, 2) if report_clarity is not None else None,
        "completeness_score": round(report_completeness, 2) if report_completeness is not None else None,
        "correctness_score": round(report_correctness, 2) if report_correctness is not None else None,
        "overall_score": round(report_overall, 2),
        "no_of_answers_processed": eval_count
    }


def update_csv_file(scores: Dict[str, Any], csv_path: str):
    """
    Update the CSV file with the calculated scores.
    
    Args:
        scores: Dictionary containing the calculated scores
        csv_path: Path to the CSV file
    """
    # Check if file exists and has content
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    
    # Prepare row data
    row = {
        "implementation_name": scores["implementation_name"],
        "evaluation": scores["evaluation"],
        # "relevance_score": scores["relevance_score"] if scores["relevance_score"] is not None else "N/A",
        # "clarity_score": scores["clarity_score"] if scores["clarity_score"] is not None else "N/A",
        # "completeness_score": scores["completeness_score"] if scores["completeness_score"] is not None else "N/A",
        # "correctness_score": scores["correctness_score"] if scores["correctness_score"] is not None else "N/A",
        # "overall_score": scores["overall_score"] if scores["overall_score"] is not None else "N/A",
        "no_of_answers_processed": scores["no_of_answers_processed"]
    }
    
    # Open file in append mode
    with open(csv_path, 'a', newline='') as csvfile:
        # field_names = ["implementation_name", "evaluation", "relevance_score", "clarity_score", 
        #                "completeness_score", "correctness_score", "overall_score", "no_of_answers_processed"]
        field_names = ["implementation_name", "evaluation", "no_of_answers_processed"]
        
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write row
        writer.writerow(row)


def update_or_append_implementation(scores: Dict[str, Any], csv_path: str):
    """
    Update an existing entry or append a new one for the implementation.
    
    Args:
        scores: Dictionary containing the calculated scores
        csv_path: Path to the CSV file
    """
    implementation_name = scores["implementation_name"]
    
    # Check if file exists
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        # Create new file with header and single row
        update_csv_file(scores, csv_path)
        return
    
    # File exists, load it and check for existing entry
    temp_file = csv_path + ".temp"
    implementation_found = False
    
    with open(csv_path, 'r', newline='') as csvfile, open(temp_file, 'w', newline='') as tempfile:
        reader = csv.DictReader(csvfile)
        # Always use the complete field names to ensure consistency
        # field_names = ["implementation_name", "evaluation", "relevance_score", 
        #                "clarity_score", "completeness_score", "correctness_score", 
        #                "overall_score", "no_of_answers_processed"]
        field_names = ["implementation_name", "evaluation", "no_of_answers_processed"]
 
        writer = csv.DictWriter(tempfile, fieldnames=field_names)
        writer.writeheader()
        
        # Check for existing implementation
        for row in reader:
            if row["implementation_name"] == implementation_name:
                # Replace with new scores
                implementation_found = True
                new_row = {
                    "implementation_name": scores["implementation_name"],
                    "evaluation": scores["evaluation"],
                    # "relevance_score": scores["relevance_score"] if scores["relevance_score"] is not None else "N/A",
                    # "clarity_score": scores["clarity_score"] if scores["clarity_score"] is not None else "N/A",
                    # "completeness_score": scores["c"] if scores["completeness_score"] is not None else "N/A",
                    # "correctness_score": scores["correctness_score"] if scores["correctness_score"] is not None else "N/A",
                    # "overall_score": scores["overall_score"] if scores["overall_score"] is not None else "N/A",
                    "no_of_answers_processed": scores["no_of_answers_processed"]
                }
                writer.writerow(new_row)
            else:
                # Keep existing row, fill missing fields with N/A if needed
                complete_row = {
                    "implementation_name": row.get("implementation_name", ""),
                    "evaluation": row.get("evaluation", ""),
                    # "relevance_score": row.get("relevance_score", "N/A"),
                    # "clarity_score": row.get("clarity_score", ""),
                    # "completeness_score": row.get("completeness_score", ""),
                    # "correctness_score": row.get("correctness_score", ""),
                    # "overall_score": row.get("overall_score", ""),
                    "no_of_answers_processed": row.get("no_of_answers_processed", row.get("no_of_answers_generated", ""))
                }
                writer.writerow(complete_row)
        
        # Add new entry if not found
        if not implementation_found:
            new_row = {
                "implementation_name": scores["implementation_name"],
                "evaluation": scores["evaluation"],
                # "relevance_score": scores["relevance_score"] if scores["relevance_score"] is not None else "N/A",
                # "clarity_score": scores["clarity_score"] if scores["clarity_score"] is not None else "N/A",
                # "completeness_score": scores["completeness_score"] if scores["completeness_score"] is not None else "N/A",
                # "correctness_score": scores["correctness_score"] if scores["correctness_score"] is not None else "N/A",
                # "overall_score": scores["overall_score"] if scores["overall_score"] is not None else "N/A",
                "no_of_answers_processed": scores["no_of_answers_processed"]
            }
            writer.writerow(new_row)
    
    # Replace original file with the updated one
    os.replace(temp_file, csv_path)


def get_common_question_hashes(data_path: str, valid: bool = False) -> List[str]:
    """
    Get question hashes that are answered by all implementations.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        List of question hashes that exist in all implementations
    """
    csv_path = os.path.join(data_path, "report.csv")
    implementations = get_implementation_names(csv_path)
    if not implementations:
        return []
    
    # Get question hashes for each implementation
    all_question_sets = []
    for impl in implementations:
        evaluations_path = os.path.join(data_path, impl, "answer_evaluations.json")
        if os.path.exists(evaluations_path):
            with open(evaluations_path, "r", encoding="utf-8") as f:
                evaluations = json.load(f)
            question_hashes = {eval_item.get("question_hash") for eval_item in evaluations 
                              if eval_item.get("question_hash")}
            all_question_sets.append(question_hashes)
            if valid:
                answers_path = os.path.join(data_path, impl, "generated_answers.json")
                if os.path.exists(answers_path):
                    with open(answers_path, "r", encoding="utf-8") as f:
                        answers = json.load(f)
                valid_answers = [item for item in answers if item.get("retrieved_article_paths")]
                valid_answers_len = len(valid_answers)
                print(f"Found {valid_answers_len} valid evaluations for {impl}")
                if valid_answers_len > 1 and valid_answers_len < len(answers):
                    question_hashes = {item.get("question_hash") for item in valid_answers 
                                      if item.get("question_hash")}
                    all_question_sets.append(question_hashes)

    
    # Find intersection of all question sets
    if all_question_sets:
        common_questions = set.intersection(*all_question_sets)
        return list(common_questions)
    else:
        return []


def calculate_and_update_report(implementation_name: Optional[str] = None, data_path: str = "data", common: bool = False):
    """
    Calculate scores for a specific implementation or all implementations and update the CSV file.
    
    Args:
        implementation_name: Name of the implementation (if None, calculate for all)
        data_path: Path to the data directory
        common: Only include questions that are answered by all implementations
    """
    csv_path = os.path.join(data_path, "report.csv")
    
    # Get common questions if the common flag is set
    common_question_hashes = get_common_question_hashes(data_path) if common else None
    
    if implementation_name:
        # Calculate for specific implementation
        scores = calculate_report_scores(implementation_name, data_path, common_question_hashes)
        if scores:
            update_or_append_implementation(scores, csv_path)
            print(f"Updated scores for {implementation_name}")
    else:
        # Calculate for all implementations
        implementations = get_implementation_names(csv_path)
        for impl in implementations:
            scores = calculate_report_scores(impl, data_path, common_question_hashes)
            if scores:
                update_or_append_implementation(scores, csv_path)
                print(f"Updated scores for {impl}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Calculate average scores for RAG implementations")
    parser.add_argument("--implementation", "-i", help="Specific implementation name")
    parser.add_argument("--data-path", "-d", default="data", help="Path to data directory")
    
    args = parser.parse_args()
    
    calculate_and_update_report(args.implementation, args.data_path)
    print(f"Scores updated in {os.path.join(args.data_path, 'report.csv')}")


if __name__ == "__main__":
    main()