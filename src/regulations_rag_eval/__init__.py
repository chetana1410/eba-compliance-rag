import asyncio
import os
import click
from . import config
from .eval_framework import rag_eval
from .eval_framework.evaluate_answers import generate_evaluate_with_files
from .eval_framework.score_analysis import get_implementation_scores, format_score_report, save_score_report

@click.group()
def cli():
    """Command line interface for the regulations RAG evaluation framework."""
    pass

@cli.command()
@click.option("--rag-implementation", "-r", required=True, help="Name of the RAG implementation to evaluate.")
@click.option("--truncate", "-t", type=int, default=0, help="Truncate the list of questions to this length if not zero.")
@click.option("--offset", type=int, default=0, help="Start from question N (0-indexed).")
@click.option("--incremental", "-i", is_flag=True, default=False, help="Skip questions that already have answers.")
@click.option("--questions-filter-file", help="Path to file containing question hashes to filter (one per line).")
@click.option("--force-update", is_flag=True, default=False, help="Force update existing data.")
def run(rag_implementation: str, truncate: int, offset: int, incremental: bool, questions_filter_file: str, force_update: bool):
    """Evaluate a RAG implementation.
    
    Examples:
        # Run full evaluation with naive_rag implementation
        $ regulations-rag-eval run -r naive_rag
        
        # Run with RAPTOR, limiting to first 5 questions
        $ regulations-rag-eval run -r RAPTOR -t 5
        
        # Run with offset and truncate (questions 10-14)
        $ regulations-rag-eval run -r naive_rag --offset 10 -t 5
        
        # Run incrementally, skipping already answered questions
        $ regulations-rag-eval run -r naive_rag --incremental
    """
    asyncio.run(rag_eval(rag_implementation, truncate_n=truncate, offset=offset, incremental=incremental, questions_filter_file=questions_filter_file, force_update=force_update))

@cli.command()
@click.option("--answers-file", "-a", required=True, help="Path to the JSON file containing generated answers.")
@click.option("--questions-file", "-q", help="Path to the JSON file containing regulation questions.")
@click.option("--golden-answers-file", "-g", help="Path to the JSON file containing golden answers.")
@click.option("--output-file", "-o", help="Path to write the evaluation results.")
@click.option("--model", "-m", default="openai/gpt-4o", help="The model identifier to use for evaluation.")
@click.option("--incremental", "-i", is_flag=True, default=False, help="Skip answers that already have evaluations.")
@click.option("--valid", "-v", is_flag=True, default=False, help="Only evaluate answers with non-empty retrieved_article_paths.")
@click.option("--questions-filter-file", help="Path to file containing question hashes to filter (one per line).")
def evaluate(answers_file: str, questions_file: str, golden_answers_file: str, output_file: str, model: str, incremental: bool, valid: bool, questions_filter_file: str):
    """Evaluate generated answers against golden answers.
    
    Examples:
        # Basic evaluation with default settings (except answers file)
        $ regulations-rag-eval evaluate -a data/naive_rag/generated_answers.json
        
        # Full custom paths
        $ regulations-rag-eval evaluate -a data/RAPTOR/generated_answers.json -q data/qa_datasets/regulation_questions.json -g data/qa_datasets/golden_answers.json -o data/RAPTOR/evaluations.json
        
        # Using a different evaluation model
        $ regulations-rag-eval evaluate -a data/naive_rag/generated_answers.json -m anthropic/claude-3-opus-20240229
        
        # Run incrementally, skipping already evaluated answers
        $ regulations-rag-eval evaluate -a data/naive_rag/generated_answers.json --incremental
        
        # Only evaluate answers with valid retrieved article paths
        $ regulations-rag-eval evaluate -a data/naive_rag/generated_answers.json --valid
    """
    asyncio.run(generate_evaluate_with_files(
        answers_file_path=answers_file,
        questions_file_path=questions_file,
        golden_answer_file_path=golden_answers_file,
        evaluation_output_file_path=output_file,
        model=model,
        incremental=incremental,
        valid=valid,
        questions_filter_file=questions_filter_file
    ))

@cli.command()
@click.option("--implementation", "-i", help="Specific RAG implementation to analyze, or 'all' for all implementations")
@click.option("--detailed", "-d", is_flag=True, default=False, help="Show detailed statistics including score distribution")
@click.option("--output", "-o", help="Path to save the analysis results as JSON")
@click.option("--compare", "-c", is_flag=True, default=False, help="Compare all implementations and show ranking")
def analyze_scores(implementation: str | None, detailed: bool, output: str, compare: bool):
    """Calculate and display average evaluation scores for RAG implementations.
    
    Examples:
        # Calculate average score for naive_rag implementation
        $ regulations-rag-eval analyze-scores -i naive_rag
        
        # Show detailed statistics for naive_rag implementation
        $ regulations-rag-eval analyze-scores -i naive_rag -d
        
        # Calculate and compare scores for all implementations
        $ regulations-rag-eval analyze-scores --compare
        
        # Save analysis results to a JSON file
        $ regulations-rag-eval analyze-scores -i naive_rag -o analysis/naive_vector_scores.json
    """
    data_path = os.getenv("DATA_PATH", "data")
    
    # If compare is True, ignore the implementation parameter
    if compare:
        implementation = None
    
    # Calculate scores
    results = get_implementation_scores(data_path, implementation)
    
    # Display the results
    print(format_score_report(results, detailed))
    
    # Save to file if requested
    if output:
        save_score_report(results, output)
        print(f"\nAnalysis saved to: {output}")

@cli.command()
@click.option("--common", "-c", is_flag=True, default=False, help="Only include questions that are answered by all implementations.")
@click.option("--offset", type=int, default=0, help="Start from question N (0-indexed).")
@click.option("--truncate", "-t", type=int, default=0, help="Process only this many questions from offset (0 means all).")
@click.option("--valid", "-v", is_flag=True, default=False, help="Only evaluate answers with non-empty retrieved_article_paths.")
def report(common: bool, offset: int, truncate: int, valid: bool):
    """Generate evaluation report for all implementations.
    
    Examples:
        # Generate report for all implementations
        $ regulations-rag-eval report
        
        # Generate report only for questions answered by all implementations
        $ regulations-rag-eval report --common
        
        # Generate report for questions 10-14 (5 questions starting from index 10)
        $ regulations-rag-eval report --offset 10 --truncate 5
        
        # Generate report for common questions starting from question 5
        $ regulations-rag-eval report --common --offset 5
    """
    from .eval_framework.evaluate_answers import report as report_func
    asyncio.run(report_func(common=common, offset=offset, truncate=truncate, valid=valid))

def main():
    """Main entry point for the application."""
    cli()

if __name__ == "__main__":
    main()