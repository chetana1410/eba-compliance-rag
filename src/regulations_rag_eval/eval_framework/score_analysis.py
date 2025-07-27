"""
Score analysis module for the regulation RAG evaluation framework.
Provides functions for analyzing and comparing evaluation scores.
"""

import os
import json
import glob
from statistics import mean, median, stdev
from pathlib import Path
from typing import Dict, List, Optional, Any


def load_evaluation_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluations from a JSON file.
    
    Args:
        file_path: Path to the evaluation JSON file
        
    Returns:
        List of evaluation entries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading {file_path}: {str(e)}")
        return []


def calculate_average_score(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate average score and other statistics from a list of evaluations.
    
    Args:
        evaluations: List of evaluation entries
        
    Returns:
        Dictionary with statistics (average, median, std_dev, count, min, max)
    """
    if not evaluations:
        return {
            "average": 0.0,
            "median": 0.0,
            "std_dev": 0.0,
            "count": 0,
            "min": 0,
            "max": 0,
            "scores_distribution": {}
        }
    
    scores = [e["evaluation"] for e in evaluations]
    
    # Calculate score distribution
    score_distribution = {}
    for score in scores:
        score_distribution[score] = score_distribution.get(score, 0) + 1
    
    # Convert to percentages
    total = len(scores)
    score_distribution = {
        score: {"count": count, "percentage": (count / total) * 100} 
        for score, count in sorted(score_distribution.items())
    }
    
    return {
        "average": mean(scores),
        "median": median(scores),
        "std_dev": stdev(scores) if len(scores) > 1 else 0.0,
        "count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "scores_distribution": score_distribution
    }


def get_all_implementation_dirs(data_path: str) -> List[str]:
    """
    Get all implementation directories in the data path.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        List of implementation directory paths
    """
    impl_dirs = glob.glob(os.path.join(data_path, "*"))
    return [d for d in impl_dirs if os.path.isdir(d)]


def get_implementation_scores(
    data_path: str, 
    implementation: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate scores for one or all implementations.
    
    Args:
        data_path: Path to the data directory
        implementation: Specific implementation to analyze, or None for all
        
    Returns:
        Dictionary mapping implementation names to score statistics
    """
    results = {}
    
    # Find all implementation directories or just the specified one
    if implementation:
        impl_dirs = [os.path.join(data_path, implementation)]
    else:
        impl_dirs = get_all_implementation_dirs(data_path)
    
    # Process each implementation directory
    for impl_dir in impl_dirs:
        impl_name = os.path.basename(impl_dir)
        eval_file = os.path.join(impl_dir, "answer_evaluations.json")
        
        if os.path.exists(eval_file):
            evaluations = load_evaluation_file(eval_file)
            if evaluations:
                results[impl_name] = calculate_average_score(evaluations)
                
    return results


def format_score_report(results: Dict[str, Dict[str, Any]], detailed: bool = False) -> str:
    """
    Format the score results as a human-readable report.
    
    Args:
        results: Dictionary mapping implementation names to score statistics
        detailed: Whether to include detailed statistics
        
    Returns:
        Formatted report string
    """
    if not results:
        return "No evaluation data found"
    
    # Sort implementations by average score (descending)
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1]["average"], 
        reverse=True
    )
    
    lines = [
        "\nEvaluation Score Summary:",
        "--------------------------"
    ]
    
    for impl_name, stats in sorted_results:
        lines.append(
            f"{impl_name}: {stats['average']:.2f} (from {stats['count']} evaluations)"
        )
        
        if detailed:
            lines.append(f"  Median: {stats['median']:.2f}")
            lines.append(f"  Std Dev: {stats['std_dev']:.2f}")
            lines.append(f"  Range: {stats['min']} - {stats['max']}")
            lines.append("  Score Distribution:")
            
            for score, info in stats["scores_distribution"].items():
                lines.append(
                    f"    Score {score}: {info['count']} responses ({info['percentage']:.1f}%)"
                )
            
            lines.append("")  # Add blank line between implementations
    
    return "\n".join(lines)


def save_score_report(results: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """
    Save score analysis results to a JSON file.
    
    Args:
        results: Dictionary mapping implementation names to score statistics
        output_path: Path to save the results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)