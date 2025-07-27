#!/usr/bin/env python3
"""
Script to filter generated answers and answer evaluations files by removing entries
with question hashes specified in an input file.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_question_hashes(input_file: str) -> List[str]:
    """Load question hashes from input file (one per line)."""
    with open(input_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def find_json_files(directory: str, filename_pattern: str) -> List[str]:
    """Find all JSON files matching the filename pattern in subdirectories."""
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if filename_pattern in file and file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def filter_json_entries(data: List[Dict[Any, Any]], question_hashes_to_remove: List[str]) -> List[Dict[Any, Any]]:
    """Filter out entries with matching question hashes, asking user confirmation for each removal."""
    filtered_data = []
    
    for entry in data:
        question_hash = entry.get('question_hash')
        if question_hash in question_hashes_to_remove:
            response = input(f"Remove entry with question_hash '{question_hash}'? (y/n): ").lower().strip()
            if response == 'y':
                print(f"Removing entry with question_hash: {question_hash}")
                continue
            else:
                print(f"Keeping entry with question_hash: {question_hash}")
        
        filtered_data.append(entry)
    
    return filtered_data


def process_file(file_path: str, question_hashes_to_remove: List[str]) -> None:
    """Process a single JSON file to remove entries with specified question hashes."""
    print(f"\nProcessing file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Warning: {file_path} does not contain a JSON array. Skipping.")
            return
        
        original_count = len(data)
        filtered_data = filter_json_entries(data, question_hashes_to_remove)
        removed_count = original_count - len(filtered_data)
        
        if removed_count > 0:
            print(f"Removed {removed_count} entries from {file_path}")
            with open(file_path, 'w') as f:
                json.dump(filtered_data, f, indent=2)
            print(f"File updated: {file_path}")
        else:
            print(f"No entries removed from {file_path}")
            
    except json.JSONDecodeError as e:
        print(f"Error reading JSON from {file_path}: {e}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python filter_by_question_hash.py <input_file_with_hashes>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    # Load question hashes to remove
    question_hashes_to_remove = load_question_hashes(input_file)
    print(f"Loaded {len(question_hashes_to_remove)} question hashes to remove")
    
    if not question_hashes_to_remove:
        print("No question hashes found in input file.")
        sys.exit(0)
    
    # Find all generated answers files in data/ subdirectories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_dir  # Since script is in data/ directory
    generated_answers_files = find_json_files(data_dir, "generated_answers")
    answer_evaluations_files = find_json_files(data_dir, "answer_evaluations")
    
    all_files = generated_answers_files + answer_evaluations_files
    
    if not all_files:
        print("No matching JSON files found.")
        sys.exit(0)
    
    print(f"Found {len(generated_answers_files)} generated answers files")
    print(f"Found {len(answer_evaluations_files)} answer evaluations files")
    
    # Process each file with user confirmation
    for file_path in all_files:
        response = input(f"\nProceed to process file '{file_path}'? (y/n): ").lower().strip()
        if response == 'y':
            process_file(file_path, question_hashes_to_remove)
        else:
            print(f"Skipping file: {file_path}")
    
    print("\nProcessing complete.")


if __name__ == "__main__":
    main()