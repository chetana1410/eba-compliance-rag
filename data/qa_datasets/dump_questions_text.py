#!/usr/bin/env python3

import json
import argparse
import sys

def print_question_texts(filepath):
    """
    Reads a JSON file containing a list of question objects,
    and prints the 'question_text' for each.

    Args:
        filepath (str): The path to the JSON file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error: Could not decode JSON from '{filepath}'. Invalid JSON format.", file=sys.stderr)
                print(f"Details: {e}", file=sys.stderr)
                sys.exit(1)

        if not isinstance(data, list):
            print(f"Error: Expected a JSON list (array) in '{filepath}', but got {type(data)}.", file=sys.stderr)
            sys.exit(1)

        for item in data:
            if isinstance(item, dict):
                question_text = item.get("question_text")
                if question_text is not None and len(question_text) < 350:
                    print(question_text + "\n\n")  # print() automatically adds a newline
            else:
                print(f"Warning: Expected a dictionary (object) within the list, but found: {type(item)}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prints the 'question_text' from each object in a JSON file."
    )
    parser.add_argument(
        "filepath",
        help="Path to the JSON file containing question data."
    )

    args = parser.parse_args()
    print_question_texts(args.filepath)

