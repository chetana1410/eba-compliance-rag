#!/usr/bin/env python3
import sys
import json
from lxml import etree
from typing import Any, List, Dict


def convert(xml_input: str, xslt_path: str) -> str:
    """Apply the XSLT transform and return transformed output."""
    dom = etree.parse(xml_input)
    xslt = etree.parse(xslt_path)
    transformer = etree.XSLT(xslt)
    result = transformer(dom)
    return str(result)


def parse_articles(content: str) -> List[Dict]:
    """Parse article sections from XSL output into hierarchical structure."""
    articles = []
    article_sections = content.split('---ARTICLE-SPLIT---')
    
    for article_section in article_sections[1:]:  # Skip first empty section
        if not article_section.strip():
            continue
            
        lines = article_section.strip().split('\n')
        
        # Find article metadata
        article_metadata: Dict[str, Any] = { 'type': 'article' }
        content_start_idx = 0
        
        for i, line in enumerate(lines):
            if line.startswith('IDENTIFIER:'):
                article_metadata['identifier'] = line.split(':', 1)[1].strip()
            elif line.startswith('TITLE:'):
                article_metadata['title'] = line.split(':', 1)[1].strip()
            elif line.startswith('SUBTITLE:'):
                article_metadata['subtitle'] = line.split(':', 1)[1].strip()
            elif line == '---CONTENT---':
                content_start_idx = i + 1
                break
         
        # Extract article content
        if content_start_idx < len(lines):
            article_content = '\n'.join(lines[content_start_idx:]).strip()
        else:
            article_content = ""
        
        # Parse splits within this article
        splits = parse_splits_in_content(
            article_content, 
            article_metadata['identifier'],
            article_metadata
        )
        
        article_metadata['splits'] = splits
        articles.append(article_metadata)
    
    return articles


def parse_splits_in_content(content: str, article_id: str, article_metadata: Dict = None) -> List[Dict[str, str]]:
    """Parse split sections within article content."""
    splits = []
    split_sections = content.split('---PARAG-SPLIT---')
    
    # Handle initial content before first split split (if any)
    original_content = split_sections[0].strip()
    
    # Create intro content with article metadata
    intro_content = ""
    if article_metadata:
        intro_content += f"# {article_metadata['title']}\n\n"
        intro_content += f"**{article_metadata['subtitle']}**\n\n"
        intro_content += f"*Article {article_metadata['identifier']}*\n\n"
    intro_content += original_content
    
    initial_split = {
        'type': 'article_split',
        'paragraph_number': 'intro',
        'content': intro_content
    }
    splits.append(initial_split)
    
    # Process numbered splits
    for split_section in split_sections[1:]:
        if not split_section.strip():
            continue
            
        lines = split_section.strip().split('\n')
        
        # Find split metadata
        split_metadata = {}
        content_start_idx = 0
        
        for i, line in enumerate(lines):
            if line.startswith('IDENTIFIER:'):
                split_metadata['identifier'] = line.split(':', 1)[1].strip()
            elif line.startswith('NUMBER:'):
                split_metadata['paragraph_number'] = line.split(':', 1)[1].strip()
            elif line == '---CONTENT---':
                content_start_idx = i + 1
                break
        
        # Extract split content
        if content_start_idx < len(lines):
            split_content = '\n'.join(lines[content_start_idx:]).strip()
        else:
            split_content = ""
        
        # Store original content and create formatted version
        split_metadata['type'] = 'article_split'
        split_metadata['content'] = format_split_with_metadata({
            'paragraph_number': split_metadata.get('paragraph_number', 'intro'),
            'content': split_content
        })
        splits.append(split_metadata)
    
    return splits


def format_split_with_metadata(split: Dict, article_title: str = None) -> str:
    """Format individual split with metadata as markdown."""
    markdown = ""
    
    # Add split number if not intro
    if split['paragraph_number'] != 'intro':
        markdown += f"**{split['paragraph_number']}** "
    
    markdown += f"{split['content']}\n\n"
    
    return markdown


def extract_articles_to_json(xml_input: str, xslt_path: str, output_file: str, splits_file: str = None) -> None:
    """Parse articles and save to JSON file with proper escaping."""
    # Convert XML using XSL
    transformed_content = convert(xml_input, xslt_path)
    
    # Parse articles hierarchically
    articles = parse_articles(transformed_content)
    
    # Save to JSON with proper escaping
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(articles)} articles to {output_file}")
    
    # Count total splits
    total_splits = sum(len(article['splits']) for article in articles)
    print(f"Total splits: {total_splits}")
    
    # If second file specified, save all split contents joined
    if splits_file:
        all_splits_content = []
        for article in articles:
            for split in article['splits']:
                all_splits_content.append(split['content'])
        
        # Join all split contents
        joined_content = '\n'.join(all_splits_content)
        
        with open(splits_file, 'w', encoding='utf-8') as f:
            f.write(joined_content)
        
        print(f"Saved joined split contents to {splits_file}")
        print(f"Total content length: {len(joined_content)} characters")


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print(f"Usage: {sys.argv[0]} <input.xml> <transform.xsl> [output.json] [splits.txt]")
        print("If output.json is provided, saves to JSON file")
        print("If splits.txt is also provided, saves joined split contents")
        print("Otherwise, shows test output")
        sys.exit(1)

    input_xml = sys.argv[1]
    xslt_file = sys.argv[2]
    
    if len(sys.argv) >= 4:
        # Save to JSON mode
        output_file = sys.argv[3]
        splits_file = sys.argv[4] if len(sys.argv) == 5 else None
        extract_articles_to_json(input_xml, xslt_file, output_file, splits_file)
        
        