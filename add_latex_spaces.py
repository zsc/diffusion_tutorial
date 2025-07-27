#!/usr/bin/env python3
"""Add spaces around single dollar LaTeX expressions in markdown files."""

import re
import sys

def add_spaces_around_latex(content):
    """Add spaces around single dollar LaTeX expressions.
    
    Examples:
    "我$a+b$，" -> "我 $a+b$ ，"
    "其中$x$是" -> "其中 $x$ 是"
    """
    # Pattern to match single dollar signs with content between them
    # Negative lookbehind and lookahead to avoid matching double dollars
    pattern = r'(?<!\$)\$([^\$\n]+?)\$(?!\$)'
    
    def replace_func(match):
        latex_content = match.group(1)
        full_match = match.group(0)
        start_pos = match.start()
        end_pos = match.end()
        
        # Get the character before and after the match
        before_char = content[start_pos - 1] if start_pos > 0 else ''
        after_char = content[end_pos] if end_pos < len(content) else ''
        
        # Add space before if needed (not already space or newline or start of string)
        prefix = ' ' if before_char and before_char not in ' \n\t' else ''
        
        # Add space after if needed (not already space or newline or end of string)
        suffix = ' ' if after_char and after_char not in ' \n\t' else ''
        
        return f"{prefix}${latex_content}${suffix}"
    
    # First pass: add spaces around LaTeX
    result = content
    
    # We need to process from end to beginning to avoid position shifts
    matches = list(re.finditer(pattern, content))
    for match in reversed(matches):
        start_pos = match.start()
        end_pos = match.end()
        
        # Get surrounding characters
        before_char = content[start_pos - 1] if start_pos > 0 else ''
        after_char = content[end_pos] if end_pos < len(content) else ''
        
        # Build replacement with appropriate spaces
        latex_expr = match.group(0)
        replacement = latex_expr
        
        # Add space before if needed
        if before_char and before_char not in ' \n\t':
            replacement = ' ' + replacement
        
        # Add space after if needed  
        if after_char and after_char not in ' \n\t':
            replacement = replacement + ' '
            
        # Replace in result
        result = result[:start_pos] + replacement + result[end_pos:]
    
    return result

def process_file(filepath):
    """Process a markdown file to add spaces around LaTeX."""
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add spaces around LaTeX
    processed_content = add_spaces_around_latex(content)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(processed_content)
    
    print(f"Completed processing {filepath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_latex_spaces.py <file1.md> [file2.md ...]")
        sys.exit(1)
    
    for filepath in sys.argv[1:]:
        process_file(filepath)