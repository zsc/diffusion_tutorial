#!/usr/bin/env python3
"""Add spaces around single dollar LaTeX expressions and ensure newlines before $$ blocks in markdown files."""

import re
import sys

def add_spaces_around_latex(content, escape_underscores=False):
    """Add spaces around single dollar LaTeX expressions and ensure newline before $$ blocks.
    
    Examples:
    "我$a+b$，" -> "我 $a+b$ ，"
    "其中$x$是" -> "其中 $x$ 是"
    "文字$$公式$$" -> "文字\n$$公式$$"
    
    If escape_underscores=True, also escapes underscores in LaTeX contexts:
    "$x_i$" -> "$x\\_i$"
    """
    # First, handle $$ blocks - ensure newline before starting $$
    # New approach: Find all $$ positions and process them correctly
    # We need to identify which $$ are starts and which are ends
    
    result = content
    
    # Find all $$ positions
    dollar_positions = []
    i = 0
    while i < len(result) - 1:
        if result[i:i+2] == '$$':
            dollar_positions.append(i)
            i += 2
        else:
            i += 1
    
    # Process $$ pairs from right to left (to avoid position shifts)
    # Every odd-indexed $$ is a start, even-indexed is an end
    for idx in range(len(dollar_positions) - 1, -1, -1):
        if idx % 2 == 0:  # This is a start $$
            pos = dollar_positions[idx]
            # Check if there's a newline before this $$
            if pos > 0 and result[pos-1] != '\n':
                # Insert newline before $$
                result = result[:pos] + '\n' + result[pos:]
    
    # Now handle single $ inline math
    # Pattern to match single dollar signs with content between them
    # Negative lookbehind and lookahead to avoid matching double dollars
    pattern = r'(?<!\$)\$([^\$\n]+?)\$(?!\$)'
    
    # We need to process from end to beginning to avoid position shifts
    matches = list(re.finditer(pattern, result))
    for match in reversed(matches):
        start_pos = match.start()
        end_pos = match.end()
        
        # Get surrounding characters
        before_char = result[start_pos - 1] if start_pos > 0 else ''
        after_char = result[end_pos] if end_pos < len(result) else ''
        
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
    
    # Optionally escape underscores within LaTeX contexts
    if escape_underscores:
        # First, handle display math blocks $$...$$
        display_blocks = []
        i = 0
        while i < len(result) - 1:
            if result[i:i+2] == '$$':
                start = i
                i += 2
                # Find the closing $$
                while i < len(result) - 1:
                    if result[i:i+2] == '$$':
                        end = i + 2
                        display_blocks.append((start, end))
                        i += 2
                        break
                    i += 1
            else:
                i += 1
        
        # Process display blocks from end to beginning to avoid position shifts
        for start, end in reversed(display_blocks):
            block_content = result[start+2:end-2]
            # Replace unescaped underscores
            escaped_content = re.sub(r'(?<!\\)_', r'\_', block_content)
            result = result[:start+2] + escaped_content + result[end-2:]
        
        # Now handle inline math $...$
        # Pattern to match inline math expressions
        inline_pattern = r'(?<!\$)\$([^\$\n]+?)\$(?!\$)'
        matches = list(re.finditer(inline_pattern, result))
        
        # Process from end to beginning to avoid position shifts
        for match in reversed(matches):
            start_pos = match.start() + 1  # Position after the opening $
            end_pos = match.end() - 1      # Position before the closing $
            math_content = match.group(1)
            
            # Replace unescaped underscores
            escaped_content = re.sub(r'(?<!\\)_', r'\_', math_content)
            
            # Replace the math content
            result = result[:start_pos] + escaped_content + result[end_pos:]
    
    result = result.replace('\n$$', '\n\n$$').replace('\n\n\n$$', '\n\n$$')
    return result

def process_file(filepath, escape_underscores=False):
    """Process a markdown file to add spaces around LaTeX."""
    print(f"Processing {filepath}...")
    if escape_underscores:
        print("  (with underscore escaping)")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add spaces around LaTeX
    processed_content = add_spaces_around_latex(content, escape_underscores=escape_underscores)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(processed_content)
    
    print(f"Completed processing {filepath}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add spaces around LaTeX expressions in markdown files")
    parser.add_argument('files', nargs='+', help='Markdown files to process')
    parser.add_argument('--escape-underscores', '-u', action='store_true', 
                        help='Escape underscores in LaTeX contexts (default: off)')
    
    args = parser.parse_args()
    
    for filepath in args.files:
        process_file(filepath, escape_underscores=args.escape_underscores)
