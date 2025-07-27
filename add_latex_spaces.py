#!/usr/bin/env python3
"""Add spaces around single dollar LaTeX expressions and ensure newlines before $$ blocks in markdown files."""

import re
import sys

def add_spaces_around_latex(content):
    """Add spaces around single dollar LaTeX expressions and ensure newline before $$ blocks.
    
    Examples:
    "我$a+b$，" -> "我 $a+b$ ，"
    "其中$x$是" -> "其中 $x$ 是"
    "文字$$公式$$" -> "文字\n$$公式$$"
    """
    # First, handle $$ blocks - ensure newline before starting $$
    # Pattern to match $$ at the start of a display equation (not at the end)
    # This looks for $$ that is NOT preceded by another $ and is followed by non-$ content
    display_pattern = r'([^\n])(\$\$)(?!\$)([^\$]+?\$\$)'
    
    def ensure_newline_before_display(match):
        before_char = match.group(1)
        display_start = match.group(2)
        rest = match.group(3)
        
        # Add newline before $$ if not already there
        return f"{before_char}\n{display_start}{rest}"
    
    # Apply display math fix
    result = re.sub(display_pattern, ensure_newline_before_display, content)
    
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