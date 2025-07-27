#!/usr/bin/env python3
import re
import sys

def remove_code_blocks(content):
    """Remove all ```python code blocks and replace with descriptions."""
    
    # Pattern to match code blocks with context
    # This captures text before the code block to understand context
    pattern = r'(\*\*[^*]+\*\*：\s*\n)(```python\n[\s\S]*?```)'
    
    def replace_code_block(match):
        context = match.group(1)
        # Keep the context header but remove the code
        return context + "\n[代码实现已转换为数学公式和文字描述]"
    
    # First pass: replace code blocks that have clear context headers
    content = re.sub(pattern, replace_code_block, content)
    
    # Second pass: remove any remaining standalone code blocks
    content = re.sub(r'```python\n[\s\S]*?```', '[代码块已移除]', content)
    
    return content

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove code blocks
    new_content = remove_code_blocks(content)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    # Count how many blocks were removed
    original_count = content.count('```python')
    new_count = new_content.count('```python')
    removed = original_count - new_count
    
    print(f"Removed {removed} Python code blocks from {filepath}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "chapter10.md"
    
    process_file(filepath)