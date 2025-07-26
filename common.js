// Common JavaScript functionality for all tutorial pages

// Initialize KaTeX for math rendering
document.addEventListener("DOMContentLoaded", function() {
    if (typeof renderMathInElement !== 'undefined') {
        renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\(', right: '\\)', display: false},
                {left: '\\[', right: '\\]', display: true}
            ],
            throwOnError: false
        });
    }
});

// Toggle answer visibility
function toggleAnswer(answerId) {
    const answer = document.getElementById(answerId);
    const button = answer.previousElementSibling;
    
    if (answer.style.display === 'none' || answer.style.display === '') {
        answer.style.display = 'block';
        button.textContent = '隐藏答案';
    } else {
        answer.style.display = 'none';
        button.textContent = '显示答案';
    }
}

// Toggle code block visibility
function toggleCode(codeId) {
    const codeContent = document.getElementById(codeId);
    const button = codeContent.previousElementSibling;
    
    if (codeContent.classList.contains('collapsed')) {
        codeContent.classList.remove('collapsed');
        button.classList.remove('collapsed');
    } else {
        codeContent.classList.add('collapsed');
        button.classList.add('collapsed');
    }
}

// Create collapsible code block
function createCollapsibleCode(title, code, language = 'python') {
    const id = 'code-' + Math.random().toString(36).substr(2, 9);
    return `
        <div class="code-collapsible">
            <button class="code-toggle collapsed" onclick="toggleCode('${id}')">
                <span>${title}</span>
            </button>
            <div id="${id}" class="code-content collapsed">
                <pre><code class="language-${language}">${escapeHtml(code)}</code></pre>
            </div>
        </div>
    `;
}

// Escape HTML for safe display
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add copy button to code blocks
document.addEventListener('DOMContentLoaded', function() {
    const codeBlocks = document.querySelectorAll('.code-block pre, .code-content pre');
    
    codeBlocks.forEach(block => {
        const wrapper = document.createElement('div');
        wrapper.style.position = 'relative';
        block.parentNode.insertBefore(wrapper, block);
        wrapper.appendChild(block);
        
        const button = document.createElement('button');
        button.textContent = '复制';
        button.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: #3498db;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        `;
        
        button.addEventListener('click', function() {
            const code = block.textContent;
            navigator.clipboard.writeText(code).then(() => {
                button.textContent = '已复制!';
                setTimeout(() => {
                    button.textContent = '复制';
                }, 2000);
            });
        });
        
        wrapper.appendChild(button);
    });
});

// Progress tracking
function saveProgress(chapter, section) {
    const progress = JSON.parse(localStorage.getItem('diffusion-tutorial-progress') || '{}');
    progress[chapter] = section;
    localStorage.setItem('diffusion-tutorial-progress', JSON.stringify(progress));
}

function loadProgress(chapter) {
    const progress = JSON.parse(localStorage.getItem('diffusion-tutorial-progress') || '{}');
    return progress[chapter] || null;
}

// Smooth scroll to sections
document.addEventListener('DOMContentLoaded', function() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Add navigation keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Alt + Left: Previous chapter
    if (e.altKey && e.key === 'ArrowLeft') {
        const prevLink = document.querySelector('.nav-bar a[href*="chapter"]:first-child');
        if (prevLink && !prevLink.textContent.includes('返回目录')) {
            window.location.href = prevLink.href;
        }
    }
    
    // Alt + Right: Next chapter
    if (e.altKey && e.key === 'ArrowRight') {
        const nextLink = document.querySelector('.nav-bar a[href*="chapter"]:last-child');
        if (nextLink) {
            window.location.href = nextLink.href;
        }
    }
    
    // Alt + Home: Back to index
    if (e.altKey && e.key === 'Home') {
        window.location.href = 'index.html';
    }
});

// Syntax highlighting with Prism.js (if available)
document.addEventListener('DOMContentLoaded', function() {
    if (typeof Prism !== 'undefined') {
        Prism.highlightAll();
    }
});

// Create interactive plot placeholder
function createPlotPlaceholder(id, description) {
    return `
        <div class="visualization" id="${id}">
            <div style="background: #f0f0f0; padding: 60px 20px; border-radius: 8px; border: 2px dashed #ccc;">
                <p style="color: #666; margin: 0;">📊 交互式图表占位符</p>
                <p style="color: #888; font-size: 0.9em; margin-top: 10px;">${description}</p>
            </div>
        </div>
    `;
}

// Export functions for use in chapter files
window.tutorialUtils = {
    toggleAnswer,
    toggleCode,
    createCollapsibleCode,
    saveProgress,
    loadProgress,
    createPlotPlaceholder
};