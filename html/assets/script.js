// Sidebar toggle for mobile
document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('active');
        });
    }
    
    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function(event) {
        const isClickInside = sidebar.contains(event.target);
        const isToggleClick = sidebarToggle.contains(event.target);
        
        if (!isClickInside && !isToggleClick && sidebar.classList.contains('active')) {
            sidebar.classList.remove('active');
        }
    });
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Pure CSS Tree Navigation
    const treeNav = document.querySelector('.tree-nav');
    if (treeNav) {
        const TREE_STATE_KEY = 'tree-nav-state';
        
        // Load saved state
        function loadTreeState() {
            try {
                const saved = localStorage.getItem(TREE_STATE_KEY);
                return saved ? JSON.parse(saved) : {};
            } catch (e) {
                return {};
            }
        }
        
        // Save tree state
        function saveTreeState() {
            const state = {};
            document.querySelectorAll('.tree-folder').forEach(folder => {
                const header = folder.querySelector('.tree-folder-header');
                if (header) {
                    const title = header.querySelector('.tree-title');
                    if (title) {
                        const folderPath = title.textContent.trim();
                        state[folderPath] = folder.classList.contains('expanded');
                    }
                }
            });
            try {
                localStorage.setItem(TREE_STATE_KEY, JSON.stringify(state));
            } catch (e) {
                // Ignore localStorage errors
            }
        }
        
        // Apply saved state
        function applyTreeState() {
            const state = loadTreeState();
            document.querySelectorAll('.tree-folder').forEach(folder => {
                const header = folder.querySelector('.tree-folder-header');
                if (header) {
                    const title = header.querySelector('.tree-title');
                    if (title) {
                        const folderPath = title.textContent.trim();
                        // Check if folder contains active item
                        const hasActive = folder.querySelector('.tree-item.active');
                        
                        if (hasActive) {
                            // Always expand folders containing active items
                            folder.classList.add('expanded');
                        } else if (state[folderPath] !== undefined) {
                            // Apply saved state for other folders
                            if (state[folderPath]) {
                                folder.classList.add('expanded');
                            } else {
                                folder.classList.remove('expanded');
                            }
                        }
                    }
                }
            });
        }
        
        // Initialize state on page load
        setTimeout(() => {
            applyTreeState();
        }, 0);
        
        // Handle folder clicks
        document.querySelectorAll('.tree-folder-header').forEach(header => {
            header.addEventListener('click', function(e) {
                e.preventDefault();
                const folder = this.parentElement;
                folder.classList.toggle('expanded');
                saveTreeState();
            });
        });
        
        // Search functionality
        const searchInput = document.getElementById('sidebar-search-input');
        if (searchInput) {
            let searchTimeout;
            
            function performTreeSearch() {
                const query = searchInput.value.toLowerCase().trim();
                const items = document.querySelectorAll('.tree-item');
                const folders = document.querySelectorAll('.tree-folder');
                
                // Clear previous matches
                document.querySelectorAll('.search-match').forEach(el => {
                    el.classList.remove('search-match');
                });
                
                if (!query) {
                    // Show all items
                    items.forEach(item => item.style.display = '');
                    folders.forEach(folder => folder.style.display = '');
                    applyTreeState();
                    return;
                }
                
                // Search and highlight
                let hasResults = false;
                items.forEach(item => {
                    const title = item.querySelector('.tree-title').textContent.toLowerCase();
                    if (title.includes(query)) {
                        item.style.display = '';
                        item.classList.add('search-match');
                        hasResults = true;
                        
                        // Expand parent folders
                        let parent = item.parentElement;
                        while (parent && parent !== treeNav) {
                            if (parent.classList.contains('tree-folder')) {
                                parent.classList.add('expanded');
                                parent.style.display = '';
                            }
                            parent = parent.parentElement;
                        }
                    } else {
                        item.style.display = 'none';
                    }
                });
                
                // Hide empty folders
                folders.forEach(folder => {
                    const visibleItems = folder.querySelectorAll('.tree-item:not([style*="none"])');
                    if (visibleItems.length === 0) {
                        folder.style.display = 'none';
                    }
                });
            }
            
            searchInput.addEventListener('input', function() {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(performTreeSearch, 250);
            });
            
            searchInput.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    this.value = '';
                    performTreeSearch();
                }
            });
        }
    }
    
    // Legacy flat list search (when jsTree is not used)
    const navList = document.querySelector('.nav-list:not(.jstree)');
    const isTreeNav = false; // jsTree handles tree navigation now
    
    if (navList && !document.querySelector('#jstree-container ul')) {
        // Only use legacy search for flat lists
        const searchInput = document.getElementById('sidebar-search-input');
        
        function performSearch() {
        const searchTerm = searchInput.value.toLowerCase().trim();
        let visibleCount = 0;
        
        // Show/hide clear button
        searchClear.style.display = searchTerm ? 'block' : 'none';
        
        if (isTreeNav) {
            // Tree navigation search for new structure
            const allFiles = navList.querySelectorAll('.nav-file');
            const directories = navList.querySelectorAll('.nav-directory');
            
            // Search through all file items
            allFiles.forEach(item => {
                const link = item.querySelector('a');
                const text = link ? link.textContent.toLowerCase() : '';
                
                if (!searchTerm || text.includes(searchTerm)) {
                    item.style.display = '';
                    visibleCount++;
                    // Show all parent directories
                    let parent = item.parentElement;
                    while (parent && parent !== navList) {
                        if (parent.classList.contains('nav-directory')) {
                            parent.style.display = '';
                            if (searchTerm) {
                                parent.classList.add('expanded');
                            }
                        }
                        parent = parent.parentElement;
                    }
                } else {
                    item.style.display = 'none';
                }
            });
            
            // Handle directories visibility
            directories.forEach(dir => {
                const hasVisibleFiles = dir.querySelectorAll('.nav-file:not([style*="none"])').length > 0;
                const hasVisibleSubDirs = dir.querySelectorAll('.nav-directory:not([style*="none"])').length > 0;
                
                if (!searchTerm) {
                    dir.style.display = '';
                } else if (!hasVisibleFiles && !hasVisibleSubDirs) {
                    dir.style.display = 'none';
                }
            });
        } else {
            // Flat navigation search
            const navItems = navList ? navList.querySelectorAll('li') : [];
            navItems.forEach(item => {
                const link = item.querySelector('a');
                const text = link ? link.textContent.toLowerCase() : '';
                
                if (!searchTerm || text.includes(searchTerm)) {
                    item.style.display = '';
                    visibleCount++;
                } else {
                    item.style.display = 'none';
                }
            });
        }
        
        // Show a message if no results found
        let noResultsMsg = document.getElementById('no-search-results');
        if (searchTerm && visibleCount === 0) {
            if (!noResultsMsg) {
                noResultsMsg = document.createElement('div');
                noResultsMsg.id = 'no-search-results';
                noResultsMsg.className = 'no-results';
                noResultsMsg.textContent = '没有找到匹配的结果';
                navList.parentNode.insertBefore(noResultsMsg, navList);
            }
            noResultsMsg.style.display = 'block';
        } else if (noResultsMsg) {
            noResultsMsg.style.display = 'none';
        }
        
        // Restore original expanded state when search is cleared
        if (!searchTerm && isTreeNav) {
            const directories = navList.querySelectorAll('.nav-directory');
            directories.forEach(dir => {
                // Check if directory contains active item
                const hasActive = dir.querySelector('.nav-subdirectory .active');
                if (hasActive) {
                    dir.classList.add('expanded');
                } else {
                    dir.classList.remove('expanded');
                }
            });
        }
    }
    
    if (searchInput) {
        // Perform search on input
        searchInput.addEventListener('input', performSearch);
        
        // Clear search when clicking X
        searchClear.addEventListener('click', function() {
            searchInput.value = '';
            performSearch();
            searchInput.focus();
        });
        
        // Clear search with Escape key
        searchInput.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                searchInput.value = '';
                performSearch();
            }
        });
    }
    }
});