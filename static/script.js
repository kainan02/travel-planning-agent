document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded, initializing script...');
    
    const form = document.getElementById('travelForm');
    const generateBtn = document.getElementById('generateBtn');
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.querySelector('.btn-loader');
    const resultContainer = document.getElementById('resultContainer');
    const errorContainer = document.getElementById('errorContainer');
    const planContent = document.getElementById('planContent');
    const copyBtn = document.getElementById('copyBtn');
    
    // Search functionality
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const searchResults = document.getElementById('searchResults');
    
    // Debug information display
    const debugContainer = document.getElementById('debugContainer');
    const debugContent = document.getElementById('debugContent');
    
    // Set custom validation messages in English
    const daysInput = document.getElementById('days');
    const destinationInput = document.getElementById('destination');
    
    // Set custom validation messages in English for all required fields
    if (daysInput) {
        daysInput.addEventListener('invalid', function(e) {
            if (daysInput.validity.valueMissing) {
                daysInput.setCustomValidity('Please enter the number of travel days');
            } else if (daysInput.validity.rangeUnderflow) {
                daysInput.setCustomValidity('Please enter a number greater than or equal to 1');
            } else if (daysInput.validity.rangeOverflow) {
                daysInput.setCustomValidity('Please enter a number less than or equal to 30');
            } else {
                daysInput.setCustomValidity('Please enter a valid number');
            }
        });
        
        daysInput.addEventListener('input', function() {
            daysInput.setCustomValidity('');
        });
    }
    
    if (destinationInput) {
        destinationInput.addEventListener('invalid', function(e) {
            if (destinationInput.validity.valueMissing) {
                destinationInput.setCustomValidity('Please enter your travel destination');
            } else {
                destinationInput.setCustomValidity('Please enter a valid destination');
            }
        });
        
        destinationInput.addEventListener('input', function() {
            destinationInput.setCustomValidity('');
        });
    }
    
    // Debug log function
    function debugLog(message, data = null) {
        const timestamp = new Date().toLocaleTimeString();
        let logMessage = `[${timestamp}] ${message}`;
        if (data) {
            logMessage += '\n' + JSON.stringify(data, null, 2);
        }
        
        // Display on page
        if (debugContainer && debugContent) {
            debugContainer.style.display = 'block';
            debugContent.textContent += logMessage + '\n\n';
            debugContent.scrollTop = debugContent.scrollHeight;
        }
        
        // Also output to console
        console.log(message, data || '');
    }
    
    // Check if elements exist
    const elementsCheck = {
        searchInput: !!searchInput,
        searchBtn: !!searchBtn,
        searchResults: !!searchResults
    };
    debugLog('Search elements check', elementsCheck);
    
    if (!searchInput || !searchBtn || !searchResults) {
        debugLog('Error: Search elements not found', elementsCheck);
    }

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Hide previous results and errors
        resultContainer.style.display = 'none';
        errorContainer.style.display = 'none';
        
        // Validate form fields first
        if (!form.checkValidity()) {
            // Trigger validation messages
            form.reportValidity();
            return;
        }
        
        // Get form data
        const budgetAmount = document.getElementById('budget').value.trim();
        const currency = document.getElementById('currency').value;
        
        // Validate budget: if amount is entered, currency must be selected
        if (budgetAmount && !currency) {
            showError('Please select a currency if you enter a budget amount');
            return;
        }
        
        // Combine budget amount and currency
        let budget = '';
        if (budgetAmount && currency) {
            budget = `${budgetAmount} ${currency}`;
        }
        
        const formData = {
            days: document.getElementById('days').value,
            destination: document.getElementById('destination').value,
            budget: budget,
            preferences: document.getElementById('preferences').value,
            llm_mode: document.getElementById('llmMode').value
        };
        
        // 如果使用本地 LLM 且有搜索上下文，将搜索信息添加到请求中
        if (formData.llm_mode === 'local' && searchContextData && searchContextData.summary) {
            formData.search_context = searchContextData.summary;
            formData.references = searchContextData.references;
            debugLog('Including search context in plan generation request', {
                hasSummary: !!formData.search_context,
                referencesCount: formData.references ? formData.references.length : 0
            });
        }
        
        // Show loading state
        generateBtn.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline';
        const isLocalMode = formData.llm_mode === 'local';
        if (isLocalMode) {
            btnLoader.textContent = 'Generating with local LLM (this may take 3-10 minutes, please be patient)...';
        } else {
            btnLoader.textContent = 'Generating your travel plan, please wait...';
        }
        
        try {
            debugLog('Preparing to send generate plan request', formData);
            
            // Set timeout (10 minutes for local LLM, 2 minutes for cloud)
            const isLocalMode = formData.llm_mode === 'local';
            const timeoutDuration = isLocalMode ? 600000 : 120000; // 10 minutes for local, 2 minutes for cloud
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeoutDuration);
            
            debugLog('Sending request to /api/generate-plan...');
            const response = await fetch('/api/generate-plan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData),
                signal: controller.signal,
                credentials: 'same-origin'
            });
            
            clearTimeout(timeoutId);
            
            debugLog('Received server response', { status: response.status, ok: response.ok, statusText: response.statusText });
            
            if (!response.ok) {
                const errorText = await response.text();
                debugLog('Server returned error', { status: response.status, error: errorText });
                throw new Error(`HTTP Error: ${response.status} - ${errorText.substring(0, 100)}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                // Store the original plan text for copying and exporting
                const originalPlanText = data.plan;
                const destination = document.getElementById('destination').value;
                
                // Display results
                resultContainer.style.display = 'block';
                
                // Convert markdown format to HTML (simple processing)
                planContent.innerHTML = formatPlan(data.plan);
                
                // Store original text and destination in data attributes
                planContent.setAttribute('data-original-text', originalPlanText);
                planContent.setAttribute('data-destination', destination);
                
                // Show export options
                const exportOptions = document.getElementById('exportOptions');
                if (exportOptions) {
                    exportOptions.style.display = 'block';
                    // Set default start date to tomorrow
                    const tomorrow = new Date();
                    tomorrow.setDate(tomorrow.getDate() + 1);
                    const startDateInput = document.getElementById('startDate');
                    if (startDateInput) {
                        startDateInput.value = tomorrow.toISOString().split('T')[0];
                    }
                }
                
                // Plan generation doesn't use search, so no reference links
                
                // Scroll to result area
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
                // Display error
                let errorMsg = data.error || 'Error generating plan, please try again later';
                if (data.detail) {
                    console.error('Detailed error information:', data.detail);
                }
                showError(errorMsg);
            }
        } catch (error) {
            console.error('Error:', error);
            debugLog('Error generating plan', { 
                name: error.name, 
                message: error.message,
                stack: error.stack 
            });
            
            let errorMsg = 'Error generating plan: ';
            if (error.name === 'AbortError') {
                const timeoutMinutes = formData.llm_mode === 'local' ? 10 : 2;
                errorMsg = `Request timeout (exceeded ${timeoutMinutes} minutes). Local LLM may need more time. Please try again or use Cloud mode for faster response.`;
            } else if (error.message === 'Failed to fetch') {
                errorMsg = 'Unable to connect to server. Please check:\n1. Is the server running (run python app.py)\n2. Is the server address correct\n3. Is the network connection normal';
            } else {
                errorMsg += error.message;
            }
            
            showError(errorMsg);
        } finally {
            // Restore button state
            generateBtn.disabled = false;
            btnText.style.display = 'inline';
            btnLoader.style.display = 'none';
        }
    });

    // Search functionality
    if (searchBtn) {
        searchBtn.addEventListener('click', async function(e) {
            e.preventDefault(); // Prevent default behavior
            debugLog('Search button clicked');
            
            const query = searchInput ? searchInput.value.trim() : '';
            if (!query) {
                alert('Please enter a search keyword');
                debugLog('Error: Search keyword is empty');
                return;
            }
            
            debugLog('Starting search', { query: query });
            searchBtn.disabled = true;
            searchBtn.textContent = 'Searching and summarizing, please wait...';
            if (searchResults) {
                searchResults.style.display = 'none';
            }
        
        try {
            debugLog('Sending search request to server...');
            debugLog('Request URL: /api/search');
            debugLog('Request data', { query: query });
            
            // 尝试从表单中获取目的地（如果已填写）
            const destinationInput = document.getElementById('destination');
            const destination = destinationInput ? destinationInput.value.trim() : '';
            
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    query: query,
                    destination: destination  // 传递目的地信息（如果已填写）
                }),
                credentials: 'same-origin'  // Ensure cookies are sent
            });
            
            debugLog('Received server response', { status: response.status, ok: response.ok, statusText: response.statusText });
            
            if (!response.ok) {
                throw new Error(`HTTP Error: ${response.status}`);
            }
            
            const data = await response.json();
            debugLog('Parsed response data', { success: data.success, hasSummary: !!data.summary, referencesCount: data.references ? data.references.length : 0 });
            
            displaySearchResults(data);
            if (searchResults) {
                searchResults.style.display = 'block';
                searchResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            debugLog('Search completed, displaying results');
        } catch (error) {
            debugLog('Search error', { 
                name: error.name, 
                message: error.message,
                stack: error.stack 
            });
            
            // Provide more friendly error message
            let errorMsg = 'Error during search: ';
            if (error.message === 'Failed to fetch') {
                errorMsg = 'Unable to connect to server. Please check:\n1. Is the server running\n2. Is the network connection normal\n3. Is the server address correct';
            } else {
                errorMsg += error.message;
            }
            
            alert(errorMsg);
        } finally {
            if (searchBtn) {
                searchBtn.disabled = false;
                searchBtn.textContent = 'Search';
            }
        }
        });
    } else {
        debugLog('Error: Search button element not found, cannot bind event');
    }
    
    // Support Enter key search
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                if (searchBtn) {
                    searchBtn.click();
                }
            }
        });
    }
    
    // 存储搜索上下文，用于后续生成计划时使用 RAG
    let searchContextData = null;
    
    function displaySearchResults(data) {
        if (!data.success) {
            searchResults.innerHTML = `<div class="error-container"><div class="error-message">${data.error || 'Search failed'}</div></div>`;
            // 清除搜索上下文
            searchContextData = null;
            return;
        }
        
        // 保存搜索上下文数据，用于后续生成计划
        searchContextData = {
            summary: data.summary || '',
            references: data.references || []
        };
        
        // Display AI summary results
        let html = '<div style="background: white; padding: 25px; border-radius: 10px; margin-bottom: 20px;">';
        html += '<h3 style="color: #1459CF; margin-bottom: 15px; font-size: 1.4em;">Search Results Summary</h3>';
        
        // If there's an error, display a warning
        if (data.summary_error) {
            html += '<div style="background: #F8F2EC; border: 2px solid #A9423C; padding: 15px; border-radius: 8px; margin-bottom: 15px; color: #A9423C;">';
            html += '<strong>Warning:</strong> AI summary failed. Showing search results instead.';
            html += '</div>';
        }
        
        html += '<div style="line-height: 1.8; color: #120E10; white-space: pre-wrap; margin-bottom: 20px;">';
        html += data.summary || 'No summary available';
        html += '</div>';
        
        // 添加提示信息：搜索信息将用于生成计划（仅在使用本地 LLM 时）
        html += '<div style="background: #E8F4F8; border: 1px solid #7097DE; padding: 12px; border-radius: 8px; margin-top: 15px; color: #1459CF; font-size: 0.9em;">';
        html += '<strong>💡 Tip:</strong> When you generate a travel plan using local LLM mode, this search information will be automatically included to create a more accurate and detailed plan.';
        html += '</div>';
        html += '</div>';
        
        // Display reference links
        if (data.references && data.references.length > 0) {
            html += '<div style="background: #F8F2EC; padding: 20px; border-radius: 10px; border-top: 2px solid #7097DE;">';
            html += '<h3 style="color: #1459CF; margin-bottom: 15px; font-size: 1.2em;">Reference Sources</h3>';
            html += '<p style="color: #120E10; margin-bottom: 15px;">The following articles were found from web search. Click the links to view the original articles:</p>';
            html += '<ul style="list-style: none; padding: 0;">';
            data.references.forEach((ref, index) => {
                if (ref.link) {
                    html += `<li style="margin-bottom: 10px;"><a href="${ref.link}" target="_blank" style="color: #1459CF; text-decoration: none; word-break: break-all;">${index + 1}. ${ref.title || ref.link}</a></li>`;
                } else {
                    html += `<li style="margin-bottom: 10px; color: #120E10;">${index + 1}. ${ref.title || 'No title'}</li>`;
                }
            });
            html += '</ul>';
            html += '<p style="color: #120E10; opacity: 0.7; font-size: 0.9em; margin-top: 15px; font-style: italic;">*Note: The above links are for reference only. Please verify the actual information.*</p>';
            html += '</div>';
        }
        
        searchResults.innerHTML = html;
    }

    // Copy functionality
    if (copyBtn) {
        copyBtn.addEventListener('click', function() {
            // Get text content - try multiple methods to ensure we get the text
            let text = '';
            
            // First try: get original text from data attribute (most reliable)
            if (planContent.getAttribute('data-original-text')) {
                text = planContent.getAttribute('data-original-text');
            }
            // Second try: get text from the element directly
            else if (planContent.textContent) {
                text = planContent.textContent;
            } else if (planContent.innerText) {
                text = planContent.innerText;
            } else {
                // Fallback: create a temporary element to extract text
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = planContent.innerHTML;
                text = tempDiv.textContent || tempDiv.innerText || '';
            }
            
            // Remove extra whitespace but preserve line breaks
            text = text.trim();
            
            if (!text) {
                alert('No content to copy. Please generate a travel plan first.');
                return;
            }
            
            // Use Clipboard API
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(function() {
                    const originalText = copyBtn.textContent;
                    copyBtn.textContent = 'Copied!';
                    copyBtn.style.background = '#28a745';
                    setTimeout(function() {
                        copyBtn.textContent = originalText;
                        copyBtn.style.background = '#1D4C50';
                    }, 2000);
                }).catch(function(err) {
                    console.error('Copy failed:', err);
                    // Fallback: use execCommand
                    fallbackCopyTextToClipboard(text);
                });
            } else {
                // Fallback for browsers that don't support Clipboard API
                fallbackCopyTextToClipboard(text);
            }
        });
    }

    // Export to ICS functionality
    const exportIcsBtn = document.getElementById('exportIcsBtn');
    if (exportIcsBtn) {
        exportIcsBtn.addEventListener('click', async function() {
            const planText = planContent.getAttribute('data-original-text');
            const destination = planContent.getAttribute('data-destination') || 'Travel Plan';
            const startDateInput = document.getElementById('startDate');
            const startDate = startDateInput ? startDateInput.value : null;
            
            if (!planText) {
                alert('No travel plan to export. Please generate a travel plan first.');
                return;
            }
            
            try {
                const response = await fetch('/api/export-ics', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        plan: planText,
                        destination: destination,
                        start_date: startDate
                    }),
                    credentials: 'same-origin'
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ error: 'Failed to export calendar' }));
                    throw new Error(errorData.error || 'Failed to export calendar');
                }
                
                // Get the file blob
                const blob = await response.blob();
                
                // Create download link
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `travel_plan_${destination.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.ics`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                alert('Calendar file exported successfully! You can import it into your calendar application.');
            } catch (error) {
                console.error('Export error:', error);
                alert(`Failed to export calendar: ${error.message}`);
            }
        });
    }
    
    // Fallback copy function for older browsers
    function fallbackCopyTextToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            const successful = document.execCommand('copy');
            if (successful) {
                const originalText = copyBtn.textContent;
                copyBtn.textContent = 'Copied!';
                copyBtn.style.background = '#28a745';
                setTimeout(function() {
                    copyBtn.textContent = originalText;
                    copyBtn.style.background = '#1D4C50';
                }, 2000);
            } else {
                alert('Copy failed. Please manually select and copy the text.');
            }
        } catch (err) {
            console.error('Fallback copy failed:', err);
            alert('Copy failed. Please manually select and copy the text.');
        } finally {
            document.body.removeChild(textArea);
        }
    }

    function showError(message) {
        errorContainer.style.display = 'block';
        errorContainer.querySelector('.error-message').textContent = message;
        errorContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function formatPlan(text) {
        // Simple markdown to HTML conversion
        let html = text;
        
        // Headings
        html = html.replace(/^## (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^### (.*$)/gim, '<h4>$1</h4>');
        
        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // List items
        html = html.replace(/^\- (.*$)/gim, '<li>$1</li>');
        
        // Wrap consecutive list items in ul tags
        html = html.replace(/(<li>.*<\/li>\n?)+/g, function(match) {
            return '<ul>' + match.replace(/\n/g, '') + '</ul>';
        });
        
        // Line breaks
        html = html.replace(/\n\n/g, '</p><p>');
        html = '<p>' + html + '</p>';
        
        return html;
    }
});

