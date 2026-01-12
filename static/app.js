// State management
let selectedFiles = [];
let isProcessing = false;

// DOM elements
const fileInput = document.getElementById('file-input');
const fileList = document.getElementById('file-list');
const ingestBtn = document.getElementById('ingest-btn');
const clearBtn = document.getElementById('clear-btn');
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');
const chatMessages = document.getElementById('chat-messages');
const topKInput = document.getElementById('top-k');
const thresholdInput = document.getElementById('threshold');
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const docCount = document.getElementById('doc-count');
const statsContent = document.getElementById('stats-content');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadStats();
    setupEventListeners();

    // Auto-resize textarea
    queryInput.addEventListener('input', () => {
        queryInput.style.height = 'auto';
        queryInput.style.height = queryInput.scrollHeight + 'px';
    });
});

// Event Listeners
function setupEventListeners() {
    fileInput.addEventListener('change', handleFileSelect);
    ingestBtn.addEventListener('click', handleIngest);
    clearBtn.addEventListener('click', handleClear);
    sendBtn.addEventListener('click', handleQuery);

    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleQuery();
        }
    });
}

// File handling
function handleFileSelect(e) {
    const files = Array.from(e.target.files);

    files.forEach(file => {
        if (file.type === 'application/pdf') {
            selectedFiles.push(file);
        } else {
            showNotification('Only PDF files are allowed', 'error');
        }
    });

    updateFileList();
    ingestBtn.disabled = selectedFiles.length === 0;
}

function updateFileList() {
    if (selectedFiles.length === 0) {
        fileList.innerHTML = '<p style="color: var(--text-secondary); font-size: 0.875rem;">No files selected</p>';
        return;
    }

    fileList.innerHTML = selectedFiles.map((file, index) => `
        <div class="file-item">
            <span class="file-name" title="${file.name}">${file.name}</span>
            <span class="file-remove" onclick="removeFile(${index})">✕</span>
        </div>
    `).join('');
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateFileList();
    ingestBtn.disabled = selectedFiles.length === 0;
}

// API calls
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        if (data.ollama_connected && data.groq_connected) {
            statusIndicator.className = 'status-dot connected';
            statusText.textContent = 'Connected';
            sendBtn.disabled = false;
        } else if (!data.groq_connected) {
            statusIndicator.className = 'status-dot disconnected';
            statusText.textContent = 'Groq not connected';
            sendBtn.disabled = true;
            showNotification('Groq is not reachable or the model is unavailable. Check your API key and model id.', 'error');
        } else {
            statusIndicator.className = 'status-dot disconnected';
            statusText.textContent = 'Ollama not connected';
            sendBtn.disabled = true;
            showNotification('Ollama is not running. Please start Ollama to use embeddings.', 'error');
        }

        docCount.textContent = `${data.total_documents} documents`;
    } catch (error) {
        statusIndicator.className = 'status-dot disconnected';
        statusText.textContent = 'Connection error';
        sendBtn.disabled = true;
    }
}

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        if (data.total_documents === 0) {
            statsContent.innerHTML = '<p>No documents loaded</p>';
        } else {
            statsContent.innerHTML = `
                <p><strong>Total Documents:</strong> ${data.total_documents}</p>
                <p><strong>Embedding Dimension:</strong> ${data.embedding_dimension}</p>
                <p><strong>Source Files:</strong></p>
                <ul style="margin-left: 1rem; margin-top: 0.5rem;">
                    ${data.source_files.map(f => `<li>${f}</li>`).join('')}
                </ul>
            `;
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

async function handleIngest() {
    if (selectedFiles.length === 0 || isProcessing) return;

    isProcessing = true;
    ingestBtn.disabled = true;
    ingestBtn.textContent = 'Ingesting...';

    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });

    try {
        const response = await fetch('/api/ingest', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Ingestion failed');
        }

        const data = await response.json();

        showNotification(
            `Successfully ingested ${data.files_processed} files (${data.chunks_created} chunks)`,
            'success'
        );

        // Clear file selection
        selectedFiles = [];
        fileInput.value = '';
        updateFileList();

        // Refresh stats
        await checkHealth();
        await loadStats();

    } catch (error) {
        showNotification(`Ingestion error: ${error.message}`, 'error');
    } finally {
        isProcessing = false;
        ingestBtn.textContent = 'Ingest Documents';
        ingestBtn.disabled = selectedFiles.length === 0;
    }
}

async function handleClear() {
    if (!confirm('Are you sure you want to clear the entire knowledge base?')) {
        return;
    }

    try {
        const response = await fetch('/api/clear', { method: 'DELETE' });

        if (!response.ok) {
            throw new Error('Failed to clear knowledge base');
        }

        showNotification('Knowledge base cleared successfully', 'success');

        // Clear chat
        chatMessages.innerHTML = `
            <div class="welcome-message">
                <h2>Welcome to the Agentic RAG System</h2>
                <p>Upload PDF documents to get started, then ask questions about them!</p>
            </div>
        `;

        // Refresh stats
        await checkHealth();
        await loadStats();

    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
    }
}

async function handleQuery() {
    const query = queryInput.value.trim();
    if (!query || isProcessing) return;

    isProcessing = true;
    sendBtn.disabled = true;

    // Add user message
    addMessage('user', query);

    // Clear input
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // Add loading message
    const loadingId = addLoadingMessage();

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                top_k: parseInt(topKInput.value),
                similarity_threshold: parseFloat(thresholdInput.value)
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Query failed');
        }

        const data = await response.json();

        // Remove loading message
        removeLoadingMessage(loadingId);

        // Add assistant response
        addMessage('assistant', data.answer, data);

    } catch (error) {
        removeLoadingMessage(loadingId);
        addMessage('assistant', `Error: ${error.message}`);
        showNotification(`Query error: ${error.message}`, 'error');
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
    }
}

// UI helpers
function addMessage(role, content, metadata = null) {
    // Remove welcome message if present
    const welcomeMsg = chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;

    let metadataHTML = '';
    if (metadata) {
        metadataHTML = `
            <div class="message-metadata">
                <div class="metadata-item"><strong>Intent:</strong> ${metadata.intent}</div>
                ${metadata.query_transformed ? `<div class="metadata-item"><strong>Transformed Query:</strong> ${metadata.query_transformed}</div>` : ''}
                ${metadata.metadata ? `<div class="metadata-item"><strong>Chunks Retrieved:</strong> ${metadata.metadata.chunks_retrieved || 0}</div>` : ''}
                ${metadata.metadata ? `<div class="metadata-item"><strong>Chunks Used:</strong> ${metadata.metadata.chunks_used || 0}</div>` : ''}
            </div>
        `;

        if (metadata.citations && metadata.citations.length > 0) {
            metadataHTML += `
                <div class="citations">
                    <div class="citations-title">📚 Citations:</div>
                    ${metadata.citations.map(citation => `
                        <div class="citation">
                            <div class="citation-source">
                                ${citation.source_file}
                                ${citation.page_number ? ` (Page ${citation.page_number})` : ''}
                                - Score: ${citation.similarity_score.toFixed(3)}
                            </div>
                            <div class="citation-preview">${citation.text_preview}</div>
                        </div>
                    `).join('')}
                </div>
            `;
        }
    }

    messageDiv.innerHTML = `
        <div class="message-header">
            ${role === 'user' ? '👤 You' : '🤖 Assistant'}
        </div>
        <div class="message-content">
            ${content}
            ${metadataHTML}
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addLoadingMessage() {
    const loadingDiv = document.createElement('div');
    const id = 'loading-' + Date.now();
    loadingDiv.id = id;
    loadingDiv.className = 'message assistant-message';
    loadingDiv.innerHTML = `
        <div class="message-header">🤖 Assistant</div>
        <div class="message-content">
            <div class="loading">
                <span>Processing your query</span>
                <div class="loading-dots">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
            </div>
        </div>
    `;

    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return id;
}

function removeLoadingMessage(id) {
    const loadingDiv = document.getElementById(id);
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

function showNotification(message, type = 'info') {
    // Simple notification - could be enhanced with a toast library
    console.log(`[${type.toUpperCase()}] ${message}`);

    // You could add a toast notification here
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'error' ? '#fee2e2' : type === 'success' ? '#dcfce7' : '#dbeafe'};
        color: ${type === 'error' ? '#991b1b' : type === 'success' ? '#166534' : '#1e40af'};
        border-radius: 8px;
        box-shadow: var(--shadow-lg);
        z-index: 1000;
        max-width: 400px;
        animation: slideIn 0.3s ease-out;
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}
