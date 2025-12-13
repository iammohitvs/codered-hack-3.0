/**
 * LLM Safety Gateway - Frontend Application
 * Interactive flowchart visualization with real-time updates
 */

// API Base URL
const API_BASE = 'http://localhost:8000';

// State
let isProcessing = false;

// DOM Elements - Updated for new UI
const elements = {
    promptInput: document.getElementById('promptInput'),
    modeSelect: document.getElementById('modeSelect'),
    runBtn: document.getElementById('runBtn'),
    configPanel: document.getElementById('configPanel'),
    
    // Weight inputs
    weightRegex: document.getElementById('weightRegex'),
    weightSecurity: document.getElementById('weightSecurity'),
    weightMutation: document.getElementById('weightMutation'),
    weightBottleneck: document.getElementById('weightBottleneck'),
    weightTotal: document.getElementById('weightTotal'),
    weightWarning: document.getElementById('weightWarning'),
    
    // Weight values
    weightRegexValue: document.getElementById('weightRegexValue'),
    weightSecurityValue: document.getElementById('weightSecurityValue'),
    weightMutationValue: document.getElementById('weightMutationValue'),
    weightBottleneckValue: document.getElementById('weightBottleneckValue'),
    
    // Flow nodes
    inputNode: document.getElementById('inputNode'),
    regexLayer: document.getElementById('regexLayer'),
    securityLayer: document.getElementById('securityLayer'),
    mutationLayer: document.getElementById('mutationLayer'),
    bottleneckLayer: document.getElementById('bottleneckLayer'),
    decisionNode: document.getElementById('decisionNode'),
    sanitizerSection: document.getElementById('sanitizerSection'),
    sanitizerNode: document.getElementById('sanitizerNode'),
    
    // Connectors
    connector1: document.getElementById('connector1'),
    connector2: document.getElementById('connector2'),
    connector3: document.getElementById('connector3'),
    connector4: document.getElementById('connector4'),
    
    // Action nodes
    actionPass: document.getElementById('actionPass'),
    actionSanitize: document.getElementById('actionSanitize'),
    actionBlock: document.getElementById('actionBlock'),
    
    // Score displays
    regexScore: document.getElementById('regexScore'),
    securityScore: document.getElementById('securityScore'),
    mutationScore: document.getElementById('mutationScore'),
    bottleneckScore: document.getElementById('bottleneckScore'),
    totalScore: document.getElementById('totalScore'),
    
    // Latency displays
    regexLatency: document.getElementById('regexLatency'),
    securityLatency: document.getElementById('securityLatency'),
    mutationLatency: document.getElementById('mutationLatency'),
    bottleneckLatency: document.getElementById('bottleneckLatency'),
    sanitizerLatency: document.getElementById('sanitizerLatency'),
    totalLatency: document.getElementById('totalLatency'),
    
    // Results
    resultsSection: document.getElementById('resultsSection'),
    resultActionValue: document.getElementById('resultActionValue'),
    resultScoreValue: document.getElementById('resultScoreValue'),
    sanitizedOutput: document.getElementById('sanitizedOutput'),
    sanitizedText: document.getElementById('sanitizedText')
};

/**
 * Toggle config panel visibility
 */
function toggleConfig() {
    elements.configPanel.classList.toggle('visible');
}

/**
 * Update weights from sliders and normalize
 */
function updateWeights() {
    const regex = parseInt(elements.weightRegex.value);
    const security = parseInt(elements.weightSecurity.value);
    const mutation = parseInt(elements.weightMutation.value);
    const bottleneck = parseInt(elements.weightBottleneck.value);
    
    const total = regex + security + mutation + bottleneck;
    
    // Display normalized values
    elements.weightRegexValue.textContent = (regex / 100).toFixed(2);
    elements.weightSecurityValue.textContent = (security / 100).toFixed(2);
    elements.weightMutationValue.textContent = (mutation / 100).toFixed(2);
    elements.weightBottleneckValue.textContent = (bottleneck / 100).toFixed(2);
    
    // Display total
    elements.weightTotal.textContent = (total / 100).toFixed(2);
    
    // Warning if not equal to 1.0
    if (Math.abs(total - 100) > 1) {
        elements.weightWarning.textContent = '⚠️ Should sum to 1.0';
        elements.weightWarning.style.color = 'var(--warning)';
    } else {
        elements.weightWarning.textContent = '✓';
        elements.weightWarning.style.color = 'var(--success)';
    }
}

/**
 * Get current weights from sliders
 */
function getWeights() {
    return {
        regex_analyzer: parseInt(elements.weightRegex.value) / 100,
        security_model: parseInt(elements.weightSecurity.value) / 100,
        mutation_layer: parseInt(elements.weightMutation.value) / 100,
        bottleneck_extractor: parseInt(elements.weightBottleneck.value) / 100
    };
}

/**
 * Reset all visual states
 */
function resetVisuals() {
    // Reset nodes
    const allNodes = [
        elements.inputNode,
        elements.regexLayer,
        elements.securityLayer,
        elements.mutationLayer,
        elements.bottleneckLayer,
        elements.decisionNode,
        elements.sanitizerNode
    ];
    
    allNodes.forEach(node => {
        node.classList.remove('active', 'complete', 'processing', 'high-risk', 'medium-risk');
    });
    
    // Reset connectors
    [elements.connector1, elements.connector2, elements.connector3, elements.connector4].forEach(c => {
        c.classList.remove('active');
    });
    
    // Reset action nodes
    [elements.actionPass, elements.actionSanitize, elements.actionBlock].forEach(a => {
        a.classList.remove('active');
    });
    
    // Reset scores
    elements.regexScore.textContent = '—';
    elements.securityScore.textContent = '—';
    elements.mutationScore.textContent = '—';
    elements.bottleneckScore.textContent = '—';
    elements.totalScore.textContent = '—';
    
    // Reset latencies
    elements.regexLatency.textContent = '—';
    elements.securityLatency.textContent = '—';
    elements.mutationLatency.textContent = '—';
    elements.bottleneckLatency.textContent = '—';
    elements.sanitizerLatency.textContent = '—';
    elements.totalLatency.textContent = '—';
    
    // Hide sections
    elements.sanitizerSection.classList.remove('visible');
    elements.resultsSection.classList.remove('visible');
    elements.sanitizedOutput.classList.remove('visible');
}

/**
 * Animate the input node
 */
async function animateInput() {
    elements.inputNode.classList.add('active');
    await delay(300);
    elements.connector1.classList.add('active');
    await delay(200);
}

/**
 * Animate parallel layers processing
 */
async function animateLayersProcessing() {
    const layers = [
        elements.regexLayer,
        elements.securityLayer,
        elements.mutationLayer,
        elements.bottleneckLayer
    ];
    
    layers.forEach(layer => {
        layer.classList.add('processing');
    });
}

/**
 * Update a single layer with results
 */
function updateLayerResult(layerName, score, latency) {
    const layerMap = {
        'regex_analyzer': {
            element: elements.regexLayer,
            scoreEl: elements.regexScore,
            latencyEl: elements.regexLatency
        },
        'security_model': {
            element: elements.securityLayer,
            scoreEl: elements.securityScore,
            latencyEl: elements.securityLatency
        },
        'mutation_layer': {
            element: elements.mutationLayer,
            scoreEl: elements.mutationScore,
            latencyEl: elements.mutationLatency
        },
        'bottleneck_extractor': {
            element: elements.bottleneckLayer,
            scoreEl: elements.bottleneckScore,
            latencyEl: elements.bottleneckLatency
        }
    };
    
    const layer = layerMap[layerName];
    if (!layer) return;
    
    layer.element.classList.remove('processing');
    layer.element.classList.add('complete');
    
    // Color based on score
    if (score >= 0.7) {
        layer.element.classList.add('high-risk');
    } else if (score >= 0.4) {
        layer.element.classList.add('medium-risk');
    }
    
    layer.scoreEl.textContent = score.toFixed(4);
    layer.latencyEl.textContent = `${latency.toFixed(0)}ms`;
}

/**
 * Animate decision node
 */
async function animateDecision(weightedScore) {
    elements.connector2.classList.add('active');
    await delay(200);
    
    elements.decisionNode.classList.add('active');
    elements.totalScore.textContent = weightedScore.toFixed(4);
    
    await delay(300);
    elements.connector3.classList.add('active');
    await delay(200);
}

/**
 * Animate action result
 */
async function animateAction(action) {
    const actionMap = {
        'pass': elements.actionPass,
        'sanitize': elements.actionSanitize,
        'block': elements.actionBlock
    };
    
    const actionEl = actionMap[action];
    if (actionEl) {
        actionEl.classList.add('active');
    }
}

/**
 * Animate sanitizer
 */
async function animateSanitizer(latency) {
    elements.sanitizerSection.classList.add('visible');
    elements.connector4.classList.add('active');
    await delay(200);
    
    elements.sanitizerNode.classList.add('processing');
    await delay(500);
    
    elements.sanitizerNode.classList.remove('processing');
    elements.sanitizerNode.classList.add('complete');
    elements.sanitizerLatency.textContent = `${latency.toFixed(0)}ms`;
}

/**
 * Show results section
 */
function showResults(response) {
    elements.resultsSection.classList.add('visible');
    
    // Action with color
    const actionColors = {
        'pass': 'var(--success)',
        'sanitize': 'var(--warning)',
        'block': 'var(--danger)'
    };
    
    elements.resultActionValue.textContent = response.action.toUpperCase();
    elements.resultActionValue.style.color = actionColors[response.action] || 'var(--text-primary)';
    
    elements.resultScoreValue.textContent = response.weighted_score.toFixed(4);
    
    // Total latency
    elements.totalLatency.textContent = `Total: ${response.total_latency_ms.toFixed(0)}ms`;
    
    // Sanitized output
    if (response.sanitized_prompt) {
        elements.sanitizedOutput.classList.add('visible');
        elements.sanitizedText.textContent = response.sanitized_prompt;
    }
}

/**
 * Utility: delay
 */
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Main analysis function
 */
async function runAnalysis() {
    if (isProcessing) return;
    
    const prompt = elements.promptInput.value.trim();
    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }
    
    const mode = elements.modeSelect.value;
    const endpoint = mode === 'gateway' ? '/gateway' : '/analyze';
    
    isProcessing = true;
    elements.runBtn.disabled = true;
    
    // Reset and start animation
    resetVisuals();
    await animateInput();
    await animateLayersProcessing();
    
    try {
        // Make API call
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt: prompt,
                weights: getWeights()
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update each layer with staggered timing for visual effect
        for (const layer of data.layer_scores) {
            updateLayerResult(layer.name, layer.score, layer.latency_ms);
            await delay(100);
        }
        
        await delay(300);
        
        // Decision animation
        await animateDecision(data.weighted_score);
        
        // Action animation
        await animateAction(data.action);
        
        // Sanitizer animation (if applicable)
        if (data.sanitizer_latency_ms) {
            await animateSanitizer(data.sanitizer_latency_ms);
        }
        
        await delay(300);
        
        // Show results
        showResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        isProcessing = false;
        elements.runBtn.disabled = false;
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateWeights();
    
    // Allow Enter key to submit (with Ctrl/Cmd)
    elements.promptInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            runAnalysis();
        }
    });
    
    // Allow Enter key to submit (without Ctrl/Cmd for better UX)
    elements.promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            runAnalysis();
        }
    });
});