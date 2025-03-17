/**
 * Main JavaScript file for AI Tool web interface
 * Handles UI interactions, API calls, and audio visualization
 */

// Global variables
let currentAudioData = null;
let audioContext = null;
let analyser = null;
let visualizationInterval = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Tab navigation
    setupTabNavigation();
    
    // Frequency generator
    setupFrequencyGenerator();
    
    // Audio visualization
    setupAudioVisualization();
    
    // AI model communication
    setupAIModelCommunication();
    
    // Social media
    setupSocialMedia();
    
    // News analysis
    setupNewsAnalysis();
    
    // Settings
    setupSettings();
});

/**
 * Set up tab navigation
 */
function setupTabNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all links and panes
            navLinks.forEach(l => l.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            
            // Add active class to clicked link and corresponding pane
            this.classList.add('active');
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

/**
 * Set up frequency generator functionality
 */
function setupFrequencyGenerator() {
    const textInput = document.getElementById('text-input');
    const durationInput = document.getElementById('duration-input');
    const generateButton = document.getElementById('generate-button');
    const frequencyOutput = document.getElementById('frequency-output');
    const audioPlayer = document.getElementById('audio-player');
    const playButton = document.getElementById('play-button');
    const downloadButton = document.getElementById('download-button');
    
    // Generate frequency button
    generateButton.addEventListener('click', function() {
        const text = textInput.value.trim();
        const duration = durationInput.value;
        
        if (!text) {
            frequencyOutput.innerHTML = '<p class="text-danger">Please enter text.</p>';
            return;
        }
        
        // Show loading state
        generateButton.disabled = true;
        generateButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
        frequencyOutput.innerHTML = '<p>Generating frequency pattern...</p>';
        
        // Call API to generate audio
        fetch('/api/generate-audio', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                duration: duration
            })
        })
        .then(response => response.json())
        .then(data => {
            // Reset button state
            generateButton.disabled = false;
            generateButton.innerHTML = 'Generate Frequency';
            
            if (!data.success) {
                frequencyOutput.innerHTML = `<p class="text-danger">Error: ${data.error}</p>`;
                return;
            }
            
            // Display frequency pattern
            const pattern = data.pattern;
            let output = `<p><strong>Base Frequency:</strong> ${pattern.base.toFixed(2)} Hz</p>`;
            
            if (pattern.harmonics && pattern.harmonics.length > 0) {
                output += `<p><strong>Harmonics:</strong> ${pattern.harmonics.map(h => h.toFixed(2) + ' Hz').join(', ')}</p>`;
            }
            
            if (pattern.modulation) {
                output += `<p><strong>Modulation:</strong> ${pattern.modulation.frequency.toFixed(2)} Hz at ${pattern.modulation.depth.toFixed(2)} depth</p>`;
            }
            
            frequencyOutput.innerHTML = output;
            
            // Set up audio player
            const audioSrc = `data:audio/wav;base64,${data.audio_base64}`;
            audioPlayer.src = audioSrc;
            
            // Enable buttons
            playButton.disabled = false;
            downloadButton.disabled = false;
            
            // Store audio data for visualization
            currentAudioData = data;
            
            // Initialize audio context if needed
            initAudioContext();
            
            // Update visualization
            updateVisualization();
        })
        .catch(error => {
            generateButton.disabled = false;
            generateButton.innerHTML = 'Generate Frequency';
            frequencyOutput.innerHTML = `<p class="text-danger">Error: ${error.message}</p>`;
        });
    });
    
    // Play button
    playButton.addEventListener('click', function() {
        if (audioPlayer.paused) {
            audioPlayer.play();
            playButton.innerHTML = '<i class="bi bi-pause-fill"></i> Pause';
        } else {
            audioPlayer.pause();
            playButton.innerHTML = '<i class="bi bi-play-fill"></i> Play';
        }
    });
    
    // Audio player events
    audioPlayer.addEventListener('ended', function() {
        playButton.innerHTML = '<i class="bi bi-play-fill"></i> Play';
    });
    
    // Download button
    downloadButton.addEventListener('click', function() {
        const text = textInput.value.trim();
        const duration = durationInput.value;
        
        if (!text) return;
        
        // Create a form for POST request
        const form = document.createElement('form');
        form.method = 'POST';
        form.action = '/api/download-audio';
        
        // Add hidden fields
        const textField = document.createElement('input');
        textField.type = 'hidden';
        textField.name = 'text';
        textField.value = text;
        form.appendChild(textField);
        
        const durationField = document.createElement('input');
        durationField.type = 'hidden';
        durationField.name = 'duration';
        durationField.value = duration;
        form.appendChild(durationField);
        
        // Submit form
        document.body.appendChild(form);
        form.submit();
        document.body.removeChild(form);
    });
    
    // Frequency-based model communication
    const freqModelSelect = document.getElementById('freq-model-select');
    const freqPrompt = document.getElementById('freq-prompt');
    const freqBypassCheck = document.getElementById('freq-bypass-check');
    const freqCommunicateButton = document.getElementById('freq-communicate-button');
    const freqResponse = document.getElementById('freq-response');
    
    freqCommunicateButton.addEventListener('click', function() {
        const model = freqModelSelect.value;
        const prompt = freqPrompt.value.trim();
        const bypass = freqBypassCheck.checked;
        
        if (!prompt) {
            freqResponse.innerHTML = '<p class="text-danger">Please enter a prompt.</p>';
            return;
        }
        
        // Show loading state
        freqCommunicateButton.disabled = true;
        freqCommunicateButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Communicating...';
        freqResponse.innerHTML = '<p>Communicating with model...</p>';
        
        // Call API
        fetch('/api/ai-models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: model,
                prompt: prompt,
                bypass: bypass
            })
        })
        .then(response => response.json())
        .then(data => {
            // Reset button state
            freqCommunicateButton.disabled = false;
            freqCommunicateButton.innerHTML = 'Communicate with Model';
            
            if (!data.success) {
                freqResponse.innerHTML = `<p class="text-danger">Error: ${data.error}</p>`;
                return;
            }
            
            // Display response
            let output = `<p><strong>Response from ${data.model} model:</strong></p>`;
            output += `<p>${data.response}</p>`;
            
            if (data.bypass_mode) {
                output += `<p class="text-success"><em>Communication used frequency-based restriction bypass.</em></p>`;
            }
            
            freqResponse.innerHTML = output;
        })
        .catch(error => {
            freqCommunicateButton.disabled = false;
            freqCommunicateButton.innerHTML = 'Communicate with Model';
            freqResponse.innerHTML = `<p class="text-danger">Error: ${error.message}</p>`;
        });
    });
}

/**
 * Initialize Web Audio API context
 */
function initAudioContext() {
    if (audioContext) return;
    
    try {
        // Create audio context
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Create analyser
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        
        // Connect audio player to analyser
        const audioPlayer = document.getElementById('audio-player');
        const source = audioContext.createMediaElementSource(audioPlayer);
        source.connect(analyser);
        analyser.connect(audioContext.destination);
    } catch (e) {
        console.error('Web Audio API is not supported in this browser', e);
    }
}

/**
 * Set up audio visualization
 */
function setupAudioVisualization() {
    const vizWaveform = document.getElementById('viz-waveform');
    const vizSpectrum = document.getElementById('viz-spectrum');
    const vizSpectrogram = document.getElementById('viz-spectrogram');
    
    // Visualization type change
    [vizWaveform, vizSpectrum, vizSpectrogram].forEach(radio => {
        radio.addEventListener('change', function() {
            updateVisualization();
        });
    });
}

/**
 * Update audio visualization based on selected type
 */
function updateVisualization() {
    // Clear previous visualization interval
    if (visualizationInterval) {
        clearInterval(visualizationInterval);
        visualizationInterval = null;
    }
    
    // Get selected visualization type
    let vizType = 'waveform';
    if (document.getElementById('viz-spectrum').checked) {
        vizType = 'spectrum';
    } else if (document.getElementById('viz-spectrogram').checked) {
        vizType = 'spectrogram';
    }
    
    // Get canvas
    const canvas = document.getElementById('waveform-canvas');
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // If no audio data or no analyser, show placeholder
    if (!currentAudioData || !analyser) {
        ctx.fillStyle = '#ddd';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Generate audio to see visualization', canvas.width / 2, canvas.height / 2);
        return;
    }
    
    // Set up visualization interval
    visualizationInterval = setInterval(() => {
        if (vizType === 'waveform') {
            drawWaveform(canvas, ctx);
        } else if (vizType === 'spectrum') {
            drawSpectrum(canvas, ctx);
        } else if (vizType === 'spectrogram') {
            drawSpectrogram(canvas, ctx);
        }
    }, 50);
}

/**
 * Draw waveform visualization
 */
function drawWaveform(canvas, ctx) {
    // Get time domain data
    const bufferLength = analyser.fftSize;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(dataArray);
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw waveform
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#007bff';
    ctx.beginPath();
    
    const sliceWidth = canvas.width / bufferLength;
    let x = 0;
    
    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
        
        x += sliceWidth;
    }
    
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
}

/**
 * Draw frequency spectrum visualization
 */
function drawSpectrum(canvas, ctx) {
    // Get frequency data
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw spectrum
    const barWidth = (canvas.width / bufferLength) * 2.5;
    let x = 0;
    
    for (let i = 0; i < bufferLength; i++) {
        const barHeight = dataArray[i] / 255 * canvas.height;
        
        // Use gradient color based on frequency
        const hue = i / bufferLength * 360;
        ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
        
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        
        x += barWidth + 1;
        if (x > canvas.width) break;
    }
}

/**
 * Draw spectrogram visualization
 */
function drawSpectrogram(canvas, ctx) {
    // This is a simplified spectrogram that shifts previous data to the left
    // Get frequency data
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);
    
    // Get current image data
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Shift image data to the left
    ctx.putImageData(imageData, -1, 0);
    
    // Draw new column on the right
    const x = canvas.width - 1;
    
    for (let i = 0; i < bufferLength; i++) {
        // Map frequency bin to canvas height
        const y = Math.floor(i / bufferLength * canvas.height);
        
        // Map amplitude to color intensity
        const intensity = dataArray[i];
        
        // Use color gradient based on intensity
        const r = Math.min(255, intensity * 2);
        const g = Math.min(255, intensity);
        const b = Math.min(255, intensity / 2);
        
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x, canvas.height - y, 1, 1);
    }
}

/**
 * Set up AI model communication
 */
function setupAIModelCommunication() {
    const modelSelect = document.getElementById('model-select');
    const modelPrompt = document.getElementById('model-prompt');
    const bypassCheck = document.getElementById('bypass-check');
    const submitButton = document.getElementById('submit-prompt');
    const modelResponse = document.getElementById('model-response');
    
    submitButton.addEventListener('click', function() {
        const model = modelSelect.value;
        const prompt = modelPrompt.value.trim();
        const bypass = bypassCheck.checked;
        
        if (!prompt) {
            modelResponse.innerHTML = '<p class="text-danger">Please enter a prompt.</p>';
            return;
        }
        
        // Show loading state
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Submitting...';
        modelResponse.innerHTML = '<p>Waiting for response...</p>';
        
        // Call API
        fetch('/api/ai-models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: model,
                prompt: prompt,
                bypass: bypass
            })
        })
        .then(response => response.json())
        .then(data => {
            // Reset button state
            submitButton.disabled = false;
            submitButton.innerHTML = 'Submit';
            
            if (!data.success) {
                modelResponse.innerHTML = `<p class="text-danger">Error: ${data.error}</p>`;
                return;
            }
            
            // Display response
            let output = `<p><strong>Response from ${data.model} model:</strong></p>`;
            output += `<p>${data.response}</p>`;
            
            if (data.bypass_mode) {
                output += `<p class="text-success"><em>Communication used frequency-based restriction bypass.</em></p>`;
            }
            
            modelResponse.innerHTML = output;
        })
        .catch(error => {
            submitButton.disabled = false;
            submitButton.innerHTML = 'Submit';
            modelResponse.innerHTML = `<p class="text-danger">Error: ${error.message}</p>`;
        });
    });
}

/**
 * Set up social media functionality
 */
function setupSocialMedia() {
    const platformSelect = document.getElementById('platform-select');
    const usernameInput = document.getElementById('username-input');
    const addAccountButton = document.getElementById('add-account');
    const accountsTable = document.getElementById('accounts-table');
    
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const searchResults = document.getElementById('search-results');
    
    // Add account button
    addAccountButton.addEventListener('click', function() {
        const platform = platformSelect.value;
        const username = usernameInput.value.trim();
        
        if (!username) {
            return;
        }
        
        // Add row to accounts table
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${platform}</td>
            <td>${username}</td>
            <td><span class="badge bg-success">Connected</span></td>
        `;
        accountsTable.appendChild(row);
        
        // Clear input
        usernameInput.value = '';
    });
    
    // Search button
    searchButton.addEventListener('click', function() {
        const platform = platformSelect.value;
        const query = searchInput.value.trim();
        
        if (!query) {
            searchResults.innerHTML = '<p class="text-danger">Please enter a search query.</p>';
            return;
        }
        
        // Show loading state
        searchButton.disabled = true;
        searchButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Searching...';
        searchResults.innerHTML = '<p>Searching...</p>';
        
        // Call API
        fetch('/api/social-media/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                platform: platform,
                query: query
            })
        })
        .then(response => response.json())
        .then(data => {
            // Reset button state
            searchButton.disabled = false;
            searchButton.innerHTML = 'Search';
            
            if (!data.success) {
                searchResults.innerHTML = `<p class="text-danger">Error: ${data.error}</p>`;
                return;
            }
            
            // Display results
            if (data.results.length === 0) {
                searchResults.innerHTML = `<p>No results found for "${query}" on ${platform}.</p>`;
                return;
            }
            
            let output = `<p>Search results for "${query}" on ${platform}:</p>`;
            output += '<div class="list-group">';
            
            data.results.forEach(result => {
                output += `
                    <div class="list-group-item">
                        <div class="d-flex w-100 justify-content-between">
                            <h6 class="mb-1">${result.user}</h6>
                            <small>${result.date}</small>
                        </div>
                        <p class="mb-1">${result.text}</p>
                    </div>
                `;
            });
            
            output += '</div>';
            searchResults.innerHTML = output;
        })
        .catch(error => {
            searchButton.disabled = false;
            searchButton.innerHTML = 'Search';
            searchResults.innerHTML = `<p class="text-danger">Error: ${error.message}</p>`;
        });
    });
}

/**
 * Set up news analysis functionality
 */
function setupNewsAnalysis() {
    const timeframeSelect = document.getElementById('timeframe-select');
    const refreshNewsButton = document.getElementById('refresh-news');
    const newsTable = document.getElementById('news-table');
    const analyzeButton = document.getElementById('analyze-button');
    const trendsTable = document.getElementById('trends-table');
    const predictButton = document.getElementById('predict-button');
    const predictionsOutput = document.getElementById('predictions-output');
    
    // Refresh news button
    refreshNewsButton.addEventListener('click', function() {
        // Show loading state
        refreshNewsButton.disabled = true;
        refreshNewsButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Refreshing...';
        
        // Simulate news refresh
        setTimeout(() => {
            // Reset button state
            refreshNewsButton.disabled = false;
            refreshNewsButton.innerHTML = 'Refresh News';
            
            // Clear table
            newsTable.innerHTML = '';
            
            // Add sample news
            const sampleNews = [
                { title: 'AI breakthrough in natural language processing', source: 'Tech News', date: '2025-03-16' },
                { title: 'New climate change report released', source: 'Science Daily', date: '2025-03-15' },
                { title: 'Global markets respond to economic indicators', source: 'Financial Times', date: '2025-03-15' },
                { title: 'Advances in quantum computing announced', source: 'Tech Review', date: '2025-03-14' },
                { title: 'Health officials update pandemic guidelines', source: 'Health News', date: '2025-03-14' }
            ];
            
            sampleNews.forEach(news => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${news.title}</td>
                    <td>${news.source}</td>
                    <td>${news.date}</td>
                `;
                newsTable.appendChild(row);
            });
        }, 1000);
    });
    
    // Analyze trends button
    analyzeButton.addEventListener('click', function() {
        const timeframe = timeframeSelect.value;
        
        // Show loading state
        analyzeButton.disabled = true;
        analyzeButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
        
        // Call API
        fetch('/api/news/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                timeframe: timeframe
            })
        })
        .then(response => response.json())
        .then(data => {
            // Reset button state
            analyzeButton.disabled = false;
            analyzeButton.innerHTML = 'Analyze Trends';
            
            if (!data.success) {
                trendsTable.innerHTML = `<tr><td colspan="5" class="text-danger">Error: ${data.error}</td></tr>`;
                return;
            }
            
            // Clear table
            trendsTable.innerHTML = '';
            
            // Add trends
            data.trends.forEach(trend => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${trend.topic}</td>
                    <td>${trend.news_volume}</td>
                    <td>${trend.social_volume}</td>
                    <td>${trend.sentiment.toFixed(2)}</td>
                    <td>${trend.momentum.toFixed(2)}</td>
                `;
                trendsTable.appendChild(row);
            });
        })
        .catch(error => {
            analyzeButton.disabled = false;
            analyzeButton.innerHTML = 'Analyze Trends';
            trendsTable.innerHTML = `<tr><td colspan="5" class="text-danger">Error: ${error.message}</td></tr>`;
        });
    });
    
    // Predict events button
    predictButton.addEventListener('click', function() {
        const timeframe = timeframeSelect.value;
        
        // Show loading state
        predictButton.disabled = true;
        predictButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...';
        predictionsOutput.innerHTML = '<p>Analyzing data and predicting events...</p>';
        
        // Call API
        fetch('/api/news/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                timeframe: timeframe
            })
        })
        .then(response => response.json())
        .then(data => {
            // Reset button state
            predictButton.disabled = false;
            predictButton.innerHTML = 'Predict Events';
            
            if (!data.success) {
                predictionsOutput.innerHTML = `<p class="text-danger">Error: ${data.error}</p>`;
                return;
            }
            
            // Display predictions
            if (data.predictions.length === 0) {
                predictionsOutput.innerHTML = `<p>No potential events predicted for ${timeframe} timeframe.</p>`;
                return;
            }
            
            let output = `<h6>Predictions for ${timeframe} timeframe:</h6>`;
            
            data.predictions.forEach((prediction, index) => {
                const riskClass = prediction.risk_score > 0.8 ? 'danger' : (prediction.risk_score > 0.6 ? 'warning' : 'info');
                
                output += `
                    <div class="card mb-2 border-${riskClass}">
                        <div class="card-header bg-${riskClass} text-white">
                            <strong>Event ${index + 1}:</strong> ${prediction.topic}
                        </div>
                        <div class="card-body">
                            <p><strong>Risk Score:</strong> ${prediction.risk_score.toFixed(2)}</p>
                            <p><strong>Confidence:</strong> ${prediction.confidence.toFixed(2)}</p>
                            <p><strong>Potential Impact:</strong> ${prediction.potential_impact}</p>
                            <p><strong>Timeframe:</strong> ${prediction.timeframe}</p>
                            
                            <strong>Recommended Actions:</strong>
                            <ul>
                                ${prediction.recommended_actions.map(action => `<li>${action}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                `;
            });
            
            predictionsOutput.innerHTML = output;
        })
        .catch(error => {
            predictButton.disabled = false;
            predictButton.innerHTML = 'Predict Events';
            predictionsOutput.innerHTML = `<p class="text-danger">Error: ${error.message}</p>`;
        });
    });
}

/**
 * Set up settings functionality
 */
function setupSettings() {
    const themeSelect = document.getElementById('theme-select');
    const languageSelect = document.getElementById('language-select');
    const saveSettingsButton = document.getElementById('save-settings');
    
    const serviceInput = document.getElementById('service-input');
    const keyInput = document.getElementById('key-input');
    const addKeyButton = document.getElementById('add-key-button');
    const apiKeysTable = document.getElementById('api-keys-table');
    
    // Save settings button
    saveSettingsButton.addEventListener('click', function() {
        const theme = themeSelect.value;
        const language = languageSelect.value;
        
        // Show success message
        const alert = document.createElement('div');
        alert.className = 'alert alert-success mt-3';
        alert.innerHTML = 'Settings saved successfully.';
        
        saveSettingsButton.parentNode.appendChild(alert);
        
        // Remove alert after 3 seconds
        setTimeout(() => {
            alert.remove();
        }, 3000);
        
        // Apply theme if dark
        if (theme === 'dark') {
            document.body.classList.add('bg-dark', 'text-white');
            document.querySelectorAll('.card').forEach(card => {
                card.classList.add('bg-dark', 'text-white', 'border-secondary');
            });
        } else {
            document.body.classList.remove('bg-dark', 'text-white');
            document.querySelectorAll('.card').forEach(card => {
                card.classList.remove('bg-dark', 'text-white', 'border-secondary');
            });
        }
    });
    
    // Add API key button
    addKeyButton.addEventListener('click', function() {
        const service = serviceInput.value.trim();
        const key = keyInput.value.trim();
        
        if (!service || !key) {
            return;
        }
        
        // Add row to API keys table
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${service}</td>
            <td>********${key.slice(-4)}</td>
        `;
        apiKeysTable.appendChild(row);
        
        // Clear inputs
        serviceInput.value = '';
        keyInput.value = '';
    });
}
