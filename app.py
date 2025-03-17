"""
Web application setup for AI Tool with Flask and audio capabilities

This file sets up a Flask web application for the AI Tool with audio output
for the frequency generator.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import numpy as np
import io
import base64
from scipy.io.wavfile import write as write_wav
from scipy import signal

# Import existing modules
from api_connections import APIConnectionManager
from social_media_integration import SocialMediaManager, AccountManager
from news_analysis import NewsFetcher, TrendAnalyzer, EventPredictor, AdvancedAlgorithm
from frequency_generator import FrequencyGenerator, TextToFrequency

# Initialize Flask app
app = Flask(__name__)

# Initialize components
api_manager = APIConnectionManager()
social_manager = SocialMediaManager(api_manager)
account_manager = AccountManager(social_manager)
news_fetcher = NewsFetcher()
trend_analyzer = TrendAnalyzer(news_fetcher, social_manager)
event_predictor = EventPredictor(trend_analyzer)
algorithm = AdvancedAlgorithm(event_predictor)
frequency_generator = FrequencyGenerator()

# Audio generation class
class AudioGenerator:
    """Generates audio from frequency patterns"""
    
    def __init__(self):
        self.sample_rate = 44100  # Hz
        
    def generate_audio(self, frequency_pattern, duration=3.0):
        """Generate audio from a frequency pattern"""
        try:
            # Extract pattern components
            base_freq = frequency_pattern.get("base", 440.0)
            harmonics = frequency_pattern.get("harmonics", [])
            modulation = frequency_pattern.get("modulation", {})
            envelope = frequency_pattern.get("envelope", {})
            
            # Create time array
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            
            # Initialize audio signal
            audio = np.zeros_like(t)
            
            # Add base frequency
            audio += 0.5 * np.sin(2 * np.pi * base_freq * t)
            
            # Add harmonics
            for i, harmonic in enumerate(harmonics):
                # Decrease amplitude for higher harmonics
                amplitude = 0.5 / (i + 2)
                audio += amplitude * np.sin(2 * np.pi * harmonic * t)
            
            # Apply modulation if specified
            if modulation:
                mod_freq = modulation.get("frequency", base_freq / 10)
                mod_depth = modulation.get("depth", 0.3)
                
                # Apply frequency modulation
                modulator = mod_depth * np.sin(2 * np.pi * mod_freq * t)
                audio = np.sin(2 * np.pi * base_freq * t + modulator)
            
            # Apply envelope
            if envelope:
                attack = envelope.get("attack", 0.01)
                decay = envelope.get("decay", 0.1)
                sustain = envelope.get("sustain", 0.7)
                release = envelope.get("release", 0.2)
                
                # Create ADSR envelope
                attack_samples = int(attack * self.sample_rate)
                decay_samples = int(decay * self.sample_rate)
                release_samples = int(release * self.sample_rate)
                sustain_samples = len(t) - attack_samples - decay_samples - release_samples
                
                env = np.zeros_like(t)
                
                # Attack phase
                if attack_samples > 0:
                    env[:attack_samples] = np.linspace(0, 1, attack_samples)
                
                # Decay phase
                if decay_samples > 0:
                    env[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain, decay_samples)
                
                # Sustain phase
                if sustain_samples > 0:
                    env[attack_samples+decay_samples:attack_samples+decay_samples+sustain_samples] = sustain
                
                # Release phase
                if release_samples > 0:
                    env[-release_samples:] = np.linspace(sustain, 0, release_samples)
                
                # Apply envelope
                audio = audio * env
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Convert to 16-bit PCM
            audio = (audio * 32767).astype(np.int16)
            
            return {
                "success": True,
                "sample_rate": self.sample_rate,
                "duration": duration,
                "audio": audio
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_audio_to_file(self, audio_data, file_path):
        """Save audio data to a WAV file"""
        try:
            write_wav(file_path, audio_data["sample_rate"], audio_data["audio"])
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False
    
    def get_audio_buffer(self, audio_data):
        """Get audio data as a buffer for web playback"""
        try:
            buffer = io.BytesIO()
            write_wav(buffer, audio_data["sample_rate"], audio_data["audio"])
            buffer.seek(0)
            return buffer
        except Exception as e:
            print(f"Error creating audio buffer: {e}")
            return None
    
    def get_audio_base64(self, audio_data):
        """Get audio data as base64 for web playback"""
        try:
            buffer = self.get_audio_buffer(audio_data)
            if buffer:
                audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                return audio_base64
            return None
        except Exception as e:
            print(f"Error creating base64 audio: {e}")
            return None

# Initialize audio generator
audio_generator = AudioGenerator()

# Routes
@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/api/generate-frequency', methods=['POST'])
def generate_frequency():
    """Generate frequency pattern from text"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"success": False, "error": "No text provided"})
    
    # Generate frequency pattern
    result = frequency_generator.generate_frequency(text)
    
    return jsonify(result)

@app.route('/api/generate-audio', methods=['POST'])
def generate_audio():
    """Generate audio from frequency pattern"""
    data = request.get_json()
    text = data.get('text', '')
    duration = float(data.get('duration', 3.0))
    
    if not text:
        return jsonify({"success": False, "error": "No text provided"})
    
    # Generate frequency pattern
    pattern_result = frequency_generator.generate_frequency(text)
    
    if not pattern_result.get("success", False):
        return jsonify(pattern_result)
    
    # Generate audio
    audio_result = audio_generator.generate_audio(pattern_result["pattern"], duration)
    
    if not audio_result.get("success", False):
        return jsonify(audio_result)
    
    # Convert audio to base64 for web playback
    audio_base64 = audio_generator.get_audio_base64(audio_result)
    
    if not audio_base64:
        return jsonify({"success": False, "error": "Failed to convert audio to base64"})
    
    return jsonify({
        "success": True,
        "pattern": pattern_result["pattern"],
        "audio_base64": audio_base64,
        "sample_rate": audio_result["sample_rate"],
        "duration": audio_result["duration"]
    })

@app.route('/api/download-audio', methods=['POST'])
def download_audio():
    """Download generated audio as WAV file"""
    data = request.get_json()
    text = data.get('text', '')
    duration = float(data.get('duration', 3.0))
    
    if not text:
        return jsonify({"success": False, "error": "No text provided"})
    
    # Generate frequency pattern
    pattern_result = frequency_generator.generate_frequency(text)
    
    if not pattern_result.get("success", False):
        return jsonify(pattern_result)
    
    # Generate audio
    audio_result = audio_generator.generate_audio(pattern_result["pattern"], duration)
    
    if not audio_result.get("success", False):
        return jsonify(audio_result)
    
    # Get audio buffer
    buffer = audio_generator.get_audio_buffer(audio_result)
    
    if not buffer:
        return jsonify({"success": False, "error": "Failed to create audio buffer"})
    
    # Create a safe filename from the text
    safe_filename = "".join(c if c.isalnum() else "_" for c in text)
    filename = f"{safe_filename[:30]}.wav"
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='audio/wav'
    )

@app.route('/api/ai-models', methods=['POST'])
def ai_models():
    """Communicate with AI models"""
    data = request.get_json()
    model = data.get('model', 'manus')
    prompt = data.get('prompt', '')
    bypass = data.get('bypass', False)
    
    if not prompt:
        return jsonify({"success": False, "error": "No prompt provided"})
    
    # In a real implementation, this would use the actual model connector
    # For this prototype, we'll simulate it
    response = {
        "success": True,
        "model": model,
        "response": f"This is a simulated response from {model} to: {prompt}",
        "bypass_mode": bypass
    }
    
    return jsonify(response)

@app.route('/api/social-media/search', methods=['POST'])
def social_media_search():
    """Search social media"""
    data = request.get_json()
    platform = data.get('platform', 'twitter')
    query = data.get('query', '')
    
    if not query:
        return jsonify({"success": False, "error": "No query provided"})
    
    # In a real implementation, this would use the social media manager
    # For this prototype, we'll simulate it
    results = []
    for i in range(5):
        results.append({
            "id": f"result_{i}",
            "text": f"Result {i+1} for '{query}' on {platform}",
            "user": f"user_{i}",
            "date": "2025-03-16"
        })
    
    response = {
        "success": True,
        "platform": platform,
        "query": query,
        "results": results
    }
    
    return jsonify(response)

@app.route('/api/news/analyze', methods=['POST'])
def news_analyze():
    """Analyze news and trends"""
    data = request.get_json()
    timeframe = data.get('timeframe', 'week')
    
    # In a real implementation, this would use the trend analyzer
    # For this prototype, we'll simulate it
    trends = []
    for i in range(5):
        trends.append({
            "topic": f"Trend {i+1}",
            "news_volume": i * 100 + 50,
            "social_volume": i * 1000 + 500,
            "sentiment": 0.5 + (i / 10),
            "momentum": 0.6 + (i / 10)
        })
    
    response = {
        "success": True,
        "timeframe": timeframe,
        "trends": trends
    }
    
    return jsonify(response)

@app.route('/api/news/predict', methods=['POST'])
def news_predict():
    """Predict events based on news and trends"""
    data = request.get_json()
    timeframe = data.get('timeframe', 'week')
    
    # In a real implementation, this would use the event predictor
    # For this prototype, we'll simulate it
    predictions = []
    for i in range(3):
        predictions.append({
            "topic": f"Potential Event {i+1}",
            "risk_score": 0.7 + (i / 10),
            "confidence": 0.8 + (i / 20),
            "potential_impact": ["economic", "social"][i % 2],
            "timeframe": ["immediate", "short-term", "long-term"][i % 3],
            "recommended_actions": [
                "Monitor situation closely",
                "Prepare contingency plans",
                "Alert relevant stakeholders"
            ]
        })
    
    response = {
        "success": True,
        "timeframe": timeframe,
        "predictions": predictions
    }
    
    return jsonify(response)

# Run the app
if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
