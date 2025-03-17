"""
Enhanced Frequency Generator Module with Audio Output

This module extends the original frequency generator with audio output capabilities,
converting text to frequencies and generating audio waveforms.
"""

import numpy as np
import math
import random
import hashlib
from typing import Dict, List, Any, Optional, Union
import io
from scipy.io import wavfile

class TextToFrequency:
    """Converts text to frequency patterns"""
    
    def __init__(self):
        self.base_frequency_range = (220.0, 1760.0)  # A3 to A6
        self.harmonic_count_range = (3, 7)
        self.modulation_depth_range = (0.3, 0.8)
        
    def text_to_hash(self, text: str) -> str:
        """Convert text to a hash value"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def hash_to_seed(self, hash_value: str) -> int:
        """Convert hash to a numeric seed"""
        return int(hash_value, 16)
    
    def generate_pattern(self, text: str) -> Dict[str, Any]:
        """Generate a frequency pattern from text"""
        # Get hash and seed
        hash_value = self.text_to_hash(text)
        seed = self.hash_to_seed(hash_value)
        
        # Set random seed for deterministic generation
        random.seed(seed)
        
        # Generate base frequency
        base_freq = self._generate_base_frequency(text)
        
        # Generate harmonics
        harmonics = self._generate_harmonics(base_freq, text)
        
        # Generate modulation
        modulation = self._generate_modulation(base_freq, text)
        
        # Generate envelope
        envelope = self._generate_envelope(text)
        
        return {
            "base": base_freq,
            "harmonics": harmonics,
            "modulation": modulation,
            "envelope": envelope
        }
    
    def _generate_base_frequency(self, text: str) -> float:
        """Generate base frequency from text"""
        # Use text length and character values to influence frequency
        text_length = len(text)
        char_sum = sum(ord(c) for c in text)
        
        # Map to frequency range
        min_freq, max_freq = self.base_frequency_range
        freq_range = max_freq - min_freq
        
        # Use a combination of text properties for base frequency
        normalized_value = (char_sum % 1000) / 1000.0
        base_freq = min_freq + normalized_value * freq_range
        
        # Add some variation based on text length
        variation = (text_length % 10) / 10.0 * 50.0
        base_freq += variation
        
        return base_freq
    
    def _generate_harmonics(self, base_freq: float, text: str) -> List[float]:
        """Generate harmonics based on base frequency and text"""
        # Determine number of harmonics
        min_count, max_count = self.harmonic_count_range
        harmonic_count = min_count + (hash(text) % (max_count - min_count + 1))
        
        # Generate harmonics
        harmonics = [base_freq]  # Include base frequency as first harmonic
        
        for i in range(1, harmonic_count):
            # Use different harmonic ratios based on text characteristics
            if i % 2 == 0:
                # Even harmonics (octaves and fifths)
                ratio = 2.0 if i % 4 == 0 else 1.5
            else:
                # Odd harmonics (thirds and sevenths)
                ratio = 1.25 if i % 3 == 0 else 1.75
            
            # Add some variation
            variation = random.uniform(-0.05, 0.05)
            ratio += variation
            
            # Calculate harmonic frequency
            harmonic_freq = base_freq * ratio * (i + 1)
            
            # Ensure it's within a reasonable range
            if harmonic_freq < 20000:  # Below 20kHz (human hearing limit)
                harmonics.append(harmonic_freq)
        
        return harmonics
    
    def _generate_modulation(self, base_freq: float, text: str) -> Dict[str, float]:
        """Generate frequency modulation parameters"""
        # Determine if modulation should be applied
        should_modulate = hash(text) % 10 > 3  # 60% chance
        
        if not should_modulate:
            return {}
        
        # Calculate modulation frequency (typically much lower than base frequency)
        mod_freq = base_freq / random.uniform(10, 20)
        
        # Calculate modulation depth
        min_depth, max_depth = self.modulation_depth_range
        mod_depth = min_depth + random.random() * (max_depth - min_depth)
        
        return {
            "frequency": mod_freq,
            "depth": mod_depth
        }
    
    def _generate_envelope(self, text: str) -> Dict[str, float]:
        """Generate ADSR envelope parameters"""
        # Use text characteristics to influence envelope
        text_length = len(text)
        
        # More complex text gets more complex envelope
        complexity = min(1.0, text_length / 50.0)
        
        # Generate ADSR parameters
        attack = 0.01 + random.random() * 0.1 * complexity
        decay = 0.1 + random.random() * 0.2 * complexity
        sustain = 0.5 + random.random() * 0.4
        release = 0.1 + random.random() * 0.3 * complexity
        
        return {
            "attack": attack,
            "decay": decay,
            "sustain": sustain,
            "release": release
        }


class AudioGenerator:
    """Generates audio from frequency patterns"""
    
    def __init__(self):
        self.sample_rate = 44100  # Hz
        
    def generate_audio(self, frequency_pattern: Dict[str, Any], duration: float = 3.0) -> Dict[str, Any]:
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
    
    def save_audio_to_file(self, audio_data: Dict[str, Any], file_path: str) -> bool:
        """Save audio data to a WAV file"""
        try:
            wavfile.write(file_path, audio_data["sample_rate"], audio_data["audio"])
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False
    
    def get_audio_buffer(self, audio_data: Dict[str, Any]) -> Optional[io.BytesIO]:
        """Get audio data as a buffer for web playback"""
        try:
            buffer = io.BytesIO()
            wavfile.write(buffer, audio_data["sample_rate"], audio_data["audio"])
            buffer.seek(0)
            return buffer
        except Exception as e:
            print(f"Error creating audio buffer: {e}")
            return None


class FrequencyVisualizer:
    """Visualizes frequency patterns"""
    
    def __init__(self):
        pass
    
    def visualize(self, audio_data: Dict[str, Any], visualization_type: str = "waveform") -> Dict[str, Any]:
        """Generate visualization data for audio"""
        try:
            if visualization_type == "waveform":
                return self._visualize_waveform(audio_data)
            elif visualization_type == "spectrum":
                return self._visualize_spectrum(audio_data)
            elif visualization_type == "spectrogram":
                return self._visualize_spectrogram(audio_data)
            else:
                return {
                    "success": False,
                    "error": f"Unknown visualization type: {visualization_type}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _visualize_waveform(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate waveform visualization data"""
        audio = audio_data["audio"]
        sample_rate = audio_data["sample_rate"]
        
        # Downsample for visualization
        downsample_factor = max(1, len(audio) // 1000)
        waveform = audio[::downsample_factor]
        
        # Normalize to -1 to 1
        waveform = waveform / 32767.0
        
        return {
            "success": True,
            "type": "waveform",
            "data": waveform.tolist(),
            "sample_rate": sample_rate,
            "time_points": [i * downsample_factor / sample_rate for i in range(len(waveform))]
        }
    
    def _visualize_spectrum(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate frequency spectrum visualization data"""
        audio = audio_data["audio"]
        sample_rate = audio_data["sample_rate"]
        
        # Calculate FFT
        n = len(audio)
        fft_data = np.abs(np.fft.rfft(audio / 32767.0))
        
        # Get frequency points
        freq_points = np.fft.rfftfreq(n, 1 / sample_rate)
        
        # Limit to audible range (20Hz - 20kHz)
        mask = (freq_points >= 20) & (freq_points <= 20000)
        freq_points = freq_points[mask]
        fft_data = fft_data[mask]
        
        # Normalize
        fft_data = fft_data / np.max(fft_data)
        
        return {
            "success": True,
            "type": "spectrum",
            "data": fft_data.tolist(),
            "frequency_points": freq_points.tolist()
        }
    
    def _visualize_spectrogram(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate spectrogram visualization data"""
        audio = audio_data["audio"]
        sample_rate = audio_data["sample_rate"]
        
        # Parameters
        nperseg = 256  # Window size
        noverlap = 128  # Overlap
        
        # Calculate spectrogram
        f, t, Sxx = self._stft(audio / 32767.0, sample_rate, nperseg, noverlap)
        
        # Convert to dB scale
        Sxx = 10 * np.log10(Sxx + 1e-10)
        
        # Normalize
        Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx))
        
        return {
            "success": True,
            "type": "spectrogram",
            "data": Sxx.tolist(),
            "time_points": t.tolist(),
            "frequency_points": f.tolist()
        }
    
    def _stft(self, x, fs, nperseg, noverlap):
        """Short-time Fourier transform"""
        # Window function
        window = np.hanning(nperseg)
        
        # Calculate number of segments
        step = nperseg - noverlap
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
        strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
        x_segments = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        
        # Apply window
        x_segments = x_segments * window
        
        # Calculate FFT
        Sxx = np.abs(np.fft.rfft(x_segments, axis=-1))**2
        
        # Frequency points
        f = np.fft.rfftfreq(nperseg, 1/fs)
        
        # Time points
        t = np.arange(Sxx.shape[0]) * step / fs
        
        return f, t, Sxx


class ModelCommunicator:
    """Uses frequencies to communicate with AI models"""
    
    def __init__(self, api_manager=None):
        self.api_manager = api_manager
        
    def communicate(self, model_name: str, prompt: str, frequency_pattern: Dict[str, Any], bypass: bool = False) -> Dict[str, Any]:
        """Communicate with an AI model using frequency patterns"""
        try:
            # In a real implementation, this would use the frequency pattern to encode the prompt
            # and communicate with the model in a special way
            # For this prototype, we'll simulate it
            
            if self.api_manager and hasattr(self.api_manager, model_name):
                # Use the API manager to communicate with the model
                model = getattr(self.api_manager, model_name)
                
                # Prepare request with frequency encoding
                request = {
                    "prompt": prompt,
                    "frequency_encoding": {
                        "base": frequency_pattern.get("base", 0),
                        "harmonics": frequency_pattern.get("harmonics", []),
                        "modulation": frequency_pattern.get("modulation", {})
                    },
                    "bypass_restrictions": bypass
                }
                
                # Call the model
                response = model.generate(request)
                
                return {
                    "success": True,
                    "model": model_name,
                    "response": response.get("text", ""),
                    "bypass_mode": bypass
                }
            else:
                # Simulate a response
                return {
                    "success": True,
                    "model": model_name,
                    "response": f"This is a simulated response from {model_name} to: {prompt}",
                    "bypass_mode": bypass
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class RestrictionBypass:
    """Handles bypassing restrictions in AI models"""
    
    def __init__(self):
        pass
    
    def analyze_restrictions(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Analyze potential restrictions for a prompt"""
        # In a real implementation, this would analyze the prompt for potential restrictions
        # For this prototype, we'll simulate it
        
        # Check for potentially restricted keywords
        restricted_keywords = ["hack", "illegal", "exploit", "bypass", "crack"]
        found_keywords = [kw for kw in restricted_keywords if kw in prompt.lower()]
        
        restriction_level = 0
        if found_keywords:
            restriction_level = len(found_keywords) / len(restricted_keywords)
        
        return {
            "success": True,
            "model": model_name,
            "restriction_level": restriction_level,
            "found_keywords": found_keywords,
            "bypass_recommended": restriction_level > 0
        }
    
    def generate_bypass_pattern(self, model_name: str, prompt: str, base_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a specialized frequency pattern to bypass restrictions"""
        # In a real implementation, this would modify the frequency pattern to bypass restrictions
        # For this prototype, we'll simulate it by modifying the base pattern
        
        # Start with the base pattern
        bypass_pattern = base_pattern.copy()
        
        # Modify the pattern for bypassing
        if "base" in bypass_pattern:
            bypass_pattern["base"] *= 1.1  # Increase base frequency
        
        if "harmonics" in bypass_pattern:
            # Add more harmonics
            max_harmonic = max(bypass_pattern["harmonics"]) if bypass_pattern["harmonics"] else 1000
            bypass_pattern["harmonics"].extend([max_harmonic * 1.5, max_harmonic * 2.0])
        
        if "modulation" in bypass_pattern:
            # Modify modulation
            if "depth" in bypass_pattern["modulation"]:
                bypass_pattern["modulation"]["depth"] = min(0.95, bypass_pattern["modulation"]["depth"] * 1.5)
        else:
            # Add modulation if not present
            bypass_pattern["modulation"] = {
                "frequency": bypass_pattern.get("base", 440.0) / 8,
                "depth": 0.8
            }
        
        return bypass_pattern


class FrequencyGenerator:
    """Main class for the frequency generator module"""
    
    def __init__(self, api_manager=None):
        self.text_to_frequency = TextToFrequency()
        self.audio_generator = AudioGenerator()
        self.visualizer = FrequencyVisualizer()
        self.model_communicator = ModelCommunicator(api_manager)
        self.restriction_bypass = RestrictionBypass()
        
    def generate_frequency(self, text: str) -> Dict[str, Any]:
        """Generate a frequency pattern from text"""
        try:
            pattern = self.text_to_frequency.generate_pattern(text)
            
            return {
                "success": True,
                "pattern": pattern
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_audio(self, text: str, duration: float = 3.0) -> Dict[str, Any]:
        """Generate audio from text"""
        try:
            # Generate frequency pattern
            pattern_result = self.generate_frequency(text)
            
            if not pattern_result.get("success", False):
                return pattern_result
            
            # Generate audio
            audio_result = self.audio_generator.generate_audio(pattern_result["pattern"], duration)
            
            if not audio_result.get("success", False):
                return audio_result
            
            return {
                "success": True,
                "pattern": pattern_result["pattern"],
                "audio": audio_result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def visualize(self, text: str, visualization_type: str = "waveform", duration: float = 3.0) -> Dict[str, Any]:
        """Generate visualization for text frequency"""
        try:
            # Generate audio
            audio_result = self.generate_audio(text, duration)
            
            if not audio_result.get("success", False):
                return audio_result
            
            # Generate visualization
            visualization_result = self.visualizer.visualize(audio_result["audio"], visualization_type)
            
            if not visualization_result.get("success", False):
                return visualization_result
            
            return {
                "success": True,
                "pattern": audio_result["pattern"],
                "visualization": visualization_result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def communicate_with_model(self, model_name: str, prompt: str, bypass: bool = False) -> Dict[str, Any]:
        """Communicate with an AI model using frequency patterns"""
        try:
            # Generate frequency pattern from prompt
            pattern_result = self.generate_frequency(prompt)
            
            if not pattern_result.get("success", False):
                return pattern_result
            
            # If bypass is requested, analyze restrictions and modify pattern
            if bypass:
                restriction_analysis = self.restriction_bypass.analyze_restrictions(model_name, prompt)
                
                if restriction_analysis.get("bypass_recommended", False):
                    pattern = self.restriction_bypass.generate_bypass_pattern(
                        model_name, prompt, pattern_result["pattern"]
                    )
                else:
                    pattern = pattern_result["pattern"]
            else:
                pattern = pattern_result["pattern"]
            
            # Communicate with model
            communication_result = self.model_communicator.communicate(model_name, prompt, pattern, bypass)
            
            if not communication_result.get("success", False):
                return communication_result
            
            return {
                "success": True,
                "pattern": pattern,
                "response": communication_result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_audio_to_file(self, text: str, file_path: str, duration: float = 3.0) -> Dict[str, Any]:
        """Generate audio from text and save to file"""
        try:
            # Generate audio
            audio_result = self.generate_audio(text, duration)
            
            if not audio_result.get("success", False):
                return audio_result
            
            # Save to file
            success = self.audio_generator.save_audio_to_file(audio_result["audio"], file_path)
            
            if not success:
                return {
                    "success": False,
                    "error": "Failed to save audio to file"
                }
            
            return {
                "success": True,
                "pattern": audio_result["pattern"],
                "file_path": file_path
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Example usage
if __name__ == "__main__":
    # Create frequency generator
    generator = FrequencyGenerator()
    
    # Generate a frequency pattern
    text = "Hello, world!"
    pattern_result = generator.generate_frequency(text)
    
    if pattern_result["success"]:
        print("Frequency Pattern:")
        pattern = pattern_result["pattern"]
        print(f"Base Frequency: {pattern['base']:.2f} Hz")
        
        if "harmonics" in pattern:
            print("Harmonics:", ", ".join([f"{h:.2f} Hz" for h in pattern["harmonics"]]))
        
        if "modulation" in pattern:
            mod = pattern["modulation"]
            print(f"Modulation: {mod.get('frequency', 0):.2f} Hz at {mod.get('depth', 0):.2f} depth")
        
        # Generate audio
        audio_result = generator.generate_audio(text)
        
        if audio_result["success"]:
            print("\nAudio generated successfully")
            print(f"Sample Rate: {audio_result['audio']['sample_rate']} Hz")
            print(f"Duration: {audio_result['audio']['duration']:.2f} seconds")
            print(f"Samples: {len(audio_result['audio']['audio'])}")
            
            # Save to file
            generator.save_audio_to_file(text, "hello_world.wav")
            print("\nAudio saved to hello_world.wav")
    else:
        print(f"Error: {pattern_result['error']}")
