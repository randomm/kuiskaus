import numpy as np
import os
import time
from typing import Optional, Dict, Any, List
import threading

# Try to import MLX-optimized version first, fall back to faster-whisper
try:
    import mlx_whisper
    USE_MLX = True
    USE_FASTER_WHISPER = False
except ImportError:
    try:
        from faster_whisper import WhisperModel
        USE_MLX = False
        USE_FASTER_WHISPER = True
    except ImportError:
        # Fallback to original whisper if available
        import whisper
        USE_MLX = False
        USE_FASTER_WHISPER = False

class WhisperTranscriber:
    def __init__(self, model_name: str = "turbo", device: Optional[str] = None):
        """
        Initialize Whisper transcriber
        
        Args:
            model_name: Model size - for V3 Turbo use "turbo"
            device: Device to use (None for auto-detect)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.model_lock = threading.Lock()
        
        # Load model in background
        self.load_thread = threading.Thread(target=self._load_model)
        self.load_thread.start()
        
    def _load_model(self):
        """Load the Whisper model (MLX-optimized or faster-whisper if available)"""
        print(f"Loading Whisper model: {self.model_name}")
        start_time = time.time()
        
        with self.model_lock:
            if USE_MLX:
                print("Using MLX-optimized Whisper for Apple Silicon")
                # MLX version needs the model path/name mapping
                if self.model_name == "turbo":
                    model_path = "mlx-community/whisper-large-v3-turbo"
                else:
                    model_path = f"mlx-community/whisper-{self.model_name}"
                self.model = mlx_whisper.load_models.load_model(model_path)
            elif USE_FASTER_WHISPER:
                print("Using faster-whisper (optimized implementation)")
                # Map model names for faster-whisper
                if self.model_name == "turbo":
                    actual_model_name = "large-v3-turbo"  # Use the actual turbo model
                else:
                    actual_model_name = self.model_name
                
                # Determine compute type based on device
                # Use int8 for better compatibility
                compute_type = "int8"
                
                self.model = WhisperModel(
                    actual_model_name, 
                    device=self.device or "auto",
                    compute_type=compute_type
                )
            else:
                print("Using standard Whisper implementation")
                # Map "turbo" to the correct model name for standard whisper
                if self.model_name == "turbo":
                    actual_model_name = "large-v3"  # V3 Turbo is based on large-v3
                else:
                    actual_model_name = self.model_name
                    
                self.model = whisper.load_model(actual_model_name, device=self.device)
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
    
    def ensure_model_loaded(self):
        """Ensure the model is loaded before transcription"""
        if self.load_thread.is_alive():
            print("Waiting for model to load...")
            self.load_thread.join()
    
    def transcribe(self, 
                   audio: np.ndarray, 
                   language: Optional[str] = None,
                   task: str = "transcribe",
                   **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio to text
        
        Args:
            audio: Audio data as numpy array (float32, mono, 16kHz)
            language: Language code (e.g., 'en' for English) or None for auto-detect
            task: "transcribe" or "translate" (to English)
            **kwargs: Additional arguments for whisper
            
        Returns:
            Dictionary with transcription results
        """
        self.ensure_model_loaded()
        
        if len(audio) == 0:
            return {"text": "", "segments": [], "language": "en"}
        
        # Ensure audio is the right format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Pad audio if too short (Whisper expects at least 0.1 seconds)
        min_length = int(0.1 * 16000)  # 0.1 seconds at 16kHz
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
        
        with self.model_lock:
            start_time = time.time()
            
            if USE_MLX:
                # MLX version has limited parameter support
                # Map model names to proper MLX model paths
                if self.model_name == "turbo":
                    model_path = "mlx-community/whisper-large-v3-turbo"
                else:
                    model_path = f"mlx-community/whisper-{self.model_name}"
                
                # MLX only supports these parameters
                mlx_kwargs = {
                    "verbose": kwargs.get("verbose", False),
                    "temperature": kwargs.get("temperature", 0.0),
                    "compression_ratio_threshold": kwargs.get("compression_ratio_threshold", 2.4),
                    "logprob_threshold": kwargs.get("logprob_threshold", -1.0),
                    "no_speech_threshold": kwargs.get("no_speech_threshold", 0.6),
                }
                
                result = mlx_whisper.transcribe(
                    audio, 
                    path_or_hf_repo=model_path,
                    language=language,
                    task=task,
                    **mlx_kwargs
                )
            elif USE_FASTER_WHISPER:
                # Set default parameters for faster transcription (non-MLX)
                default_kwargs = {
                    "beam_size": 1,  # Faster with beam_size=1
                    "best_of": 1,    # No need for multiple candidates
                    "temperature": 0.0,  # Deterministic
                    "condition_on_previous_text": False,  # Faster without conditioning
                    "fp16": True,
                    "verbose": False
                }
                default_kwargs.update(kwargs)
                # faster-whisper returns segments, we need to format like whisper
                segments, info = self.model.transcribe(
                    audio,
                    language=language,
                    task=task,
                    beam_size=default_kwargs.get("beam_size", 1),
                    temperature=default_kwargs.get("temperature", 0.0),
                    vad_filter=True,  # Enable VAD for better performance
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Convert segments to whisper-like format
                segments_list = list(segments)
                text = " ".join([seg.text for seg in segments_list])
                
                result = {
                    "text": text,
                    "segments": [
                        {
                            "text": seg.text,
                            "start": seg.start,
                            "end": seg.end
                        } for seg in segments_list
                    ],
                    "language": info.language
                }
            else:
                # Standard whisper
                default_kwargs = {
                    "beam_size": 1,  # Faster with beam_size=1
                    "best_of": 1,    # No need for multiple candidates
                    "temperature": 0.0,  # Deterministic
                    "condition_on_previous_text": False,  # Faster without conditioning
                    "fp16": True,
                    "verbose": False
                }
                default_kwargs.update(kwargs)
                result = self.model.transcribe(
                    audio,
                    language=language,
                    task=task,
                    **default_kwargs
                )
            
            transcribe_time = time.time() - start_time
            
            # Add timing information
            result["transcribe_time"] = transcribe_time
            result["audio_duration"] = len(audio) / 16000.0
            result["rtf"] = transcribe_time / result["audio_duration"]  # Real-time factor
            
            return result
    
    def cleanup(self):
        """Clean up resources"""
        # MLX models don't need explicit cleanup
        if not USE_MLX and self.model:
            # Clear model from memory if using standard whisper
            del self.model
            self.model = None