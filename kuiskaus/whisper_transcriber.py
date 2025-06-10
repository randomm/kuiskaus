import numpy as np
import time
from typing import Optional, Dict, Any
import threading
import mlx_whisper

class WhisperTranscriber:
    def __init__(self, model_name: str = "turbo", device: Optional[str] = None):
        """
        Initialize Whisper transcriber for Apple Silicon
        
        Args:
            model_name: Model size - for V3 Turbo use "turbo"
            device: Ignored (kept for API compatibility)
        """
        self.model_name = model_name
        self.model = None
        self.model_lock = threading.Lock()
        
        # Map model names to MLX model paths
        self.model_paths = {
            "turbo": "mlx-community/whisper-large-v3-turbo",
            "base": "mlx-community/whisper-base",
            "small": "mlx-community/whisper-small",
            "medium": "mlx-community/whisper-medium",
            "large": "mlx-community/whisper-large-v3"
        }
        
        # Load model in background
        self.load_thread = threading.Thread(target=self._load_model)
        self.load_thread.start()
        
    def _load_model(self):
        """Load the MLX-optimized Whisper model"""
        print(f"Loading Whisper model: {self.model_name}")
        print("Using MLX-optimized Whisper for Apple Silicon")
        start_time = time.time()
        
        with self.model_lock:
            model_path = self.model_paths.get(self.model_name, f"mlx-community/whisper-{self.model_name}")
            self.model = mlx_whisper.load_models.load_model(model_path)
        
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
        Transcribe audio to text using MLX Whisper
        
        Args:
            audio: Audio data as numpy array (float32, mono, 16kHz)
            language: Language code (e.g., 'en' for English) or None for auto-detect
            task: "transcribe" or "translate" (to English)
            **kwargs: Additional arguments for MLX whisper
            
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
            
            # Get model path for transcription
            model_path = self.model_paths.get(self.model_name, f"mlx-community/whisper-{self.model_name}")
            
            # MLX whisper parameters
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
            
            transcribe_time = time.time() - start_time
            
            # Add timing information
            result["transcribe_time"] = transcribe_time
            result["audio_duration"] = len(audio) / 16000.0
            result["rtf"] = transcribe_time / result["audio_duration"]  # Real-time factor
            
            return result
    
    def cleanup(self):
        """Clean up resources"""
        # MLX models don't need explicit cleanup
        if self.model:
            del self.model
            self.model = None