import pyaudio
import numpy as np
import threading
import queue
import time
from typing import Optional, Callable

class AudioRecorder:
    def __init__(self, 
                 sample_rate: int = 16000,  # Whisper expects 16kHz
                 chunk_size: int = 1024,
                 channels: int = 1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16
        
        self.pyaudio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread: Optional[threading.Thread] = None
        
        # Find the default input device
        self.input_device_index = self._find_default_input_device()
        
    def _find_default_input_device(self) -> int:
        """Find the default system microphone"""
        try:
            info = self.pyaudio.get_default_input_device_info()
            return info['index']
        except:
            # Fallback to first available input device
            for i in range(self.pyaudio.get_device_count()):
                info = self.pyaudio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    return i
            raise RuntimeError("No input device found")
    
    def _recording_worker(self):
        """Worker thread for continuous audio recording"""
        self.stream = self.pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk_size
        )
        
        while self.recording:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_queue.put(data)
            except Exception as e:
                print(f"Error recording audio: {e}")
                break
        
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
    
    def start_recording(self):
        """Start recording audio"""
        if not self.recording:
            self.recording = True
            self.audio_queue = queue.Queue()  # Clear any old data
            self.recording_thread = threading.Thread(target=self._recording_worker)
            self.recording_thread.start()
            
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the audio data as numpy array"""
        if self.recording:
            self.recording = False
            if self.recording_thread:
                self.recording_thread.join(timeout=1.0)
            
            # Collect all audio data
            audio_chunks = []
            while not self.audio_queue.empty():
                audio_chunks.append(self.audio_queue.get())
            
            if audio_chunks:
                # Convert to numpy array
                audio_data = b''.join(audio_chunks)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                # Convert to float32 and normalize
                audio_float = audio_array.astype(np.float32) / 32768.0
                return audio_float
            
        return np.array([], dtype=np.float32)
    
    def cleanup(self):
        """Clean up PyAudio resources"""
        if self.recording:
            self.stop_recording()
        self.pyaudio.terminate()
        
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()