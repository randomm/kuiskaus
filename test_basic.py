#!/usr/bin/env python3
"""
Basic test to verify core functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kuiskaus.audio_recorder import AudioRecorder
from kuiskaus.whisper_transcriber import WhisperTranscriber
import time

def main():
    print("Testing basic functionality...")
    
    # Test audio recorder
    print("\n1. Testing audio recorder...")
    recorder = AudioRecorder()
    recorder.start_recording()
    time.sleep(1)
    audio = recorder.stop_recording()
    print(f"✅ Recorded {len(audio)/16000:.1f}s of audio")
    
    # Test transcriber
    print("\n2. Testing Whisper transcriber...")
    transcriber = WhisperTranscriber(model_name="turbo")
    print("Waiting for model to load...")
    transcriber.ensure_model_loaded()
    print("✅ Model loaded")
    
    # Test transcription
    print("\n3. Testing transcription...")
    result = transcriber.transcribe(audio)
    print(f"✅ Transcription complete: '{result.get('text', '').strip()}'")
    
    # Cleanup
    recorder.cleanup()
    transcriber.cleanup()
    
    print("\n✅ All basic tests passed!")

if __name__ == "__main__":
    main()