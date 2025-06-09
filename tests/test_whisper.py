#!/usr/bin/env python3
"""
Test Whisper model loading and basic transcription functionality.
This helps verify the model is correctly installed and working.
"""

import sys
import time
import numpy as np
import os

def test_whisper_import():
    """Test if Whisper can be imported"""
    print("=== Whisper Import Test ===\n")
    
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper import successful")
        
        # List available models
        print("\nAvailable models:")
        models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"]
        for model in models:
            print(f"  - {model}")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import faster-whisper: {e}")
        return False

def test_mlx_whisper():
    """Test if MLX-optimized Whisper is available"""
    print("\n=== MLX Whisper Test ===\n")
    
    try:
        import mlx_whisper
        print("✅ MLX-optimized Whisper is available (Apple Silicon acceleration)")
        return True
    except ImportError:
        print("ℹ️  MLX Whisper not available (normal - only for Apple Silicon)")
        return True  # Not a failure, just informational

def test_model_loading(model_size="base"):
    """Test loading a Whisper model"""
    print(f"\n=== Model Loading Test ({model_size}) ===\n")
    
    try:
        from faster_whisper import WhisperModel
        
        print(f"Loading {model_size} model... (this may take a moment)")
        start_time = time.time()
        
        model = WhisperModel(model_size, device="auto", compute_type="int8")
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        # Check model info
        print(f"\nModel info:")
        print(f"  - Model size: {model_size}")
        print(f"  - Device: auto")
        print(f"  - Compute type: int8")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def test_transcription():
    """Test basic transcription with synthetic audio"""
    print("\n=== Transcription Test ===\n")
    
    try:
        from faster_whisper import WhisperModel
        
        # Create a short silent audio (1 second at 16kHz)
        print("Creating test audio...")
        audio = np.zeros(16000, dtype=np.float32)
        
        print("Loading base model for test...")
        model = WhisperModel("base", device="auto", compute_type="int8")
        
        print("Transcribing silent audio...")
        start_time = time.time()
        
        segments, info = model.transcribe(
            audio,
            language="en",
            vad_filter=True
        )
        
        segments_list = list(segments)
        text = " ".join([seg.text for seg in segments_list])
        
        transcribe_time = time.time() - start_time
        
        print(f"✅ Transcription completed in {transcribe_time:.2f} seconds")
        print(f"   Result: '{text.strip()}' (expected empty or minimal)")
        print(f"   Detected language: {info.language} with probability {info.language_probability:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_cache():
    """Check if models are cached locally"""
    print("\n=== Model Cache Check ===\n")
    
    cache_dir = os.path.expanduser("~/.cache/whisper")
    
    if os.path.exists(cache_dir):
        print(f"Model cache directory: {cache_dir}")
        files = os.listdir(cache_dir)
        
        if files:
            print("\nCached models:")
            for file in files:
                size = os.path.getsize(os.path.join(cache_dir, file)) / (1024**3)
                print(f"  - {file} ({size:.2f} GB)")
        else:
            print("No models cached yet")
    else:
        print("Model cache directory not found (models will be downloaded on first use)")
    
    return True

def main():
    """Run all Whisper tests"""
    print("🎤 Kuiskaus Whisper Model Test Suite\n")
    
    tests = [
        ("Whisper Import", test_whisper_import),
        ("MLX Whisper Check", test_mlx_whisper),
        ("Model Cache", check_model_cache),
        ("Model Loading", lambda: test_model_loading("base")),
        ("Transcription", test_transcription)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test failed with error: {e}")
            results.append((name, False))
        print("\n" + "-" * 60 + "\n")
    
    # Summary
    print("=== Test Summary ===\n")
    all_passed = True
    critical_passed = True
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        
        # MLX test is not critical
        if not result and name != "MLX Whisper Check":
            critical_passed = False
        if not result:
            all_passed = False
    
    if critical_passed:
        print("\n🎉 All critical tests passed! Whisper is ready to use.")
        
        print("\n💡 Tips:")
        print("- First model download may take several minutes")
        print("- The 'turbo' model is recommended for best speed/accuracy balance")
        print("- Models are cached after first download")
        
        return 0
    else:
        print("\n⚠️  Some critical tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())