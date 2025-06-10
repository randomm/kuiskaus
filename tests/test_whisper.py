#!/usr/bin/env python3
"""
Test MLX Whisper Turbo model on Apple Silicon.
Simple, focused test for the core functionality.
"""

import sys
import time
import numpy as np
import os
import platform
import subprocess

def check_apple_silicon():
    """Verify we're running on Apple Silicon"""
    print("=== System Check ===\n")
    
    if platform.system() != "Darwin":
        print("❌ Not running on macOS")
        return False
        
    result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                          capture_output=True, text=True)
    cpu_info = result.stdout.strip()
    
    if "Apple" in cpu_info:
        print(f"✅ Running on Apple Silicon: {cpu_info}")
        return True
    else:
        print(f"❌ Not running on Apple Silicon: {cpu_info}")
        return False

def test_mlx_whisper():
    """Test MLX Whisper with Turbo model"""
    print("\n=== MLX Whisper Test ===\n")
    
    try:
        import mlx_whisper
        print("✅ MLX Whisper imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import mlx_whisper: {e}")
        return False
    
    # Create test audio (5 seconds)
    print("\nCreating 5-second test audio...")
    audio = np.random.randn(5 * 16000).astype(np.float32) * 0.01
    
    print("Testing with Turbo model...")
    start_time = time.time()
    
    try:
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            language="en",
            verbose=False
        )
        
        elapsed = time.time() - start_time
        text = result.get("text", "").strip()
        
        print(f"✅ Transcription completed in {elapsed:.2f}s")
        print(f"   Speed: {5.0/elapsed:.1f}x real-time")
        
        if elapsed < 1.0:
            print("   🚀 Excellent performance!")
        elif elapsed < 2.5:
            print("   ✅ Good performance")
        else:
            print("   ⚠️  Performance slower than expected")
            
        return True
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return False

def check_model_cache():
    """Check if Turbo model is cached"""
    print("\n=== Model Cache ===\n")
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    turbo_model = "models--mlx-community--whisper-large-v3-turbo"
    
    if os.path.exists(os.path.join(cache_dir, turbo_model)):
        print("✅ Turbo model is cached and ready")
        return True
    else:
        print("ℹ️  Turbo model will be downloaded on first use")
        print("   This may take 5-10 minutes (~1.5GB)")
        return True

def main():
    """Run minimal test suite"""
    print("🎤 Kuiskaus Test Suite\n")
    
    # Check Apple Silicon
    if not check_apple_silicon():
        print("\n❌ This application requires Apple Silicon")
        return 1
    
    # Check model cache
    check_model_cache()
    
    # Test MLX Whisper
    if test_mlx_whisper():
        print("\n🎉 All tests passed! Ready to use.")
        return 0
    else:
        print("\n❌ Test failed. Please check the error above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())