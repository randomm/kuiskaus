#!/usr/bin/env python3
"""
Basic sanity check for Kuiskaus installation.
Quick test to ensure core components work.
"""

import sys
import subprocess

def check_system():
    """Check if running on Apple Silicon Mac"""
    try:
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                              capture_output=True, text=True)
        if "Apple" not in result.stdout:
            print("❌ This app requires Apple Silicon")
            return False
        print("✅ Apple Silicon detected")
        return True
    except:
        return False

def check_imports():
    """Check if core modules can be imported"""
    modules = [
        ("pyaudio", "Audio recording"),
        ("numpy", "Audio processing"),
        ("mlx_whisper", "MLX Whisper"),
        ("rumps", "Menu bar support"),
        ("AppKit", "macOS integration")
    ]
    
    all_good = True
    for module, desc in modules:
        try:
            __import__(module)
            print(f"✅ {desc} ({module})")
        except ImportError:
            print(f"❌ {desc} ({module}) - run: pip install -r requirements.txt")
            all_good = False
    
    return all_good

def main():
    print("🎤 Kuiskaus Quick Check\n")
    
    if not check_system():
        return 1
        
    print("\nChecking dependencies:")
    if not check_imports():
        return 1
    
    print("\n✅ Basic checks passed! Run ./run_tests.sh for full tests.")
    return 0

if __name__ == "__main__":
    sys.exit(main())