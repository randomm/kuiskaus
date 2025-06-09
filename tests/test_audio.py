#!/usr/bin/env python3
"""
Simple audio recording test to verify microphone access and PyAudio functionality.
Run this to troubleshoot audio recording issues.
"""

import pyaudio
import numpy as np
import time
import sys

def test_audio_devices():
    """List all available audio devices"""
    print("=== Audio Devices Test ===\n")
    
    try:
        p = pyaudio.PyAudio()
        print(f"PyAudio version: {pyaudio.__version__}")
        print(f"Number of devices: {p.get_device_count()}\n")
        
        default_input = None
        input_devices = []
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_devices.append(info)
                marker = ""
                if i == p.get_default_input_device_info()['index']:
                    default_input = info
                    marker = " (DEFAULT)"
                print(f"Input Device {i}: {info['name']}{marker}")
                print(f"  - Channels: {info['maxInputChannels']}")
                print(f"  - Sample Rate: {info['defaultSampleRate']}")
                print()
        
        p.terminate()
        
        if not input_devices:
            print("❌ No input devices found!")
            return False
            
        print(f"✅ Found {len(input_devices)} input device(s)")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing PyAudio: {e}")
        return False

def test_recording(duration=3):
    """Test recording audio for a specified duration"""
    print(f"\n=== Recording Test ({duration} seconds) ===\n")
    
    try:
        p = pyaudio.PyAudio()
        
        # Use default input device
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        print("🎤 Recording... (speak into your microphone)")
        frames = []
        
        for i in range(0, int(16000 / 1024 * duration)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
            
            # Show volume level
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            bar = '=' * int(volume / 100)
            print(f"\rVolume: [{bar:<50}] {volume:5.0f}", end='', flush=True)
        
        print("\n✅ Recording complete!")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Check if we got any audio
        all_data = b''.join(frames)
        audio_array = np.frombuffer(all_data, dtype=np.int16)
        max_volume = np.abs(audio_array).max()
        
        if max_volume < 100:
            print("⚠️  Warning: Very low audio levels detected. Check your microphone.")
            return False
        else:
            print(f"✅ Audio levels OK (max: {max_volume})")
            return True
            
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        return False

def test_permissions():
    """Check if we have microphone permissions"""
    print("=== Permissions Test ===\n")
    
    # On macOS, PyAudio will trigger permission request
    print("If prompted, please grant microphone access to Terminal/Python")
    print("You may need to go to System Preferences > Security & Privacy > Microphone")
    print()
    
    return True

def main():
    """Run all tests"""
    print("🎤 Kuiskaus Audio Test Suite\n")
    
    tests = [
        ("Permissions", test_permissions),
        ("Audio Devices", test_audio_devices),
        ("Recording", test_recording)
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
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Audio recording should work correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())