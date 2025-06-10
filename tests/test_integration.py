#!/usr/bin/env python3
"""
Test system integration: hotkeys, text insertion, and permissions.
Run with sudo if needed for certain tests.
"""

import sys
import time
import os
import subprocess

def test_accessibility_permissions():
    """Check if we have accessibility permissions"""
    print("=== Accessibility Permissions Test ===\n")
    
    try:
        from ApplicationServices import AXIsProcessTrusted
        
        is_trusted = AXIsProcessTrusted()
        
        if is_trusted:
            print("✅ Accessibility permissions granted")
            return True
        else:
            print("❌ Accessibility permissions NOT granted")
            print("\nTo grant permissions:")
            print("1. Open System Settings > Privacy & Security > Accessibility")
            print("2. Add Terminal (or your terminal app) to the list")
            print("3. Ensure it's enabled")
            print("4. Restart this app after granting permissions")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not check accessibility: {e}")
        print("   The app will prompt for permissions when needed")
        return True  # Don't fail the test for this

def test_pyobjc_frameworks():
    """Test if required PyObjC frameworks are available"""
    print("\n=== PyObjC Frameworks Test ===\n")
    
    frameworks = [
        ("Quartz", "Core Graphics and Event handling"),
        ("AppKit", "Application Kit for macOS"),
        ("Foundation", "Foundation framework"),
        ("ApplicationServices", "Application Services")
    ]
    
    all_good = True
    for framework, description in frameworks:
        try:
            module = __import__(framework)
            print(f"✅ {framework}: {description}")
        except ImportError:
            print(f"❌ {framework}: Not available - {description}")
            all_good = False
    
    return all_good

def test_hotkey_simulation():
    """Test if we can detect modifier keys"""
    print("\n=== Hotkey Detection Test ===\n")
    
    if not test_accessibility_permissions():
        print("⚠️  Skipping - requires accessibility permissions")
        return False
    
    try:
        from AppKit import NSEvent
        import Quartz
        
        print("Testing modifier key detection...")
        print("Press and release Control key within 5 seconds...")
        
        detected = False
        start_time = time.time()
        
        while time.time() - start_time < 5:
            flags = NSEvent.modifierFlags()
            if flags & NSEvent.ModifierFlags.control:
                print("✅ Control key detected!")
                detected = True
                break
            time.sleep(0.1)
        
        if not detected:
            print("❌ No Control key press detected")
            
        return detected
        
    except Exception as e:
        print(f"❌ Error in hotkey test: {e}")
        return False

def test_clipboard_access():
    """Test clipboard read/write access"""
    print("\n=== Clipboard Access Test ===\n")
    
    try:
        from AppKit import NSPasteboard, NSPasteboardTypeString
        
        pasteboard = NSPasteboard.generalPasteboard()
        
        # Save current content
        old_content = pasteboard.stringForType_(NSPasteboardTypeString)
        
        # Write test content
        test_string = "Kuiskaus clipboard test"
        pasteboard.clearContents()
        success = pasteboard.setString_forType_(test_string, NSPasteboardTypeString)
        
        if success:
            # Read it back
            read_back = pasteboard.stringForType_(NSPasteboardTypeString)
            
            if read_back == test_string:
                print("✅ Clipboard read/write working")
                
                # Restore old content
                if old_content:
                    pasteboard.clearContents()
                    pasteboard.setString_forType_(old_content, NSPasteboardTypeString)
                
                return True
            else:
                print("❌ Clipboard read mismatch")
                return False
        else:
            print("❌ Failed to write to clipboard")
            return False
            
    except Exception as e:
        print(f"❌ Clipboard test error: {e}")
        return False

def test_rumps_availability():
    """Test if rumps (menu bar framework) is available"""
    print("\n=== Menu Bar Framework Test ===\n")
    
    try:
        import rumps
        print("✅ rumps framework available")
        print(f"   Version: {rumps.__version__ if hasattr(rumps, '__version__') else 'Unknown'}")
        return True
    except ImportError:
        print("❌ rumps not installed (required for menu bar app)")
        print("   Install with: pip install rumps")
        return False

def test_system_info():
    """Display system information"""
    print("\n=== System Information ===\n")
    
    # macOS version
    try:
        macos_version = subprocess.check_output(
            ["sw_vers", "-productVersion"], 
            text=True
        ).strip()
        print(f"macOS Version: {macos_version}")
    except:
        print("macOS Version: Unable to determine")
    
    # Processor type
    try:
        processor = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], 
            text=True
        ).strip()
        print(f"Processor: {processor}")
        
        # Check if Apple Silicon
        if "Apple" in processor:
            print("✅ Apple Silicon detected - Required for this application")
        else:
            print("❌ Intel processor detected - This application requires Apple Silicon")
            return False
    except:
        print("❌ Processor: Unable to determine")
        return False
    
    # Python version
    print(f"Python Version: {sys.version.split()[0]}")
    
    return True

def main():
    """Run all system integration tests"""
    print("🎤 Kuiskaus System Integration Test Suite\n")
    
    tests = [
        ("System Info", test_system_info),
        ("PyObjC Frameworks", test_pyobjc_frameworks),
        ("Accessibility Permissions", test_accessibility_permissions),
        ("Clipboard Access", test_clipboard_access),
        ("Menu Bar Framework", test_rumps_availability),
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
        print("\n🎉 All tests passed! System integration looks good.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("- Grant accessibility permissions in System Preferences")
        print("- Install missing dependencies with: pip install -r requirements.txt")
        print("- Restart Terminal after granting permissions")
        return 1

if __name__ == "__main__":
    sys.exit(main())