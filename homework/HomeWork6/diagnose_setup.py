#!/usr/bin/env python3
"""
Diagnostic Script for Voice Chatbot Setup
Checks if all requirements are met and helps troubleshoot issues
"""

import sys
import subprocess
import importlib
from typing import Tuple, List

def check_python_version() -> Tuple[bool, str]:
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor} (need 3.8+)"

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, f"✅ {package_name} ({version})"
    except ImportError:
        return False, f"❌ {package_name} not installed"

def check_ollama() -> Tuple[bool, str, List[str]]:
    """Check if Ollama is running and list models"""
    try:
        # Check if ollama command exists
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        
        if result.returncode == 0:
            models = []
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            
            if models:
                return True, f"✅ Ollama running ({len(models)} models)", models
            else:
                return False, "⚠️  Ollama running but no models installed", []
        else:
            return False, "❌ Ollama not responding", []
            
    except FileNotFoundError:
        return False, "❌ Ollama not installed", []
    except subprocess.TimeoutExpired:
        return False, "❌ Ollama timeout (may not be running)", []
    except Exception as e:
        return False, f"❌ Ollama error: {str(e)}", []

def check_audio_devices():
    """Check audio input/output devices"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        
        has_input = len(input_devices) > 0
        has_output = len(output_devices) > 0
        
        status = []
        if has_input:
            status.append(f"✅ {len(input_devices)} input device(s)")
        else:
            status.append("❌ No input devices found")
            
        if has_output:
            status.append(f"✅ {len(output_devices)} output device(s)")
        else:
            status.append("❌ No output devices found")
        
        return has_input and has_output, " | ".join(status)
        
    except Exception as e:
        return False, f"❌ Audio check failed: {str(e)}"

def main():
    """Run all diagnostics"""
    print("=" * 70)
    print("🔍 VOICE CHATBOT DIAGNOSTIC TOOL")
    print("=" * 70)
    print()
    
    # Python version
    print("📌 Python Version")
    print("-" * 70)
    success, msg = check_python_version()
    print(f"   {msg}")
    print()
    
    # Required packages
    print("📦 Python Packages")
    print("-" * 70)
    packages = [
        ('openai-whisper', 'whisper'),
        ('pyttsx3', 'pyttsx3'),
        ('sounddevice', 'sounddevice'),
        ('soundfile', 'soundfile'),
        ('sympy', 'sympy'),
        ('arxiv', 'arxiv'),
        ('openai', 'openai'),
        ('numpy', 'numpy'),
    ]
    
    all_packages_ok = True
    missing_packages = []
    
    for package_name, import_name in packages:
        success, msg = check_package(package_name, import_name)
        print(f"   {msg}")
        if not success:
            all_packages_ok = False
            missing_packages.append(package_name)
    print()
    
    # Ollama
    print("🤖 Ollama LLM")
    print("-" * 70)
    ollama_ok, ollama_msg, models = check_ollama()
    print(f"   {ollama_msg}")
    
    if models:
        print(f"\n   Available models:")
        for model in models:
            print(f"      • {model}")
    print()
    
    # Audio devices
    print("🎤 Audio Devices")
    print("-" * 70)
    audio_ok, audio_msg = check_audio_devices()
    print(f"   {audio_msg}")
    print()
    
    # Summary
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    issues = []
    
    if not all_packages_ok:
        issues.append("Missing Python packages")
        print("\n⚠️  MISSING PACKAGES")
        print("   Install with:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    if not ollama_ok:
        issues.append("Ollama not ready")
        print("\n⚠️  OLLAMA NOT READY")
        print("   Install Ollama:")
        print("   1. Download from https://ollama.com/download")
        print("   2. Install and start Ollama")
        print("   3. Pull a model: ollama pull llama3.2")
        print("      or: ollama pull qwen2.5:7b")
    
    if not audio_ok:
        issues.append("Audio devices not configured")
        print("\n⚠️  AUDIO ISSUES")
        print("   Check your microphone and speakers are connected")
    
    if not issues:
        print("\n✅ ALL CHECKS PASSED!")
        print("   You're ready to run the chatbot!")
        
        if models:
            print(f"\n💡 RECOMMENDED: Use this model in your code:")
            print(f"   model_name = '{models[0]}'")
    else:
        print(f"\n❌ {len(issues)} ISSUE(S) FOUND")
        print("   Fix the issues above before running the chatbot")
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
