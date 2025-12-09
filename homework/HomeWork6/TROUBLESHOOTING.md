# 🔧 Troubleshooting Guide

## Quick Diagnosis

Run this command first to check your setup:
```bash
python diagnose_setup.py
```

This will tell you exactly what's wrong.

---

## Common Errors and Solutions

### ❌ Error: "model 'qwen3:8b' not found" (404)

**Problem**: The specified model doesn't exist in your Ollama installation.

**Solutions**:

1. **Check what models you have**:
   ```bash
   ollama list
   ```

2. **Pull a recommended model**:
   ```bash
   # Option 1: Llama 3.2 (recommended)
   ollama pull llama3.2
   
   # Option 2: Qwen 2.5 7B
   ollama pull qwen2.5:7b
   
   # Option 3: Llama 3.1 8B
   ollama pull llama3.1:8b
   ```

3. **Use the fixed version** (auto-detects models):
   ```bash
   python ChatBotDeskApp_Fixed.py
   ```

4. **Or manually specify a model** in the code:
   ```python
   # Change this line:
   self.engine = VoiceChatbotEngine(model_name="qwen3:8b")
   
   # To:
   self.engine = VoiceChatbotEngine(model_name="llama3.2")
   # Or whatever model you have from 'ollama list'
   ```

---

### ❌ Error: "Connection refused" or "Cannot connect to Ollama"

**Problem**: Ollama service is not running.

**Solutions**:

1. **Start Ollama**:
   ```bash
   # On macOS/Linux
   ollama serve
   
   # Or just run a model (which starts the service):
   ollama run llama3.2
   ```

2. **Check if Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```
   
   Should return a list of models in JSON format.

3. **Restart Ollama** if it's frozen:
   ```bash
   # macOS/Linux:
   pkill ollama
   ollama serve
   
   # Windows:
   # Close Ollama from system tray, then restart
   ```

---

### ❌ Error: "ollama: command not found"

**Problem**: Ollama is not installed or not in PATH.

**Solutions**:

1. **Install Ollama**:
   - Visit: https://ollama.com/download
   - Download for your OS
   - Install and restart terminal

2. **Verify installation**:
   ```bash
   ollama --version
   ```

3. **Add to PATH** (if needed):
   ```bash
   # macOS/Linux:
   export PATH=$PATH:/usr/local/bin
   
   # Add to ~/.bashrc or ~/.zshrc for permanent fix
   ```

---

### ❌ Error: "No module named 'whisper'" or other package errors

**Problem**: Required Python packages not installed.

**Solutions**:

1. **Install all dependencies**:
   ```bash
   pip install openai-whisper pyttsx3 sounddevice soundfile sympy arxiv openai numpy
   ```

2. **Use virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install openai-whisper pyttsx3 sounddevice soundfile sympy arxiv openai numpy
   ```

3. **Upgrade pip** if installation fails:
   ```bash
   pip install --upgrade pip
   ```

---

### ❌ Error: "No audio devices found" or microphone issues

**Problem**: Audio drivers or permissions issue.

**Solutions**:

1. **Check microphone permission**:
   - macOS: System Preferences → Security & Privacy → Microphone
   - Windows: Settings → Privacy → Microphone
   - Linux: Check PulseAudio/ALSA settings

2. **Test audio devices**:
   ```python
   import sounddevice as sd
   print(sd.query_devices())
   ```

3. **Install audio drivers**:
   ```bash
   # Ubuntu/Debian:
   sudo apt-get install portaudio19-dev python3-pyaudio
   
   # macOS (via Homebrew):
   brew install portaudio
   
   # Windows: Usually works out of the box
   ```

---

### ❌ Error: TTS not working (pyttsx3 errors)

**Problem**: Text-to-speech engine issues.

**Solutions**:

1. **Reinstall pyttsx3**:
   ```bash
   pip uninstall pyttsx3
   pip install pyttsx3
   ```

2. **Windows specific**:
   ```bash
   pip install pywin32
   ```

3. **macOS/Linux**:
   ```bash
   # Install espeak
   # Ubuntu/Debian:
   sudo apt-get install espeak
   
   # macOS:
   brew install espeak
   ```

4. **Fallback**: The app will work without TTS, just no voice output.

---

### ❌ Error: LLM not calling functions correctly

**Problem**: Model doesn't follow function calling instructions.

**Solutions**:

1. **Try a different model**:
   ```bash
   # Some models follow instructions better
   ollama pull llama3.2
   ollama pull mistral
   ```

2. **Lower temperature** in code:
   ```python
   # Change from:
   temperature=0.7
   
   # To:
   temperature=0.3  # More deterministic
   ```

3. **Adjust prompt** - Make it more explicit:
   ```python
   # Add to system prompt:
   "YOU MUST output ONLY the JSON, nothing else, when calling a function."
   ```

4. **Test with examples**:
   ```bash
   python test_function_calling.py
   ```

---

### ❌ Error: Whisper model loading takes forever

**Problem**: Whisper downloading large model files.

**Solutions**:

1. **Use smaller model**:
   ```python
   # Change from:
   self.whisper_model = whisper.load_model("base")
   
   # To:
   self.whisper_model = whisper.load_model("tiny")  # Faster, less accurate
   ```

2. **Pre-download models**:
   ```python
   import whisper
   whisper.load_model("base")  # Downloads once, caches
   ```

3. **Check disk space** - Models need ~1-5GB

---

### ❌ Error: JSON parsing errors

**Problem**: LLM output not valid JSON.

**Solutions**:

1. **Already handled** in the code - it has fallback parsing

2. **But if persistent**, adjust regex patterns:
   ```python
   # In route_llm_output(), try different patterns
   ```

3. **Check LLM output** in console logs to debug

---

## Platform-Specific Issues

### macOS

**Issue**: "Operation not permitted" errors
```bash
# Grant Terminal microphone access:
# System Preferences → Security & Privacy → Microphone → Enable Terminal
```

**Issue**: Homebrew packages not found
```bash
brew update
brew upgrade
```

### Windows

**Issue**: PyAudio installation fails
```bash
# Use prebuilt wheel:
pip install pipwin
pipwin install pyaudio
```

**Issue**: Path issues with Ollama
```powershell
# Add to PATH manually:
# Right-click This PC → Properties → Advanced → Environment Variables
# Add: C:\Users\YourName\AppData\Local\Programs\Ollama
```

### Linux

**Issue**: Permission denied for audio devices
```bash
sudo usermod -a -G audio $USER
# Logout and login again
```

**Issue**: Libraries not found
```bash
sudo apt-get install python3-dev portaudio19-dev ffmpeg
```

---

## Verification Steps

After fixing issues, verify:

1. ✅ **Ollama is running**:
   ```bash
   ollama list  # Should show models
   ```

2. ✅ **Python packages installed**:
   ```bash
   python diagnose_setup.py  # Should show all green
   ```

3. ✅ **Model accessible**:
   ```bash
   ollama run llama3.2  # Should start chat
   # Type /bye to exit
   ```

4. ✅ **Test script works**:
   ```bash
   python test_function_calling.py  # Should run tests
   ```

5. ✅ **Main app starts**:
   ```bash
   python ChatBotDeskApp_Fixed.py  # Should open GUI
   ```

---

## Getting Help

If you're still stuck:

1. **Run diagnostics**:
   ```bash
   python diagnose_setup.py > diagnostics.txt
   ```

2. **Check logs**: Look for error messages in console output

3. **Try minimal version**: Use `test_function_calling.py` to isolate the issue

4. **Common gotchas**:
   - Ollama must be running BEFORE starting the app
   - Model name in code must match `ollama list` output
   - Python 3.8+ required
   - Some models work better than others for function calling

---

## Model Recommendations

### Best for Function Calling:
1. **llama3.2** (7B) - ✅ Recommended
2. **llama3.1:8b** - ✅ Good
3. **qwen2.5:7b** - ✅ Good
4. **mistral** - ✅ Decent

### Avoid for Function Calling:
- Very small models (<3B params) - often don't follow instructions well
- Base models (non-instruct versions) - not instruction-tuned

---

## Quick Fix Checklist

Run through this list:

- [ ] Ollama installed: `ollama --version`
- [ ] Ollama running: `ollama list`
- [ ] Model downloaded: `ollama pull llama3.2`
- [ ] Python 3.8+: `python --version`
- [ ] Packages installed: `pip list | grep -E "whisper|pyttsx3|sounddevice"`
- [ ] Test script works: `python test_function_calling.py`
- [ ] Use fixed version: `python ChatBotDeskApp_Fixed.py`

---

## Emergency Fallback

If nothing works, try this minimal setup:

```bash
# 1. Fresh start
pip install openai ollama-python

# 2. Minimal test
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
response = client.chat.completions.create(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Hi'}]
)
print(response.choices[0].message.content)
"
```

If this works, the issue is with additional packages (audio, etc.)

---

## Still Having Issues?

**Most common cause**: Wrong model name

**Quick fix**: Use `ChatBotDeskApp_Fixed.py` which auto-detects models!

---

Last updated: 2024
