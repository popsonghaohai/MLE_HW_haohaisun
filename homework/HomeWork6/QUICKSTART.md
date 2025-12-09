# 🚀 Quick Start Guide - Function Calling Voice Agent

Get your enhanced voice chatbot with function calling up and running in 5 minutes!

## ⚡ Quick Setup (3 Steps)

### Step 1: Install Dependencies
```bash
# Install all required packages
pip install openai-whisper pyttsx3 sounddevice soundfile sympy arxiv openai
```

### Step 2: Start Ollama
```bash
# Download and start the model
ollama pull qwen3:8b
ollama run qwen3:8b
```

### Step 3: Run the Application
```bash
# Run the enhanced chatbot
python ChatBotDeskApp_Enhanced.py
```

That's it! 🎉

---

## 🧪 Quick Test (Without GUI)

Want to test the function calling logic first?

```bash
# Run the test script
python test_function_calling.py

# Choose option 1 for automated tests
# or option 2 for interactive mode
```

This will show you how the function routing works without needing the full voice interface.

---

## 📝 Try These Example Queries

### Mathematical Calculations
```
You: "What's 15 times 23?"
Bot: [Calls calculate] "The result is: 345"

You: "Calculate sqrt(144) + 10"
Bot: [Calls calculate] "The result is: 22"

You: "What's (5 + 3) * 2?"
Bot: [Calls calculate] "The result is: 16"
```

### Academic Paper Search
```
You: "Search for papers about quantum computing"
Bot: [Calls search_arxiv] Returns paper title, authors, abstract, link

You: "Find research on neural networks"
Bot: [Calls search_arxiv] Returns relevant paper from arXiv

You: "Show me papers about transformers in NLP"
Bot: [Calls search_arxiv] Returns matching papers
```

### Normal Conversation (Still Works!)
```
You: "Hello, how are you?"
Bot: "Hello! I'm doing great, thank you for asking. How can I help you today?"

You: "What can you do?"
Bot: "I can help you with general questions, search for academic papers on arXiv, 
     and perform mathematical calculations. Just ask me anything!"
```

---

## 🎯 Using the Application

### Text Input Mode
1. Type your message in the text box at the bottom
2. Press **Enter** or click **📤 Send**
3. Watch the assistant process and respond
4. Functions are called automatically when needed

### Voice Input Mode
1. Click **🎙️ Start Recording**
2. Speak your query clearly
3. Click **🛑 Stop Recording**
4. Click **⚙️ Process Voice**
5. Listen to the response

---

## 🔍 What Happens Behind the Scenes

### When You Ask: "What's 25 times 4?"

```
1. Your Input → LLM
   ↓
2. LLM generates: {"function": "calculate", "arguments": {"expression": "25*4"}}
   ↓
3. System detects JSON function call
   ↓
4. Calls calculate("25*4")
   ↓
5. calculate() returns: "The result is: 100"
   ↓
6. Result displayed/spoken to you
```

### When You Ask: "Find quantum computing papers"

```
1. Your Input → LLM
   ↓
2. LLM generates: {"function": "search_arxiv", "arguments": {"query": "quantum computing"}}
   ↓
3. System detects JSON function call
   ↓
4. Calls search_arxiv("quantum computing")
   ↓
5. search_arxiv() queries arXiv API
   ↓
6. Returns paper information
   ↓
7. Result displayed/spoken to you
```

---

## 🐛 Troubleshooting

### Problem: "Connection Error"
**Solution**: Make sure Ollama is running:
```bash
ollama run qwen3:8b
```

### Problem: "No microphone detected"
**Solution**: Check your microphone is connected and permitted

### Problem: "TTS not working"
**Solution**: 
```bash
# Reinstall pyttsx3
pip uninstall pyttsx3
pip install pyttsx3
```

### Problem: "LLM not calling functions"
**Possible causes**:
- Model might not follow instructions well (try `ollama pull llama3.2`)
- Temperature too high (lowering it makes responses more consistent)
- Prompt needs adjustment for your specific model

### Problem: "arXiv search times out"
**Solution**: The code has built-in fallback to simulated responses

---

## 📊 Verification Checklist

After running the app, verify these work:

- [ ] GUI opens successfully
- [ ] Can type and send text messages
- [ ] Can record and process voice
- [ ] Can perform calculations (try "What's 5+5?")
- [ ] Can search papers (try "Search quantum computing")
- [ ] Normal conversation works (try "Hello")
- [ ] Chat history is maintained
- [ ] Status bar updates correctly

---

## 📚 Files Included

| File | Purpose |
|------|---------|
| `ChatBotDeskApp_Enhanced.py` | Main application with function calling |
| `test_function_calling.py` | Test script (no GUI) |
| `README_IMPLEMENTATION.md` | Detailed implementation guide |
| `CHANGES.md` | Comparison: original vs enhanced |
| `QUICKSTART.md` | This file |

---

## 🎓 Learning Flow

**Recommended Order:**

1. **Read** `QUICKSTART.md` (this file) ← You are here!
2. **Run** `test_function_calling.py` to understand routing
3. **Try** `ChatBotDeskApp_Enhanced.py` with text input first
4. **Test** voice input after text works
5. **Read** `README_IMPLEMENTATION.md` for deep dive
6. **Compare** `CHANGES.md` to see what was added
7. **Experiment** with custom queries and edge cases

---

## 💡 Pro Tips

1. **Start with Text**: Test text input before using voice
2. **Watch Console**: Monitor console output to see function calls
3. **Be Specific**: "Calculate 5+5" works better than "what's five plus five"
4. **Try Edge Cases**: What happens with "divide by zero"?
5. **Adjust Temperature**: In code, try `temperature=0.3` for more consistent function calling
6. **Custom Functions**: Easy to add! Just follow the pattern of `search_arxiv` and `calculate`

---

## 🚀 Next Steps

Once you're comfortable:

1. **Add More Tools**: Try adding a weather lookup function
2. **Chain Functions**: Make LLM call multiple functions in sequence
3. **Improve Prompts**: Experiment with different system prompts
4. **Add Validation**: Validate function arguments before calling
5. **Implement Caching**: Cache arXiv results to avoid repeated API calls

---

## 📞 Need Help?

- Check the console output - it shows what's happening
- Look at `README_IMPLEMENTATION.md` for detailed explanations
- Run tests: `python test_function_calling.py`
- Review example queries above

---

## ✅ Success Criteria

You'll know it's working when:

- ✅ You ask "What's 5+5?" and it calls the calculate function
- ✅ You ask "Search quantum papers" and it calls search_arxiv
- ✅ You say "Hello" and it responds naturally (no function call)
- ✅ Voice input transcribes correctly
- ✅ Bot speaks the responses

---

**Congratulations!** 🎉 You now have a working voice agent with function calling capabilities!

---

**Project**: Function Calling Voice Agent
**Version**: 1.0
**Difficulty**: Intermediate
**Time to Complete**: 30-60 minutes
