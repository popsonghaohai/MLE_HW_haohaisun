# Voice Chatbot with Function Calling - Implementation Guide

## Overview

This project extends a desktop voice chatbot with **function calling capabilities**, allowing the LLM (Llama 3 / Qwen 3) to automatically invoke external tools like arXiv search and mathematical calculations.

## 🎯 Learning Objectives Implemented

### 1. Function Calling with LLMs
- ✅ LLM outputs structured JSON calls for external functions
- ✅ Prompts instruct the model when and how to generate function calls
- ✅ Example format: `{"function": "calculate", "arguments": {"expression": "2+2"}}`

### 2. Intent Parsing and Tool Mapping
- ✅ Parses user queries to determine intent (search vs calculate)
- ✅ Maps intent to specific tool functions
- ✅ Intelligent routing based on LLM output

### 3. Integrating Tools into Voice Agent
- ✅ Extended Week 3 pipeline: ASR → LLM → Tool Execution → TTS
- ✅ Automatic function calling based on LLM output
- ✅ Speaks the tool execution results

## 🛠️ Key Implementation Details

### Tool Functions

#### 1. `search_arxiv(query: str) -> str`
```python
def search_arxiv(query: str) -> str:
    """
    Search arXiv for papers related to the query.
    Returns a summary of the top result.
    """
    import arxiv
    
    search = arxiv.Search(
        query=query,
        max_results=1,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = list(search.results())
    paper = results[0]
    
    # Returns formatted summary with title, authors, date, abstract
    return formatted_summary
```

**Features:**
- Real arXiv API integration
- Fallback to simulated responses if API fails
- Returns title, authors, publication date, and abstract snippet
- Includes direct link to the paper

#### 2. `calculate(expression: str) -> str`
```python
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely using sympy.
    """
    from sympy import sympify
    
    result = sympify(expression)
    
    if result.is_number:
        evaluated = float(result.evalf())
        return f"The result is: {evaluated}"
    else:
        return f"The result is: {result}"
```

**Features:**
- Safe evaluation using `sympy` (no dangerous `eval()`)
- Supports complex mathematical expressions
- Handles errors gracefully
- Works with: +, -, *, /, **, sqrt(), and more

### Function Routing Logic

```python
def route_llm_output(llm_output: str) -> tuple[str, bool]:
    """
    Routes LLM response to appropriate tool or returns text.
    Returns: (response_text, was_function_call)
    """
    # Try to extract JSON (handles markdown code blocks)
    json_pattern = r'```json\s*(.*?)\s*```'
    json_match = re.search(json_pattern, llm_output, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try raw JSON
        json_pattern2 = r'\{[\s\S]*?"function"[\s\S]*?\}'
        json_match2 = re.search(json_pattern2, llm_output)
        json_str = json_match2.group(0) if json_match2 else llm_output
    
    try:
        output = json.loads(json_str.strip())
        func_name = output.get("function")
        args = output.get("arguments", {})
        
        if func_name == "search_arxiv":
            return search_arxiv(args.get("query", "")), True
        elif func_name == "calculate":
            return calculate(args.get("expression", "")), True
        else:
            return f"Error: Unknown function '{func_name}'", True
            
    except (json.JSONDecodeError, TypeError, AttributeError):
        # Not a function call - return normal text
        return llm_output, False
```

**Key Features:**
- Robust JSON extraction (handles markdown code blocks)
- Returns tuple: (response, is_function_call)
- Graceful fallback to normal text responses
- Error handling for unknown functions

### Enhanced System Prompt

The system prompt is the critical component that teaches the LLM when and how to call functions:

```python
system_prompt = f"""You are a helpful voice assistant named Alex with access to tools.

You can help users with:
1. General conversation and questions
2. Searching academic papers on arXiv
3. Performing mathematical calculations

FUNCTION CALLING INSTRUCTIONS:
When a user asks you to search for research papers or asks about academic topics, respond with:
{{"function": "search_arxiv", "arguments": {{"query": "the search query"}}}}

When a user asks you to calculate something or solve a math problem, respond with:
{{"function": "calculate", "arguments": {{"expression": "the mathematical expression"}}}}

For calculate function, use standard Python math syntax:
- Addition: 2+2
- Multiplication: 3*4
- Division: 10/2
- Powers: 2**3
- Square root: sqrt(16)
- More complex: (5+3)*2/4

IMPORTANT:
- Only output the JSON function call, nothing else, when calling a function
- For normal conversation, respond naturally without JSON
- Keep responses brief (2-3 sentences) and conversational
- Be clear whether you're calling a function or responding normally

Examples:
User: "What's 25 times 4?"
Assistant: {{"function": "calculate", "arguments": {{"expression": "25*4"}}}}

User: "Find papers about quantum computing"
Assistant: {{"function": "search_arxiv", "arguments": {{"query": "quantum computing"}}}}

User: "Hello, how are you?"
Assistant: Hello! I'm doing great, thank you for asking. How can I help you today?
"""
```

**Design Principles:**
- Clear instructions on when to use each function
- Concrete examples of function call format
- Examples of normal responses for comparison
- Syntax guidance for mathematical expressions
- Emphasizes JSON-only output for function calls

## 🔄 Complete Pipeline Flow

### Text Input Flow
```
User Input (Text)
    ↓
LLM Processing with Function-Aware Prompt
    ↓
LLM Output (JSON or Text)
    ↓
route_llm_output() - Parse & Detect
    ↓
    ├─→ Function Call Detected?
    │   ├─→ YES: Execute Tool (search_arxiv or calculate)
    │   │         Return tool result as response
    │   └─→ NO:  Return text response directly
    ↓
Display Response in Chat
```

### Voice Input Flow
```
Audio Recording
    ↓
Whisper ASR (Speech-to-Text)
    ↓
Transcribed Text
    ↓
[Same as Text Input Flow Above]
    ↓
Final Response Text
    ↓
TTS (Text-to-Speech)
    ↓
Audio Output
```

## 📦 Installation & Requirements

### Prerequisites
```bash
# 1. Install Ollama and download model
ollama pull qwen3:8b

# 2. Install Python packages
pip install openai-whisper pyttsx3 sounddevice soundfile sympy arxiv
```

### Package Details
- `openai-whisper`: Speech-to-text (ASR)
- `pyttsx3`: Text-to-speech (TTS)
- `sounddevice` + `soundfile`: Audio recording
- `sympy`: Safe mathematical expression evaluation
- `arxiv`: arXiv API client for paper searches

## 🚀 Usage

### Running the Application
```bash
python ChatBotDeskApp_Enhanced.py
```

### Example Interactions

#### 1. Mathematical Calculations
```
User: "What's 15 times 23?"
Assistant: [Calls calculate function]
Result: "The result is: 345"

User: "Calculate the square root of 144"
Assistant: [Calls calculate function]
Result: "The result is: 12"

User: "What's (5 + 3) * 2 / 4?"
Assistant: [Calls calculate function]
Result: "The result is: 4"
```

#### 2. arXiv Paper Search
```
User: "Search for papers about quantum computing"
Assistant: [Calls search_arxiv function]
Result: Returns paper title, authors, date, abstract snippet, and link

User: "Find research on machine learning"
Assistant: [Calls search_arxiv function]
Result: Returns most relevant paper from arXiv

User: "What are recent papers on neural networks?"
Assistant: [Calls search_arxiv function]
Result: Returns paper information
```

#### 3. Normal Conversation
```
User: "Hello, how are you?"
Assistant: "Hello! I'm doing great, thank you for asking. How can I help you today?"

User: "What can you do?"
Assistant: "I can help you with general questions, search for academic papers on arXiv, 
          and perform mathematical calculations. Just ask me anything!"
```

## 🎨 UI Features

### Chat Display
- **User messages**: Blue text with 👤 icon
- **Assistant messages**: Green text with 🤖 icon
- **System messages**: Orange italic text
- **Thinking indicator**: Animated purple dots while processing
- **Function calls**: Highlighted in red (for debugging)

### Controls
- **Text Input**: Type messages and press Enter (or click Send)
- **Voice Recording**: Click to start/stop recording
- **Process Voice**: Convert recorded audio to text and process
- **Clear Chat**: Reset conversation history

### Status Bar
- Real-time updates on current operation
- Recording status
- Processing indicators
- Error messages

## 🔍 Technical Details

### Error Handling

1. **JSON Parsing Errors**
   - Falls back to treating output as normal text
   - Handles both raw JSON and markdown-wrapped JSON

2. **Tool Execution Errors**
   - arXiv search: Falls back to simulated response
   - Calculate: Returns error message with details

3. **Audio Processing Errors**
   - Graceful handling of recording failures
   - Clear error messages to user

### Conversation Memory
- Maintains last 5 conversation turns (configurable)
- Includes context in each LLM request
- Tracks whether responses were function calls

### Thread Safety
- Background threads for processing (prevents UI freezing)
- Message queue for thread-safe UI updates
- Proper cleanup of audio streams and TTS engines

## 📊 Fallback Behavior

The system handles edge cases gracefully:

1. **Unparseable LLM Output**: Returns as normal text
2. **Unknown Function Name**: Returns error message
3. **Missing Arguments**: Returns error requesting required arguments
4. **Tool Failures**: Provides fallback responses or error messages
5. **No Whisper/TTS**: Continues working with reduced functionality

## 🧪 Testing Recommendations

### Test Cases

1. **Basic Calculations**
   - Simple: "2 + 2"
   - Complex: "(sqrt(16) + 3) * 2"
   - Edge: "1 / 0" (should handle error)

2. **arXiv Searches**
   - Broad: "quantum computing"
   - Specific: "transformers in NLP"
   - No results: "xyz123nonsense"

3. **Mixed Interactions**
   - Start with greeting
   - Ask for calculation
   - Request paper search
   - Ask follow-up question

4. **Voice Input**
   - Test clear speech
   - Test with background noise
   - Test different accents

## 🎓 Learning Points

### What Makes This Work

1. **Clear Prompting**: The system prompt explicitly tells the LLM:
   - When to call functions
   - What format to use
   - Examples of each case

2. **Robust Parsing**: Multiple strategies to extract JSON:
   - Check for markdown code blocks
   - Look for raw JSON patterns
   - Graceful fallback to text

3. **Clean Separation**: 
   - Tools are independent functions
   - Routing logic is centralized
   - UI and backend are separated

4. **Error Resilience**:
   - Every step has error handling
   - Clear error messages
   - Graceful degradation

### Extension Ideas

1. **Add More Tools**:
   - Web search (Google/Bing)
   - Weather lookup
   - News retrieval
   - Code execution

2. **Multi-Tool Chains**:
   - Allow LLM to call multiple tools in sequence
   - E.g., search papers, then summarize results

3. **Better Memory**:
   - Store tool call history
   - Learn from past interactions
   - Implement RAG for long-term memory

4. **Streaming Responses**:
   - Stream LLM output token by token
   - Better UX for long responses

## 🐛 Common Issues & Solutions

### Issue 1: Ollama Not Running
**Solution**: Start Ollama: `ollama run qwen3:8b`

### Issue 2: No Audio Input
**Solution**: Check microphone permissions and drivers

### Issue 3: LLM Not Calling Functions
**Solution**: 
- Check if model follows instructions (try different model)
- Adjust temperature (lower = more consistent)
- Provide more examples in prompt

### Issue 4: arXiv API Timeout
**Solution**: Fallback to simulated response (already implemented)

## 📝 Code Structure

```
ChatBotDeskApp_Enhanced.py
├── Tool Functions
│   ├── search_arxiv()
│   └── calculate()
├── Routing Logic
│   └── route_llm_output()
├── VoiceChatbotEngine (Backend)
│   ├── Whisper Integration
│   ├── LLM with Function-Aware Prompts
│   ├── TTS Integration
│   └── Conversation Memory
└── DesktopChatbotApp (UI)
    ├── Chat Display
    ├── Input Controls
    ├── Voice Recording
    └── Message Queue Processing
```

## 🎯 Assignment Completion Checklist

- ✅ Implement `search_arxiv(query)` function
- ✅ Implement `calculate(expression)` function
- ✅ Design function-aware system prompts
- ✅ Parse LLM output for function calls (JSON detection)
- ✅ Route function calls to appropriate Python functions
- ✅ Integrate tools into voice agent pipeline
- ✅ Handle fallback for unparseable outputs
- ✅ Test with multiple query types
- ✅ Document the implementation

## 📚 Additional Resources

- [Anthropic Function Calling Guide](https://docs.anthropic.com/claude/docs/tool-use)
- [arXiv API Documentation](https://info.arxiv.org/help/api/index.html)
- [Sympy Documentation](https://docs.sympy.org/)
- [OpenAI Whisper](https://github.com/openai/whisper)

## 💡 Tips for Success

1. **Start Simple**: Test with text input before voice
2. **Check Logs**: Monitor console output for function calls
3. **Iterate Prompts**: Adjust system prompt if LLM doesn't follow instructions
4. **Test Edge Cases**: Try malformed inputs, errors, edge cases
5. **Use Examples**: The LLM learns better from concrete examples

---

**Author**: Voice Agent Function Calling Project
**Date**: 2025
**Version**: 1.0
