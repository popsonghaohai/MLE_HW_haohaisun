"""
Desktop Voice Chatbot Application with Function Calling (Windows)
Features:
- Voice input recording
- Text input window
- Speech-to-text (Whisper)
- LLM response (qwen3:8b) with function calling
- Tool integration: search_arxiv, calculate
- Text-to-speech (pyttsx3)
- Chat history display
- Thinking indicator during processing
- Similar to WeChat interface
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import tempfile
from datetime import datetime
from collections import deque
from typing import Optional
import subprocess
import sys
import itertools
import json
import re

# OpenAI client for Ollama
from openai import OpenAI


def install_required_packages():
    """Install required packages if not available"""
    required_packages = {
        'whisper': 'openai-whisper',
        'pyttsx3': 'pyttsx3',
        'sounddevice': 'sounddevice',
        'soundfile': 'soundfile',
        'sympy': 'sympy',
        'arxiv': 'arxiv',
    }

    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Install packages if needed
try:
    install_required_packages()
    import whisper
    import pyttsx3
    from sympy import sympify

    WHISPER_AVAILABLE = True
    TTS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not install required packages: {e}")
    WHISPER_AVAILABLE = False
    TTS_AVAILABLE = False


# ============================================================================
# TOOL FUNCTIONS
# ============================================================================

def search_arxiv(query: str) -> str:
    """
    Search arXiv for papers related to the query.
    Returns a summary of the top result.
    """
    try:
        import arxiv
        
        print(f"🔍 Searching arXiv for: {query}")
        
        # Search arXiv
        search = arxiv.Search(
            query=query,
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = list(search.results())
        
        if not results:
            return f"No arXiv papers found for query: '{query}'"
        
        paper = results[0]
        
        # Format the result
        summary = f"""Found arXiv paper:
Title: {paper.title}
Authors: {', '.join([author.name for author in paper.authors[:3]])}
Published: {paper.published.strftime('%Y-%m-%d')}
Summary: {paper.summary[:300]}...

Read more at: {paper.entry_id}"""
        
        return summary
        
    except Exception as e:
        print(f"Error searching arXiv: {e}")
        # Fallback to simulated response
        return f"[Simulated arXiv result for '{query}']: This is a placeholder response about research in {query}. In production, this would query the actual arXiv API."


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result as a string.
    Uses sympy for safe evaluation.
    """
    try:
        print(f"🧮 Calculating: {expression}")
        
        # Use sympy for safe evaluation
        result = sympify(expression)
        
        # Evaluate if possible
        if result.is_number:
            evaluated = float(result.evalf())
            # Format nicely
            if evaluated == int(evaluated):
                return f"The result is: {int(evaluated)}"
            else:
                return f"The result is: {evaluated}"
        else:
            return f"The result is: {result}"
            
    except Exception as e:
        return f"Error calculating expression '{expression}': {str(e)}"


# ============================================================================
# FUNCTION ROUTING LOGIC
# ============================================================================

def route_llm_output(llm_output: str) -> tuple[str, bool]:
    """
    Route LLM response to the correct tool if it's a function call.
    Returns tuple of (response_text, was_function_call)
    
    Expects LLM output in JSON format like:
    {"function": "calculate", "arguments": {"expression": "2+2"}}
    or
    {"function": "search_arxiv", "arguments": {"query": "quantum computing"}}
    """
    # Try to extract JSON from the response
    # Sometimes LLM wraps JSON in markdown code blocks
    json_pattern = r'```json\s*(.*?)\s*```'
    json_match = re.search(json_pattern, llm_output, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_pattern2 = r'\{[\s\S]*?"function"[\s\S]*?\}'
        json_match2 = re.search(json_pattern2, llm_output)
        if json_match2:
            json_str = json_match2.group(0)
        else:
            json_str = llm_output
    
    try:
        output = json.loads(json_str.strip())
        func_name = output.get("function")
        args = output.get("arguments", {})
        
        print(f"📞 Function call detected: {func_name} with args {args}")
        
        if func_name == "search_arxiv":
            query = args.get("query", "")
            if query:
                result = search_arxiv(query)
                return result, True
            else:
                return "Error: No query provided for arXiv search", True
                
        elif func_name == "calculate":
            expr = args.get("expression", "")
            if expr:
                result = calculate(expr)
                return result, True
            else:
                return "Error: No expression provided for calculation", True
                
        else:
            return f"Error: Unknown function '{func_name}'", True
            
    except (json.JSONDecodeError, TypeError, AttributeError):
        # Not a JSON function call; return the text directly
        return llm_output, False


class VoiceChatbotEngine:
    """Backend engine for voice chatbot with function calling"""

    def __init__(self, model_name="qwen3:8b", max_memory=5):
        self.model_name = model_name
        self.max_memory = max_memory
        self.conversation_memory = deque(maxlen=max_memory)
        self.temp_dir = tempfile.gettempdir()

        # Initialize Whisper
        if WHISPER_AVAILABLE:
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            print("✅ Whisper loaded!")
        else:
            self.whisper_model = None

        # Store TTS availability flag
        self.tts_available = TTS_AVAILABLE
        if TTS_AVAILABLE:
            print("✅ TTS available!")
        else:
            print("⚠️ TTS not available")

        # Initialize OpenAI client for Ollama
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='qwen3:8b'
        )

        # Audio recording state
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000

    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.audio_data = []

    def record_audio_chunk(self, indata, frames, time, status):
        """Callback for audio recording"""
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def stop_recording(self) -> Optional[str]:
        """Stop recording and save audio file"""
        self.is_recording = False

        if not self.audio_data:
            return None

        try:
            # Concatenate audio chunks
            audio_array = np.concatenate(self.audio_data, axis=0)

            # Save to file
            audio_path = os.path.join(
                self.temp_dir,
                f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
            sf.write(audio_path, audio_array, self.sample_rate)

            return audio_path
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text using Whisper"""
        try:
            if not self.whisper_model:
                return "[Whisper not available]"

            if not audio_path or not os.path.exists(audio_path):
                return "[No audio file]"

            print(f"🎤 Transcribing: {audio_path}")
            result = self.whisper_model.transcribe(audio_path, language="en")
            transcribed_text = result["text"].strip()

            if not transcribed_text:
                return "[No speech detected]"

            return transcribed_text

        except Exception as e:
            return f"[Transcription error: {str(e)}]"

    def generate_response(self, user_input: str) -> str:
        """Generate LLM response with function calling support"""
        try:
            if not user_input or user_input.startswith("["):
                return "[Invalid input]"

            print(f"🤖 Generating response for: {user_input}")

            # Build context from conversation history
            context = self._build_context()

            # Enhanced system prompt with function calling instructions
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

Previous conversation:
{context}

Examples:
User: "What's 25 times 4?"
Assistant: {{"function": "calculate", "arguments": {{"expression": "25*4"}}}}

User: "Find papers about quantum computing"
Assistant: {{"function": "search_arxiv", "arguments": {{"query": "quantum computing"}}}}

User: "Hello, how are you?"
Assistant: Hello! I'm doing great, thank you for asking. How can I help you today?"""

            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ]

            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7
            )

            llm_output = response.choices[0].message.content.strip()
            print(f"💬 LLM raw output: {llm_output[:200]}...")

            # Route the output - check if it's a function call
            final_response, was_function_call = route_llm_output(llm_output)

            if was_function_call:
                print(f"✅ Function executed, result: {final_response[:100]}...")
            else:
                print(f"💭 Normal response: {final_response[:100]}...")

            # Update memory
            self.conversation_memory.append({
                "user": user_input,
                "assistant": final_response,
                "was_function_call": was_function_call,
                "timestamp": datetime.now().isoformat()
            })

            return final_response

        except Exception as e:
            return f"Error: {str(e)}"

    def _build_context(self) -> str:
        """Build context string from conversation memory"""
        if not self.conversation_memory:
            return "No previous conversation."

        context_parts = []
        for entry in self.conversation_memory:
            context_parts.append(f"User: {entry['user']}")
            context_parts.append(f"Assistant: {entry['assistant']}")

        return "\n".join(context_parts)

    def text_to_speech_and_play(self, text: str):
        """Convert text to speech and play it"""
        if not self.tts_available:
            print("⚠️ TTS not available")
            return

        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            if engine:
                try:
                    engine.stop()
                except:
                    pass

    def clear_memory(self):
        """Clear conversation memory"""
        self.conversation_memory.clear()


class DesktopChatbotApp:
    """Desktop application UI for voice chatbot"""

    def __init__(self, root):
        self.root = root
        self.root.title("🎤 Voice Chatbot with Function Calling")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')

        # Initialize engine
        self.engine = VoiceChatbotEngine()

        # UI State
        self.is_recording = False
        self.is_processing = False
        self.audio_stream = None
        self.message_queue = queue.Queue()
        self.thinking_animation = None
        self.thinking_dots = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])

        # Setup UI
        self.setup_ui()

        # Start message queue processor
        self.root.after(100, self.process_message_queue)

    def setup_ui(self):
        """Setup the user interface"""
        # Title bar
        title_frame = tk.Frame(self.root, bg='#2196F3', height=60)
        title_frame.pack(fill=tk.X, side=tk.TOP)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="🎤 Voice Chatbot with Function Calling",
            font=('Arial', 18, 'bold'),
            bg='#2196F3',
            fg='white'
        )
        title_label.pack(pady=15)

        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Chat display area
        chat_frame = tk.LabelFrame(
            main_frame,
            text="💬 Conversation",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            padx=10,
            pady=10
        )
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=('Arial', 10),
            bg='#ffffff',
            relief=tk.FLAT,
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for styling
        self.chat_display.tag_config("user", foreground="#1976D2", font=('Arial', 10, 'bold'))
        self.chat_display.tag_config("assistant", foreground="#388E3C", font=('Arial', 10, 'bold'))
        self.chat_display.tag_config("system", foreground="#F57C00", font=('Arial', 9, 'italic'))
        self.chat_display.tag_config("thinking", foreground="#9C27B0", font=('Arial', 10, 'bold'))
        self.chat_display.tag_config("function", foreground="#D32F2F", font=('Arial', 9, 'italic'))

        # Control panel
        control_frame = tk.Frame(main_frame, bg='#f0f0f0')
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Input area
        input_frame = tk.LabelFrame(
            control_frame,
            text="✍️ Text Input",
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0',
            padx=10,
            pady=10
        )
        input_frame.pack(fill=tk.X, pady=(0, 10))

        input_container = tk.Frame(input_frame, bg='#f0f0f0')
        input_container.pack(fill=tk.X)

        self.input_text = tk.Text(
            input_container,
            height=3,
            font=('Arial', 10),
            wrap=tk.WORD,
            relief=tk.SOLID,
            borderwidth=1
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.input_text.bind('<Return>', self.send_message_event)

        self.send_btn = tk.Button(
            input_container,
            text="📤 Send",
            command=self.send_message,
            font=('Arial', 10, 'bold'),
            bg='#2196F3',
            fg='white',
            activebackground='#1976D2',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=2,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.send_btn.pack(side=tk.RIGHT)

        # Voice controls
        voice_frame = tk.LabelFrame(
            control_frame,
            text="🎙️ Voice Controls",
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0',
            padx=10,
            pady=10
        )
        voice_frame.pack(fill=tk.X)

        buttons_container = tk.Frame(voice_frame, bg='#f0f0f0')
        buttons_container.pack()

        self.record_btn = tk.Button(
            buttons_container,
            text="🎙️ Start Recording",
            command=self.toggle_recording,
            font=('Arial', 11, 'bold'),
            bg='#4CAF50',
            fg='white',
            activebackground='#45a049',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=2,
            padx=20,
            pady=10,
            cursor='hand2',
            width=20
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)

        self.process_btn = tk.Button(
            buttons_container,
            text="⚙️ Process Voice",
            command=self.process_voice,
            font=('Arial', 11, 'bold'),
            bg='#FF9800',
            fg='white',
            activebackground='#F57C00',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=2,
            padx=20,
            pady=10,
            cursor='hand2',
            width=20,
            state=tk.DISABLED
        )
        self.process_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(
            buttons_container,
            text="🗑️ Clear Chat",
            command=self.clear_chat,
            font=('Arial', 11, 'bold'),
            bg='#f44336',
            fg='white',
            activebackground='#da190b',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=2,
            padx=20,
            pady=10,
            cursor='hand2',
            width=20
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Status bar
        status_frame = tk.Frame(self.root, bg='#e0e0e0', height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            status_frame,
            text="✅ Ready - Record voice or type a message",
            font=('Arial', 9),
            bg='#e0e0e0',
            fg='#333333',
            anchor=tk.W,
            padx=10
        )
        self.status_label.pack(fill=tk.X)

        # Welcome message
        self.add_message("system", "🎉 Welcome! I can help you with:")
        self.add_message("system", "  • General conversation")
        self.add_message("system", "  • Searching arXiv papers (try: 'search for quantum computing papers')")
        self.add_message("system", "  • Mathematical calculations (try: 'calculate 15 * 23')")
        self.add_message("system", "")
        self.add_message("system", "💡 You can type or use voice input!")

    def send_message_event(self, event=None):
        """Handle Enter key press"""
        if event and event.state & 0x1:  # Shift+Enter
            return
        self.send_message()
        return "break"

    def send_message(self):
        """Send text message"""
        message = self.input_text.get("1.0", tk.END).strip()

        if not message:
            return

        if self.is_processing:
            self.update_status("⏳ Already processing... Please wait")
            return

        # Clear input
        self.input_text.delete("1.0", tk.END)

        # Disable send button
        self.send_btn.config(state=tk.DISABLED)

        # Add user message
        self.add_message("user", message)

        # Process in background thread
        thread = threading.Thread(
            target=self._process_text_message,
            args=(message,),
            daemon=True
        )
        thread.start()

    def _process_text_message(self, message: str):
        """Process text message in background thread"""
        try:
            self.is_processing = True

            # Show thinking indicator
            self.message_queue.put(("thinking_start", None))
            self.message_queue.put(("status", "🤖 Thinking..."))

            # Generate response
            response = self.engine.generate_response(message)

            # Remove thinking indicator
            self.message_queue.put(("thinking_stop", None))

            if response.startswith("Error"):
                self.message_queue.put(("system", f"❌ {response}"))
                self.message_queue.put(("status", "❌ Response generation failed"))
            else:
                self.message_queue.put(("assistant", response))
                self.message_queue.put(("status", "✅ Complete - Ready for next message"))

        except Exception as e:
            self.message_queue.put(("thinking_stop", None))
            self.message_queue.put(("system", f"❌ Error: {str(e)}"))
            self.message_queue.put(("status", f"❌ Error: {str(e)}"))
        finally:
            self.is_processing = False
            self.message_queue.put(("enable_send", None))

    def add_message(self, role: str, content: str):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)

        timestamp = datetime.now().strftime("%H:%M:%S")

        if role == "user":
            self.chat_display.insert(tk.END, f"[{timestamp}] 👤 You:\n", "user")
            self.chat_display.insert(tk.END, f"{content}\n\n")
        elif role == "assistant":
            self.chat_display.insert(tk.END, f"[{timestamp}] 🤖 Alex:\n", "assistant")
            self.chat_display.insert(tk.END, f"{content}\n\n")
        elif role == "system":
            self.chat_display.insert(tk.END, f"{content}\n", "system")
        elif role == "thinking":
            # Store the position for animation
            self.thinking_text_start = self.chat_display.index(tk.END + "-1c")
            dots = next(self.thinking_dots)
            self.chat_display.insert(tk.END, f"{dots} Thinking...\n", "thinking")

        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

        # Start thinking animation if needed
        if role == "thinking":
            self.animate_thinking()

    def animate_thinking(self):
        """Animate thinking dots"""
        if self.is_processing and hasattr(self, 'thinking_text_start'):
            try:
                self.chat_display.config(state=tk.NORMAL)

                # Get current line
                line_start = self.thinking_text_start.split('.')[0]
                line_end = f"{line_start}.end"

                # Delete current line content
                self.chat_display.delete(self.thinking_text_start, line_end)

                # Insert new animation frame
                dots = next(self.thinking_dots)
                self.chat_display.insert(self.thinking_text_start, f"{dots} Thinking...", "thinking")

                self.chat_display.config(state=tk.DISABLED)

                # Schedule next frame
                self.thinking_animation = self.root.after(500, self.animate_thinking)
            except:
                pass

    def stop_thinking_animation(self):
        """Stop thinking animation"""
        if self.thinking_animation:
            self.root.after_cancel(self.thinking_animation)
            self.thinking_animation = None

        # Remove thinking indicator
        if hasattr(self, 'thinking_text_start'):
            try:
                self.chat_display.config(state=tk.NORMAL)
                line_start = self.thinking_text_start.split('.')[0]
                self.chat_display.delete(f"{line_start}.0", f"{line_start}.end + 1 lines")
                self.chat_display.config(state=tk.DISABLED)
                delattr(self, 'thinking_text_start')
            except:
                pass

    def update_status(self, message: str):
        """Update status bar"""
        self.status_label.config(text=message)

    def toggle_recording(self):
        """Toggle audio recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording audio"""
        try:
            self.is_recording = True
            self.engine.start_recording()

            # Start audio stream
            self.audio_stream = sd.InputStream(
                samplerate=self.engine.sample_rate,
                channels=1,
                callback=self.engine.record_audio_chunk
            )
            self.audio_stream.start()

            # Update UI
            self.record_btn.config(
                text="🛑 Stop Recording",
                bg="#f44336",
                activebackground="#da190b"
            )
            self.update_status("🎤 Recording... Speak now!")
            self.add_message("system", "🎤 Recording started - Speak your message")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {e}")
            self.is_recording = False

    def stop_recording(self):
        """Stop recording audio"""
        try:
            self.is_recording = False

            # Stop audio stream
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None

            # Save audio
            audio_path = self.engine.stop_recording()

            # Update UI
            self.record_btn.config(
                text="🎙️ Start Recording",
                bg="#4CAF50",
                activebackground="#45a049"
            )

            if audio_path:
                self.current_audio_path = audio_path
                self.process_btn.config(state=tk.NORMAL)
                self.update_status("✅ Recording saved - Click 'Process Voice' to continue")
                self.add_message("system", "✅ Recording stopped - Ready to process")
            else:
                self.update_status("❌ Recording failed - Try again")
                self.add_message("system", "❌ Recording failed - No audio data")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop recording: {e}")

    def process_voice(self):
        """Process recorded voice"""
        if not hasattr(self, 'current_audio_path'):
            self.update_status("❌ No audio to process")
            return

        if self.is_processing:
            self.update_status("⏳ Already processing... Please wait")
            return

        # Disable button
        self.process_btn.config(state=tk.DISABLED)
        self.update_status("⏳ Processing...")
        self.add_message("system", "⏳ Processing your voice input...")

        # Process in thread
        thread = threading.Thread(target=self._process_voice_thread, daemon=True)
        thread.start()

    def _process_voice_thread(self):
        """Process voice in background thread"""
        try:
            self.is_processing = True

            # Step 1: Transcribe
            self.message_queue.put(("status", "🎤 Transcribing audio..."))
            transcription = self.engine.transcribe_audio(self.current_audio_path)

            if transcription.startswith("["):
                self.message_queue.put(("system", f"❌ Transcription failed: {transcription}"))
                self.message_queue.put(("status", "❌ Transcription failed - Try again"))
                return

            self.message_queue.put(("user", transcription))

            # Show thinking
            self.message_queue.put(("thinking_start", None))
            self.message_queue.put(("status", "🤖 Thinking..."))

            # Step 2: Generate response
            response = self.engine.generate_response(transcription)

            # Stop thinking
            self.message_queue.put(("thinking_stop", None))

            if response.startswith("Error"):
                self.message_queue.put(("system", f"❌ {response}"))
                self.message_queue.put(("status", "❌ Response generation failed"))
                return

            self.message_queue.put(("assistant", response))
            self.message_queue.put(("status", "🔊 Speaking response..."))

            # Step 3: Speak
            self.engine.text_to_speech_and_play(response)

            self.message_queue.put(("status", "✅ Complete - Ready for next recording"))

        except Exception as e:
            self.message_queue.put(("thinking_stop", None))
            self.message_queue.put(("system", f"❌ Error: {str(e)}"))
            self.message_queue.put(("status", f"❌ Error: {str(e)}"))
        finally:
            self.is_processing = False

    def process_message_queue(self):
        """Process messages from background threads"""
        try:
            while True:
                msg_type, content = self.message_queue.get_nowait()

                if msg_type == "status":
                    self.update_status(content)
                elif msg_type == "thinking_start":
                    self.add_message("thinking", "")
                elif msg_type == "thinking_stop":
                    self.stop_thinking_animation()
                elif msg_type == "enable_send":
                    self.send_btn.config(state=tk.NORMAL)
                elif msg_type in ["user", "assistant", "system"]:
                    self.add_message(msg_type, content)

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.process_message_queue)

    def clear_chat(self):
        """Clear chat history"""
        if messagebox.askyesno("Clear Chat", "Clear all conversation history?"):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state=tk.DISABLED)

            self.engine.clear_memory()
            self.update_status("🗑️ Chat cleared - Ready to start fresh")
            self.add_message("system", "🗑️ Conversation history cleared")


def main():
    """Main entry point"""
    print("=" * 60)
    print("🎤 Desktop Voice Chatbot with Function Calling")
    print("=" * 60)
    print("\n📋 Requirements:")
    print("  1. Ollama: ollama run qwen3:8b")
    print("  2. Whisper: pip install openai-whisper")
    print("  3. TTS: pip install pyttsx3")
    print("  4. Audio: pip install sounddevice soundfile")
    print("  5. Math: pip install sympy")
    print("  6. arXiv: pip install arxiv")
    print("\n🔧 Features:")
    print("  • Voice input & output")
    print("  • Text chat interface")
    print("  • Function calling: search_arxiv & calculate")
    print("  • Multi-turn conversation with memory")
    print("\n" + "=" * 60 + "\n")

    root = tk.Tk()
    app = DesktopChatbotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
