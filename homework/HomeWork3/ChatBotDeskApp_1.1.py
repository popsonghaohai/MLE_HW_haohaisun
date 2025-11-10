"""
Desktop Voice Chatbot Application (Windows)
Features:
- Voice input recording
- Text input window
- Speech-to-text (Whisper)
- LLM response (qwen3:1.7b)
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

# OpenAI client for Ollama
from openai import OpenAI


def install_required_packages():
    """Install required packages if not available"""
    required_packages = {
        'whisper': 'openai-whisper',
        'pyttsx3': 'pyttsx3',
        'sounddevice': 'sounddevice',
        'soundfile': 'soundfile',
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

    WHISPER_AVAILABLE = True
    TTS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not install required packages: {e}")
    WHISPER_AVAILABLE = False
    TTS_AVAILABLE = False


class VoiceChatbotEngine:
    """Backend engine for voice chatbot"""

    def __init__(self, model_name="qwen3:1.7b", max_memory=5):
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

        # Initialize TTS
 #       if TTS_AVAILABLE:
#            try:
 #               self.tts_engine = pyttsx3.init()
#                self.tts_engine.setProperty('rate', 150)
 #               self.tts_engine.setProperty('volume', 0.9)
  #              self.tts_engine.stop()
#                print("✅ TTS initialized!")
#            except Exception as e:
#                print(f"⚠️ TTS error: {e}")
#                self.tts_engine = None
#        else:
 #           self.tts_engine = None


            # Store TTS availability flag (don't initialize engine here)
            self.tts_available = TTS_AVAILABLE
            if TTS_AVAILABLE:
                print("✅ TTS available!")
            else:
                print("⚠️ TTS not available")

        # Initialize OpenAI client for Ollama
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='qwen3:1.7b'
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
        """Generate LLM response"""
        try:
            if not user_input or user_input.startswith("["):
                return "[Invalid input]"

            print(f"🤖 Generating response for: {user_input}")

            # Build context from conversation history
            context = self._build_context()

            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a helpful voice assistant named Alex. 
                    Provide clear, concise, and natural spoken responses.
                    Keep responses brief (2-3 sentences) and conversational.

                    Previous conversation:
                    {context}"""
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

            assistant_response = response.choices[0].message.content.strip()

            # Update memory
            self.conversation_memory.append({
                "user": user_input,
                "assistant": assistant_response,
                "timestamp": datetime.now().isoformat()
            })

            return assistant_response

        except Exception as e:
            return f"Error: {str(e)}"

    import pyttsx3

    def text_to_speech(text):
        engine = None
        try:
            engine = pyttsx3.init()

            # 设置属性
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.8)

            # 播放语音
            engine.say(text)
            engine.runAndWait()

        except Exception as e:
            print(f"Error: {e}")
        finally:
            # 确保资源被释放
            if engine:
                engine.stop()
                # 在 Windows 上，有时需要额外的清理
                try:
                    del engine
                except:
                    pass

    # 使用示例

    def text_to_speech_and_play(self, text: str):
        """
        Convert text to speech and play it
        Initialize TTS, speak, and release resources in one function
        """
        # Validate input
        if not text or text.startswith("[") or text.startswith("Error"):
            return

        print(f"🔊 Speaking: {text[:50]}...")

        engine = None
        try:
            # Step 1: Initialize TTS engine
            engine = pyttsx3.init()

            # Step 2: Configure engine
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)

            # Step 3: Speak the text
            engine.say(text)
            engine.runAndWait()

            print("✅ Speech completed")

        except Exception as e:
            print(f"❌ TTS error: {e}")

        finally:
            # Step 4: Release TTS resources
            if engine:
                try:
                    engine.stop()
                except:
                    pass

                try:
                    # Properly dispose of the engine
                    del engine
                except:
                    pass

    def _build_context(self) -> str:
        """Build conversation context"""
        if not self.conversation_memory:
            return "This is the start of the conversation."

        context_parts = []
        for i, turn in enumerate(self.conversation_memory, 1):
            context_parts.append(f"Turn {i} - User: {turn['user']}")
            context_parts.append(f"Turn {i} - Assistant: {turn['assistant']}")

        return "\n".join(context_parts)

    def clear_memory(self):
        """Clear conversation memory"""
        self.conversation_memory.clear()
        print("🗑️ Memory cleared")


class DesktopChatbotApp:
    """Desktop GUI application for voice chatbot"""

    def __init__(self, root):
        self.root = root
        self.root.title("🎤 Voice Chatbot - Desktop")
        self.root.geometry("900x750")
        self.root.configure(bg="#f0f0f0")

        # Initialize engine
        self.engine = VoiceChatbotEngine()

        # Queue for thread-safe UI updates
        self.message_queue = queue.Queue()

        # Recording state
        self.is_recording = False
        self.audio_stream = None

        # Processing state
        self.is_processing = False
        self.thinking_animation = None

        # Setup UI
        self.setup_ui()

        # Start message processor
        self.root.after(100, self.process_message_queue)

    def setup_ui(self):
        """Setup the user interface"""

        # Title
        title_frame = tk.Frame(self.root, bg="#2196F3", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="🎤 Voice Chatbot Assistant",
            font=("Arial", 18, "bold"),
            bg="#2196F3",
            fg="white"
        )
        title_label.pack(pady=15)

        # Chat display area
        chat_frame = tk.Frame(self.root, bg="#f0f0f0")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Chat history (scrolled text)
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Arial", 11),
            bg="white",
            fg="black",
            state=tk.DISABLED,
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Configure tags for different message types
        self.chat_display.tag_config("user", foreground="#2196F3", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("assistant", foreground="#4CAF50", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("system", foreground="#FF9800", font=("Arial", 10, "italic"))
        self.chat_display.tag_config("thinking", foreground="#9C27B0", font=("Arial", 10, "italic"))

        # Text input area
        input_frame = tk.Frame(self.root, bg="#f0f0f0")
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Text input label
        input_label = tk.Label(
            input_frame,
            text="💬 Type your message:",
            font=("Arial", 10, "bold"),
            bg="#f0f0f0",
            fg="#333"
        )
        input_label.pack(anchor=tk.W, pady=(0, 5))

        # Text input box and send button frame
        text_input_frame = tk.Frame(input_frame, bg="#f0f0f0")
        text_input_frame.pack(fill=tk.X)

        self.text_input = tk.Text(
            text_input_frame,
            font=("Arial", 11),
            bg="white",
            fg="black",
            height=3,
            wrap=tk.WORD,
            relief=tk.SOLID,
            borderwidth=1
        )
        self.text_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Send button
        self.send_btn = tk.Button(
            text_input_frame,
            text="📤\nSend",
            font=("Arial", 10, "bold"),
            bg="#2196F3",
            fg="white",
            activebackground="#0b7dda",
            command=self.send_text_message,
            width=8,
            height=3
        )
        self.send_btn.pack(side=tk.RIGHT)

        # Bind Enter key (with Shift+Enter for new line)
        self.text_input.bind("<Return>", self.on_enter_key)

        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="🎤 Ready - Click 'Start Recording' or type a message",
            font=("Arial", 10),
            bg="#e0e0e0",
            fg="black",
            anchor=tk.W,
            padx=10
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)

        # Control panel
        control_frame = tk.Frame(self.root, bg="#f0f0f0")
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Record button
        self.record_btn = tk.Button(
            control_frame,
            text="🎙️ Start Recording",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            activebackground="#45a049",
            command=self.toggle_recording,
            width=18,
            height=2
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)

        # Process button (disabled initially)
        self.process_btn = tk.Button(
            control_frame,
            text="⚡ Process Voice",
            font=("Arial", 12, "bold"),
            bg="#2196F3",
            fg="white",
            activebackground="#0b7dda",
            command=self.process_voice,
            width=18,
            height=2,
            state=tk.DISABLED
        )
        self.process_btn.pack(side=tk.LEFT, padx=5)

        # Clear button
        clear_btn = tk.Button(
            control_frame,
            text="🗑️ Clear",
            font=("Arial", 12, "bold"),
            bg="#f44336",
            fg="white",
            activebackground="#da190b",
            command=self.clear_chat,
            width=12,
            height=2
        )
        clear_btn.pack(side=tk.RIGHT, padx=5)

    def on_enter_key(self, event):
        """Handle Enter key press"""
        # Shift+Enter: new line (default behavior)
        # Enter: send message
        if not event.state & 0x1:  # Check if Shift is NOT pressed
            self.send_text_message()
            return "break"  # Prevent default new line

    def send_text_message(self):
        """Send text message from input box"""
        message = self.text_input.get("1.0", tk.END).strip()

        if not message:
            self.update_status("❌ Please type a message")
            return

        if self.is_processing:
            self.update_status("⏳ Already processing... Please wait")
            return

        # Clear input box
        self.text_input.delete("1.0", tk.END)

        # Add user message to chat
        self.add_message("user", message)
        self.update_status("⏳ Processing your message...")

        # Disable send button during processing
        self.send_btn.config(state=tk.DISABLED)

        # Run processing in separate thread
        thread = threading.Thread(target=self._process_text_thread, args=(message,), daemon=True)
        thread.start()

    def _process_text_thread(self, user_text: str):
        """Process text input in background thread"""
        try:
            self.is_processing = True

            # Show thinking indicator
            self.message_queue.put(("thinking_start", None))
            self.message_queue.put(("status", "🤖 Thinking..."))

            # Generate response
            response = self.engine.generate_response(user_text)

            # Remove thinking indicator
            self.message_queue.put(("thinking_stop", None))

            if response.startswith("Error"):
                self.message_queue.put(("system", f"❌ {response}"))
                self.message_queue.put(("status", "❌ Response generation failed"))
            else:
                self.message_queue.put(("assistant", response))
                self.message_queue.put(("status", "🔊 Speaking response..."))

                # Speak response (ONLY ONCE)
                self.engine.text_to_speech_and_play(response)

                self.message_queue.put(("status", "✅ Complete - Ready for next message"))

        except Exception as e:
            self.message_queue.put(("thinking_stop", None))
            self.message_queue.put(("system", f"❌ Error: {str(e)}"))
            self.message_queue.put(("status", f"❌ Error: {str(e)}"))
        finally:
            self.is_processing = False
            # Re-enable send button
            self.message_queue.put(("enable_send", None))

    def add_message(self, sender: str, message: str):
        """Add message to chat display"""
        self.chat_display.config(state=tk.NORMAL)

        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")

        if sender == "user":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] You: ", "user")
            self.chat_display.insert(tk.END, f"{message}\n")
        elif sender == "assistant":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] Assistant: ", "assistant")
            self.chat_display.insert(tk.END, f"{message}\n")
        elif sender == "thinking":
            # Placeholder for thinking animation
            self.thinking_mark = self.chat_display.index(tk.END)
            self.chat_display.insert(tk.END, f"\n[{timestamp}] Assistant: ", "assistant")
            self.chat_display.insert(tk.END, "🤖⏳Thinking...", "thinking")
            self.thinking_text_start = self.chat_display.index(f"{self.thinking_mark} + 2 lines")
            self.start_thinking_animation()
        else:  # system
            self.chat_display.insert(tk.END, f"\n[{timestamp}] ", "system")
            self.chat_display.insert(tk.END, f"{message}\n", "system")

        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def start_thinking_animation(self):
        """Start animated thinking indicator"""
        self.thinking_dots = itertools.cycle([".", "..", "..."])
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

                # Insert new animation frame with rotating dots
                dots = next(self.thinking_dots)
                self.chat_display.insert(self.thinking_text_start, f"{dots}", "thinking")

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

        # Remove thinking indicator from chat
        if hasattr(self, 'thinking_text_start'):
            try:
                self.chat_display.config(state=tk.NORMAL)
                line_start = self.thinking_text_start.split('.')[0]

                # Delete the entire thinking line
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
            # Start recording
            self.start_recording()
        else:
            # Stop recording
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
        """Process recorded voice (transcribe, generate response, speak)"""
        if not hasattr(self, 'current_audio_path'):
            self.update_status("❌ No audio to process")
            return

        if self.is_processing:
            self.update_status("⏳ Already processing... Please wait")
            return

        # Disable button during processing
        self.process_btn.config(state=tk.DISABLED)
        self.update_status("⏳ Processing...")
        self.add_message("system", "⏳ Processing your voice input...")

        # Run processing in separate thread
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

            # Show thinking indicator
            self.message_queue.put(("thinking_start", None))
            self.message_queue.put(("status", "🤖 Thinking..."))

            # Step 2: Generate response
            response = self.engine.generate_response(transcription)

            # Remove thinking indicator
            self.message_queue.put(("thinking_stop", None))

            if response.startswith("Error"):
                self.message_queue.put(("system", f"❌ {response}"))
                self.message_queue.put(("status", "❌ Response generation failed"))
                return

            self.message_queue.put(("assistant", response))
            self.message_queue.put(("status", "🔊 Speaking response..."))

            # Step 3: Speak response
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
    print("🎤 Desktop Voice Chatbot Application")
    print("=" * 60)
    print("\n📋 Requirements:")
    print("  1. Ollama: ollama run qwen3:1.7b")
    print("  2. Whisper: pip install openai-whisper")
    print("  3. TTS: pip install pyttsx3")
    print("  4. Audio: pip install sounddevice soundfile")
    print("\n" + "=" * 60 + "\n")

    root = tk.Tk()
    app = DesktopChatbotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()