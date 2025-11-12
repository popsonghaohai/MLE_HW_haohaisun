#Desktop Voice Chatbot Application (Windows)
##Features:
- Voice input recording
- Text input window
- Speech-to-text (Whisper)
- LLM response (qwen3:1.7b)
- Text-to-speech (pyttsx3)
- Chat history display
- Thinking indicator during processing
- Similar to WeChat interface
"""
## Download & Install ollama application for Windows
https://ollama.com/download/windows

## install qwen3:1.7b
ollama pull qwen3:1.7b

## run LLM qwen3:1.7b
ollama list
NAME                ID              SIZE      MODIFIED
llama3.2:3b         a80c4f17acd5    2.0 GB    5 days ago
llama3.2:1b         baf6a787fdff    1.3 GB    5 days ago
llama3.2:latest     a80c4f17acd5    2.0 GB    5 days ago
qwen3:1.7b          8f68893c685c    1.4 GB    5 days ago
glm-4.6:cloud       05277b76269f    -         5 days ago
qwen3-vl:2b         0635d9d857d4    1.9 GB    5 days ago
deepseek-r1:1.5b    e0979632db5a    1.1 GB    5 days ago

ollama run qwen3:1.7b
C:\Users\Administrator>ollama run qwen3:1.7b
>>> Send a message (/? for help)

##install Pychome community Version 
https://www.jetbrains.com/pycharm/download/other.html?_cl=MTsxOzE7T3B3NE1XWnBORDhTZzlkSEtZRlYydmJiUTFYc3pOQVI2WGRYT3lsbm5sM3AzTFhNaFhkOThjR050NEJkcjZ2cDs=


## create project folder
create the folder for the  project 
mkdir C:\Users\Administrator\DesktopVoiceChatbot

Lauch Pycharm Application 
File--> New Project -->Location input C:\Users\Administrator\PythonProject


##install packages for applications
active  Project virtual python environment 
cd C:\Users\Administrator\DesktopVoiceChatbot>
.venv\Scripts>activate
(.venv) C:\Users\Administrator\DesktopVoiceChatbot>

## create requirements.txt from File-->New--> File
requirements.txt

tkinter 
numpy 
sounddevice
soundfile
openai
openai-whisper
pyttsx3


#install all necessary packages
 python.exe -m pip install --upgrade pip
 
## download the file ChatBotDeskApp_1.1.py into C:\Users\Administrator\DesktopVoiceChatbot>

configure Pycharm environment to use python3.11 for openai-Whisper
File-->Setting-->Python-->Interpreter-->add Interpreter-->Python 3.11

open  file ChatBotDeskApp_1.1.py from  Pycharm 
run (Ctl+Shit+F10) , Pycharm ask to install all packages one by one.

numpy 
sounddevice
soundfile
openai
openai-whisper
pyttsx3


