from gtts import gTTS
import os

text = "Hello, this is a test."
tts = gTTS(text=text, lang='en')

try:
    tts.save("response.mp3")
    print("response.mp3 successfully created!")  # Debug message
except Exception as e:
    print(f"Error saving response.mp3: {e}")
