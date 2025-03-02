import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import pygame
import os

# Configure Gemini AI API Key
API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key="AIzaSyCQPWfZlhprqYoCtNcdJAUrY_dn8TZ_6gU")

# Initialize Pygame Mixer for Audio Playback
pygame.mixer.init()

# Function to Convert Text to Speech
def speak(text):
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    os.remove("response.mp3")  # Delete the file after playing

# Function to Get Audio Input from Microphone
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand."
        except sr.RequestError:
            return "API unavailable."

# Function to Interact with Gemini AI
def ask_gemini(prompt):
    model = genai.GenerativeModel("gemini-pro")
    
    try:
        response = model.generate_content(prompt)
        
        if response and response.text:
            text = response.text.strip()

            # Limit response length to keep it conversational
            if len(text) > 200:  
                return "That's a complex topic! Here's a short summary: " + text[:200] + "..."
            
            return text
        
        return "Sorry, I couldn't process that."
    
    except Exception as e:
        return f"Error: {str(e)}"

# Main Loop for Voice Interaction
while True:
    print("\nSay something (or 'exit' to quit)...")
    user_input = listen().lower()
    
    if "exit" in user_input:
        print("Goodbye!")
        speak("Goodbye!")
        break
    
    print(f"You: {user_input}")
    response = ask_gemini(user_input)
    print(f"Gemini AI: {response}")
    speak(response)
