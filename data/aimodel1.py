import os
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
import pygame
import gtts
import requests

# Configure API key securely
genai.configure(api_key="AIzaSyCQPWfZlhprqYoCtNcdJAUrY_dn8TZ_6gU")

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech rate
engine.setProperty('volume', 1.0)  # Set volume to max
voices = engine.getProperty('voices')
if len(voices) > 1:
    engine.setProperty('voice', voices[1].id)
else:
    engine.setProperty('voice', voices[0].id)

# Check Internet Connection
def is_connected():
    try:
        requests.get("http://www.google.com", timeout=3)
        return True
    except requests.ConnectionError:
        return False

# Use GTTS with Pygame if internet is available, else fallback to pyttsx3
def speak_output(text):
    print(f"Gemini AI: {text}")
    
    if is_connected():
        try:
            tts = gtts.gTTS(text, lang="en")
            tts.save("response.mp3")
            
            pygame.mixer.init()
            pygame.mixer.music.load("response.mp3")
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                continue  # Wait for speech to finish
        except Exception as e:
            print(f"GTTS Error: {e}. Falling back to pyttsx3.")
            engine.say(text)
            engine.runAndWait()
    else:
        engine.say(text)
        engine.runAndWait()

# Ask Gemini AI
def ask_gemini(prompt):
    model = genai.GenerativeModel("gemini-pro")
    try:
        response = model.generate_content(prompt)
        if response and response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return "Sorry, I couldn't process that."
    except Exception as e:
        return f"Error: {str(e)}"

# Listen from Microphone
def listen_from_mic():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
    except sr.RequestError:
        print("Speech recognition service is not available.")
    except Exception as e:
        print(f"Microphone error: {e}")
    return None

# Main Loop
while True:
    user_input = listen_from_mic()
    if user_input:
        if user_input.lower() in ["exit", "quit", "stop"]:
            speak_output("Goodbye!")
            break
        response = ask_gemini(user_input)
        speak_output(response)
