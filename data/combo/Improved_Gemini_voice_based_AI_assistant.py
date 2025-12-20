import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import pygame
import os
import tempfile
import re

# Configure Gemini AI API Key (Use Environment Variable for Security)
genai.configure(api_key="AIzaSyCQPWfZlhprqYoCtNcdJAUrY_dn8TZ_6gU")

# Initialize Pygame Mixer for Audio Playback
pygame.mixer.init()

# Function to Remove Markdown Formatting
def clean_text(text):
    text = re.sub(r"\*\*|\*", "", text)  # Remove bold and italic markers
    text = re.sub(r"`+", "", text)  # Remove inline code formatting
    text = re.sub(r"#+ ", "", text)  # Remove headings
    text = re.sub(r"[-*] ", "", text)  # Remove bullet points
    text = re.sub(r"\n+", " ", text)  # Replace newlines with spaces
    return text.strip()

# Function to Convert Text to Speech
def speak(text):
    cleaned_text = clean_text(text)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_audio:
        tts = gTTS(text=cleaned_text, lang="en")
        tts.save(temp_audio.name)
        pygame.mixer.music.load(temp_audio.name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)  # Reduce CPU usage

# Function to Get Audio Input from Microphone
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        speak("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio).lower()
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand."
        except sr.RequestError:
            return "API unavailable."
        except Exception as e:
            return f"Error: {str(e)}"

# Function to Read Object Detection Result
def read_detection_result():
    try:
        with open("detection_result.txt", "r") as file:
            result = file.readline().strip()
            return result if result else "No object detected."
    except FileNotFoundError:
        return "Detection result file not found."

# Function to Interact with Gemini AI
def ask_gemini(user_prompt, detection_result):
    model = genai.GenerativeModel("gemini-pro")

    # Add context about detected object automatically
    context = f"Currently detected object: {detection_result}. "
    combined_prompt = context + user_prompt

    try:
        response = model.generate_content(combined_prompt)
        if response and response.text:
            text = response.text.strip()

            # Clean text from unwanted markdown
            cleaned_text = clean_text(text)

            # Limit response length to keep it conversational
            return cleaned_text if len(cleaned_text) <= 200 else cleaned_text[:200] + "..."
        
        return "Sorry, I couldn't process that."
    
    except Exception as e:
        return f"Error: {str(e)}"

# Main Loop for Voice Interaction
while True:
    print("\nSay something (or 'exit' to quit)...")
    user_input = listen()

    if "exit" in user_input:
        print("Goodbye!")
        speak("Goodbye!")
        break

    detection_result = read_detection_result()

    # Pass both user query and detection result to Gemini
    response = ask_gemini(user_input, detection_result)

    print(f"Gemini AI: {response}")
    speak(response)
