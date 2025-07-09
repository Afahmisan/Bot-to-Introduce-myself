# Bot-to-Introduce-myself
Perkenalan DiriKu is an interactive Python chatbot that uses speech recognition and text-to-speech (TTS) to answer questions about personal information. It matches user input to predefined answers using TF-IDF and cosine similarity. The chatbot can be exited by saying "exit". Requires pandas, pyttsx3, SpeechRecognition, and scikit-learn.

# Perkenalan DiriKu - Interactive Chatbot with Speech Recognition and Text-to-Speech

## Overview
Perkenalan DiriKu is an interactive Python chatbot that answers personal questions using voice input. It uses speech recognition to convert the user's voice to text and text-to-speech (TTS) for vocal responses. The chatbot determines the most relevant answer using TF-IDF Vectorization and Cosine Similarity.

## Features
- **Speech Recognition**: Converts voice input into text using Google Speech Recognition API.
- **Text-to-Speech**: Speaks the response using `pyttsx3`.
- **Cosine Similarity Matching**: Finds the most relevant answer by comparing the user’s input to predefined questions.
- **Interactive Q&A**: Responds to questions like "Siapa nama lengkap kamu?" and "Berapa usia kamu?".
- **Exit Command**: Ends the conversation when the user says "exit".

## Technologies Used
- **Python**: Main programming language.
- **pandas**: For managing and storing question-answer data.
- **pyttsx3**: For converting text to speech.
- **speech_recognition**: For converting speech to text.
- **scikit-learn**: For text vectorization and calculating similarity using cosine similarity.

## Installation
To install the required dependencies, use the following command:
```bash
pip install pandas pyttsx3 SpeechRecognition scikit-learn
