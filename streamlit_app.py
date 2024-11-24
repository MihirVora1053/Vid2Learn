import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydub import AudioSegment
import speech_recognition as sr
import os

# Load the pre-trained tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

# Initialize model
tokenizer, model = load_model()

def process_audio(file_path):
    try:
        temp_file_path = "temp_uploaded_audio.wav"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        # Convert audio to mono
        audio = AudioSegment.from_file(temp_file_path)

        mono_audio = audio.set_channels(1)

        converted_audio_path = "temp_audio.wav"

        mono_audio.export(converted_audio_path, format="wav")
        # Perform speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(converted_audio_path) as source:
            audio_data = recognizer.record(source)
            transcribed_text = recognizer.recognize_google(audio_data)
        return transcribed_text
    except sr.UnknownValueError:
        return "Speech Recognition could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results; {e}"
    except Exception as e:
        return f"Error processing audio: {e}"

def summarize_text(text):
    input_ids = tokenizer(f"summarize: {text}", return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=130, min_length=30, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit App
st.title("Audio Transcription and Summarization")
st.write("Upload an audio file to transcribe and summarize its content.")

# File upload
uploaded_file = st.file_uploader("Upload Audio File (WAV format)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    with st.spinner("Processing audio..."):
        transcribed_text = process_audio(uploaded_file)
        st.text_area("Transcribed Text", transcribed_text, height=200)
        
        if transcribed_text and not transcribed_text.startswith("Error"):
            with st.spinner("Summarizing text..."):
                summarized_text = summarize_text(transcribed_text)
                st.subheader("Summarized Text")
                st.write(summarized_text)
        else:
            st.error(transcribed_text)
