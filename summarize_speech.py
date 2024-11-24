import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydub import AudioSegment
import speech_recognition as sr


INPUT_FILE = r"input/test.wav"
OUTPUT_FILE = r"output/transcript.txt"

audio = AudioSegment.from_file(INPUT_FILE)
mono_audio = audio.set_channels(1)
converted_audio_path = "input/mihir_test_mono.wav"
mono_audio.export(converted_audio_path, format="wav")

# Speech recognition
recognizer = sr.Recognizer()
with sr.AudioFile(converted_audio_path) as source:
    audio_data = recognizer.record(source)
    try:
        transcribed_text = recognizer.recognize_google(audio_data)
        print("Recognized text:", transcribed_text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")




tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

input_ids = tokenizer(f"summarize: {transcribed_text}", return_tensors='pt').input_ids
outputs = model.generate(input_ids)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))




