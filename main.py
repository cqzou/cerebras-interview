from flask import Flask, request, Response, send_file, send_from_directory
from llm_utils import *
from faster_whisper import WhisperModel
import tempfile
import queue
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import io
import wave
from datetime import datetime
import os
import pyaudio
import random

app = Flask(__name__)
model = WhisperModel("base", device="cpu", compute_type="int8")
audio_queue = queue.Queue()
recording = False

INITIAL_QUESTION = {
    "name": "generate_question",
    "instruction": "You are the interviewer and want to explore the participant's "
                  "!<TOPIC>!. Generate a friendly greeting and opening question."
}

FOLLOW_UP_QUESTION = {
    "name": "generate_question",
    "instruction": "!<CONTEXT>!\n\nYou are the interviewer and you want to "
                  "explore the participant's !<TOPIC>!. Given the interview "
                  "context so far, generate the interviewer's reaction and next "
                  "question. Keep a natural, conversational tone."
}

def audio_callback(indata, frames, time, status):
    if recording:
        audio_queue.put(indata.copy())

def generate_audio_stream(text):
    p = pyaudio.PyAudio()
    stream = p.open(format=8,
                   channels=1,
                   rate=24000,
                   output=True)
    
    try:
        with oai.audio.speech.with_streaming_response.create(
            model="tts-1-hd",
            voice="nova",
            input=text,
            response_format="pcm"
        ) as response:
            for chunk in response.iter_bytes(1024):
                stream.write(chunk)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def generate_audio(text):
    print(f"Generating TTS for: {text}")
    threading.Thread(target=generate_audio_stream, args=(text,)).start()
    return text

def generate_filler_audio():
    fillers = [
        "Uhh... Hmm.",
        "Uhh... Hmm, okay.",
        "Ah... Interesting.",
        "Ah...",
        "Ah... Okay.",
        "Mm-hmm..."
    ]
    filler = random.choice(fillers)
    
    p = pyaudio.PyAudio()
    stream = p.open(format=8,
                   channels=1,
                   rate=24000,
                   output=True)
    
    try:
        with oai.audio.speech.with_streaming_response.create(
            model="tts-1-hd",
            voice="nova",
            input=filler,
            response_format="pcm"
        ) as response:
            for chunk in response.iter_bytes(1024):
                stream.write(chunk)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def get_interviewer_response(context, topic, is_initial, max_retries=3):
    for attempt in range(max_retries):
        try:
            placeholders = {"context": context, "topic": topic}
            response = mod_gen_cer(
                [INITIAL_QUESTION if is_initial else FOLLOW_UP_QUESTION], 
                placeholders=placeholders
            )
            
            if not response or "generate_question" not in response:
                print(f"Failed to get valid response (attempt {attempt + 1}/{max_retries})")
                continue
                
            text = response["generate_question"]
            if not text or len(text.strip()) == 0:
                print(f"Empty response text (attempt {attempt + 1}/{max_retries})")
                continue
                
            return text
            
        except Exception as e:
            print(f"Error generating response (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
    
    raise Exception("Failed to generate valid response after all retries")

@app.route("/")
def index():
    return send_file("static/index.html")

@app.route("/start_recording", methods=["POST"])
def start_recording():
    global recording
    recording = True
    return {"status": "started"}

@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    if 'audio' not in request.files:
        return {"error": "No audio file"}, 400
        
    audio_file = request.files['audio']
    
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        audio_file.save(f.name)
        
        with open(f.name, "rb") as audio_data:
            transcript = oai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_data,
                response_format="text"
            )
    
    return {"text": transcript}

@app.route("/get_response", methods=["POST"])
def get_response():
    context = request.json.get("context", "")
    topic = request.json.get("topic", "general thoughts")
    is_initial = request.json.get("is_initial", False)
    
    # Play filler audio only for follow-up questions
    if not is_initial:
        threading.Thread(target=generate_filler_audio).start()
    
    try:
        text = get_interviewer_response(context, topic, is_initial)
        generate_audio(text)
        return {"text": text}
    except Exception as e:
        print(f"Error in get_response: {e}")
        return {"error": str(e)}, 500

@app.route("/save_transcript", methods=["POST"])
def save_transcript():
    transcript = request.json.get("transcript", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interview_{timestamp}.txt"
    
    with open(f"transcripts/{filename}", "w") as f:
        f.write(transcript)
    
    return {"filename": filename}

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    sd.default.channels = 1
    sd.default.samplerate = 16000
    sd.default.dtype = np.float32
    
    stream = sd.InputStream(callback=audio_callback)
    stream.start()
    
    os.makedirs("transcripts", exist_ok=True)
    
    app.run(debug=True)