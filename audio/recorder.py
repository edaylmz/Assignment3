import os
import sys
import wave
import struct
import pyaudio

from audio.utils import zero_crossing_rate
from config import (
    SAMPLING_RATE,
    CHANNELS,
    CHUNK_SIZE,
    SAMPLE_WIDTH,
    SILENCE_THRESHOLD,
    ZCR_THRESHOLD,
    MAX_SILENCE_DURATION,
    RECORDINGS_DIR
)


def record_audio(output_filename):
    """
    Records audio from the microphone using automatic endpoint detection.
    Stops recording after MAX_SILENCE_DURATION seconds of silence.
    Saves as 16-bit PCM .wav file at 16kHz.
    """

    # Ensure recordings directory exists
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16, # ← 16-bit PCM format
        channels=CHANNELS,
        rate=SAMPLING_RATE, # ← This sets sampling rate to 16kHz
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )


    print("Recording started. Speak into the microphone...")

    frames = []
    silent_chunks = 0
    max_silent_chunks = int(MAX_SILENCE_DURATION * SAMPLING_RATE / CHUNK_SIZE)

    while True:
        data = stream.read(CHUNK_SIZE)
        frames.append(data)

        # Convert bytes to int16 samples
        samples = struct.unpack(f"{len(data) // 2}h", data)
        avg_amplitude = sum(abs(x) for x in samples) / len(samples)
        zcr_value = zero_crossing_rate(samples)

        print(f"Amplitude: {avg_amplitude:.2f}, ZCR: {zcr_value:.3f}")

        is_silent = avg_amplitude < SILENCE_THRESHOLD and zcr_value < ZCR_THRESHOLD

        if is_silent:
            silent_chunks += 1
            print(f"Silence detected: {silent_chunks}/{max_silent_chunks}")
        else:
            silent_chunks = 0
            print("Speech detected.")

        if silent_chunks > max_silent_chunks:
            print("Silence threshold reached. Stopping recording.")
            break

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the WAV file
    file_path = os.path.join(RECORDINGS_DIR, output_filename)
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16)) # ← 16-bit = 2 bytes per sample
        wf.setframerate(SAMPLING_RATE)
        wf.writeframes(b''.join(frames))

    print(f"Recording saved to {file_path}")
    return file_path
