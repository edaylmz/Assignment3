import pyaudio
import wave
import struct
import sys


def zero_crossing_rate(samples):
    """
    Compute the zero crossing rate for the given list/array of
    integer audio samples (16-bit).

    ZCR = (number of sign changes) / (total samples - 1)

    """
    zc_count = 0
    # Loop through samples, count sign changes
    for i in range(1, len(samples)):
        current_sign = (samples[i] >= 0)
        previous_sign = (samples[i - 1] >= 0)
        if current_sign != previous_sign:
            zc_count += 1

    # Avoid division by zero if len(samples) < 2
    if len(samples) < 2:
        return 0.0

    return zc_count / (len(samples) - 1)


def record_audio(
        output_filename="output.wav",
        rate=16000,
        chunk_size=1024,
        channels=1, #1=mono, 2=stereo
        amplitude_threshold=1,  # Below this is considered "quiet"
        zcr_threshold=0.1,  # Below this is considered "low Zero-Crossing Rate"
        max_silence_len=0.3
):
    """
    Records from the microphone after pressing Enter, and stops once
    we've detected 'max_silence_len' seconds of consecutive silence
    (silence = amplitude < amplitude_threshold AND zcr < zcr_threshold).

    """

    import pyaudio
    p = pyaudio.PyAudio()

    # Open the microphone stream
    stream = p.open(
        format=pyaudio.paInt16,  # 16-bit audio
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk_size
    )

    print("Press Enter to start recording...")
    sys.stdin.readline()
    print("Recording started. Speak into the microphone...")

    frames = []
    silent_chunk_count = 0

    # Convert 'max_silence_len' from seconds -> chunk count
    # e.g. if chunk_size=1024, rate=16000 => 0.064 seconds/chunk
    silent_chunk_thresh = int(max_silence_len * (rate / chunk_size))

    while True:
        data = stream.read(chunk_size)
        frames.append(data)

        # Convert raw bytes to 16-bit samples
        samples = struct.unpack(str(len(data) // 2) + 'h', data)

        # Compute average amplitude (short-term energy) & ZCR
        avg_amplitude = sum(abs(x) for x in samples) / len(samples)
        zcr_value = zero_crossing_rate(samples)

        print(f"Amplitude: {avg_amplitude:.2f}, ZCR: {zcr_value:.3f}")

        # Determine if this chunk is "silence" or "speech"
        # We'll say it's "silence" only if BOTH amplitude < amp_thresh AND zcr < zcr_thresh
        if avg_amplitude < amplitude_threshold and zcr_value < zcr_threshold:
            silent_chunk_count += 1
            print(f"-> Silence count = {silent_chunk_count}")
        else:
            silent_chunk_count = 0
            print("-> Speech detected; resetting silence count")

        # If we've been in silence for enough chunks, end recording
        if silent_chunk_count > silent_chunk_thresh:
            print("Detected sufficient silence. Stopping...")
            break

    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Write out the frames to a WAV file
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Recording saved to {output_filename}")


if __name__ == "__main__":
    record_audio(
        output_filename="speech_output.wav",
        rate=16000,
        chunk_size=1024,
        channels=1,
        amplitude_threshold=100,
        zcr_threshold=0.15,  # Typical speech might have ZCR ~0.05-0.2, in my environment 0.15 works best
        max_silence_len=2.0  # 2 seconds of silence
    )
