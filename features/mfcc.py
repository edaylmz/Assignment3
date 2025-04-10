import librosa
from config import NUM_MFCC, FREQ_MIN, FREQ_MAX

def compute_log_mel(y, sr, n_mels, fmin=FREQ_MIN, fmax=FREQ_MAX):
    """
    Compute log-mel spectrogram from audio waveform.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=400,
        hop_length=160,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    log_mel = librosa.power_to_db(mel_spec)
    return log_mel


def compute_mfcc(file_path, n_mels=40, n_mfcc=NUM_MFCC, fmin=FREQ_MIN, fmax=FREQ_MAX):
    """
    Load audio from file and compute MFCCs using a log-mel spectrogram.
    Applies trimming to remove silence before feature extraction.
    """
    y, sr = librosa.load(file_path, sr=16000) # ← Force-load at 16kHz
    y, _ = librosa.effects.trim(y)  # Trim silence

    log_mel = compute_log_mel(y, sr, n_mels=n_mels, fmin=fmin, fmax=fmax) # This is the "log spectra"
    mfcc = librosa.feature.mfcc(S=log_mel, sr=sr, n_mfcc=n_mfcc) #  This is the cepstrum

    return log_mel, mfcc
