import numpy as np
from scipy.fftpack import dct
from typing import Tuple
import librosa

class MFCC:
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_filters: int = 40,
                 n_ceps: int = 13,
                 low_freq: int = 50,
                 high_freq: int = 7000):
        """
        Initialize MFCC computation parameters
        
        Args:
            sample_rate: Audio sample rate
            n_filters: Number of Mel filters
            n_ceps: Number of cepstral coefficients
            low_freq: Lower frequency bound
            high_freq: Upper frequency bound
        """
        self.sample_rate = sample_rate
        self.n_filters = n_filters
        self.n_ceps = n_ceps
        self.low_freq = low_freq
        self.high_freq = high_freq
        
    def compute_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute 39-dimensional MFCC features (13 cepstra + deltas + double deltas)
        
        Args:
            audio: Input audio signal
            
        Returns:
            39-dimensional feature vectors
        """
        # Compute basic MFCC features
        mfcc = self._compute_mfcc(audio)
        
        # Compute deltas
        delta = self._compute_deltas(mfcc)
        
        # Compute double deltas
        delta2 = self._compute_deltas(delta)
        
        # Concatenate features
        features = np.hstack((mfcc, delta, delta2))
        
        return features
    
    def _compute_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Compute basic MFCC features"""
        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Framing
        frame_length = int(0.025 * self.sample_rate)  # 25ms
        frame_step = int(0.010 * self.sample_rate)    # 10ms
        frames = librosa.util.frame(emphasized_audio, frame_length=frame_length, hop_length=frame_step)
        
        # Windowing
        window = np.hamming(frame_length)
        windowed_frames = frames * window[:, np.newaxis]
        
        # FFT
        fft_frames = np.fft.rfft(windowed_frames, axis=0)
        power_frames = np.abs(fft_frames) ** 2
        
        # Mel filterbank
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=frame_length,
            n_mels=self.n_filters,
            fmin=self.low_freq,
            fmax=self.high_freq
        )
        mel_features = np.dot(mel_basis, power_frames)
        
        # Log
        log_mel_features = np.log(mel_features + 1e-10)
        
        # DCT
        mfcc = dct(log_mel_features, type=2, axis=0, norm='ortho')[:self.n_ceps]
        
        return mfcc.T
    
    def _compute_deltas(self, features: np.ndarray, N: int = 2) -> np.ndarray:
        """
        Compute delta features
        
        Args:
            features: Input features
            N: Number of frames to use for delta computation
            
        Returns:
            Delta features
        """
        n_frames = len(features)
        deltas = np.zeros_like(features)
        
        for t in range(n_frames):
            numerator = 0
            denominator = 0
            
            for n in range(1, N + 1):
                if t + n < n_frames:
                    numerator += n * (features[t + n] - features[t - n])
                    denominator += n * n
            
            if denominator != 0:
                deltas[t] = numerator / (2 * denominator)
        
        return deltas 