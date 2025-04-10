import matplotlib.pyplot as plt
import librosa.display

def plot_features(log_mel, mfcc, sr=16000, title=""):
    """
    Display the log-mel spectrogram and MFCCs using matplotlib.
    """
    plt.figure(figsize=(10, 6))

    # Plot log-mel spectrogram
    plt.subplot(2, 1, 1)
    librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel', fmax=7000)
    plt.title(f"Log-Mel Spectrogram ({title})")
    plt.colorbar(format="%+2.0f dB")

    # Plot MFCCs
    plt.subplot(2, 1, 2)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.title(f"MFCCs ({title})")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
