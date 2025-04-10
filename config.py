# config.py

# Audio settings
SAMPLING_RATE = 16000 #16 kHz sampling
CHANNELS = 1
CHUNK_SIZE = 1024
SAMPLE_WIDTH = 2  # 16-bit PCM (2 bytes)

# Endpoint detection
SILENCE_THRESHOLD = 400     # Amplitude threshold
ZCR_THRESHOLD = 0.5         # Zero-crossing rate threshold
MAX_SILENCE_DURATION = 3.0  # seconds

# MFCC computation
MEL_FILTERS = [40, 30, 25]  # Number of Mel filters
NUM_MFCC = 13 #NO OF FEATURES
FREQ_MIN = 50 #MEL SPECTRAL FILTER MIN FREQUENCY
FREQ_MAX = 7000 #MEL SPECTRAL FILTER MAX FREQUENCY

# Folder paths
RECORDINGS_DIR = "recordings"
