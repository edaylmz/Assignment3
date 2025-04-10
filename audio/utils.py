import numpy as np
def zero_crossing_rate(samples):
    """
    Computes the Zero Crossing Rate (ZCR) for a list of 16-bit PCM samples.
    ZCR = (number of sign changes) / (total samples - 1)
    """
    if len(samples) < 2:
        return 0.0

    zero_crossings = 0
    for i in range(1, len(samples)):
        if (samples[i - 1] >= 0 and samples[i] < 0) or (samples[i - 1] < 0 and samples[i] >= 0):
            zero_crossings += 1

    return zero_crossings / (len(samples) - 1)

def add_noise(audio, noise_level=0.005):
    """
    Add Gaussian noise to a numpy array of audio samples.
    """
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise