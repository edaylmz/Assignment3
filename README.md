# MFCC Speech Recognition Tool
This project is built for **Assignment 2: Data Capture and Feature Computation**. It allows you to record spoken digits, extract MFCC features, and visualize them using a GUI. The system includes endpoint detection, log-mel spectral analysis, and optional noise addition.

# Assignment Requirements & Implementation

Requirement                                                                  | Implemented In                                        |
|----------------------------------------------------------------------------|--------------------------------------------------------|
| 16kHz sampling rate                                                        | `config.py` → `SAMPLING_RATE = 16000`                  |
| 16-bit PCM audio                                                           | `recorder.py` → `pyaudio.paInt16`                      |
| Automatic endpoint detection                                               | `recorder.py` → Amplitude + Zero-Crossing Rate logic   |
| Compute **log spectra**                                                    | `mfcc.py` → `power_to_db(mel_spectrogram)`             |
| Compute **cepstra** (MFCCs)                                                | `mfcc.py` → `librosa.feature.mfcc(...)`                |
| Use **40 Mel filters**, 50Hz–7000Hz                                        | `mfcc.py` → `n_mels=40, fmin=50, fmax=7000`            |
| Use **first 13 DCT coefficients (MFCC)**                                   | `mfcc.py` → `n_mfcc=13`                                |
| Compare different filter settings (40, 30, 25)                             | `main.py` and GUI dropdown                             |
| Visualize both spectrally (log-mel) and cepstrally (MFCC)                  | `plot.py`, and GUI embedded plots                      |
| Record data with noise                                                     | `utils.py → add_noise()`, GUI checkbox                 |
| Show degradation patterns under noise                                      | GUI visualization with/without noise                   |
| UI for controlling the workflow                                            | `interface.py` with Tkinter                            |
| Export MFCCs as `.csv` for analysis                                        | `interface.py` → `pandas.DataFrame.to_csv()`           |
| Replay the last recorded audio                                             | `interface.py` with `simpleaudio`                      |

---
## Code Requirements
pip install -r requirements.txt

## How the Code Runs

### 1. Launch the app:
   python main.py

### 2. The GUI will open:
  Select a digit (e.g., “zero”)
  Choose number of Mel filters (40 / 30 / 25)
  Optional: Check the Apply Background Noise box
  Press Record & Analyze

### 3. The system:
  Records your voice with endpoint detection
  Computes log-mel spectrogram and 13 MFCCs
  Shows them in the GUI
  Saves the MFCCs as .csv in /recordings  
  Allows replay of the latest audio

## Output Examples

### MFCCs of "zero" and "one" with 40 Mel Filters and No Aditional Noise
<img src="https://github.com/user-attachments/assets/4669a789-0b16-44cc-92aa-ab09f91f301b" width="500" alt="Resized Image">
<img src="https://github.com/user-attachments/assets/f0e60c60-ab57-4bff-ab58-c3c3a93795a5" width="500" alt="Resized Image">

### Log-Mel Spectrogram of digit "one" with/without Noise
<img src="https://github.com/user-attachments/assets/7c93f77a-efa2-4330-9d47-0119cdd59af9" width="500" alt="Resized Image">
<img src="https://github.com/user-attachments/assets/a1ae6f6a-3382-4de9-993c-ae4788b109ec" width="500" alt="Resized Image">

### Same Digit "seven" 40 Mel Filters No Additional Noise, Different Samples — Pattern Similarity
<img src="https://github.com/user-attachments/assets/4bf656c4-7160-4d3d-a16d-2d13481f1fbf" width="500" alt="Resized Image">
<img src="https://github.com/user-attachments/assets/434e09e4-f98e-4126-9340-14188a94a2ef" width="500" alt="Resized Image">

### Same Digit "nine" Log-Mel Spectrogram Different Filters with/without Noise
<img src="https://github.com/user-attachments/assets/4ed32364-806c-4a95-b420-c92403fce2a9" width="1000" alt="Resized Image">

### Same Digit "nine" MFCC Different Filters with/without Noise
<img src="https://github.com/user-attachments/assets/d3ce1f8c-cc46-4a63-a12e-6ce23a0e59aa" width="1000" alt="Resized Image">
