# Speech Recognition Assignment 3
This project implements a Dynamic Time Warping (DTW) based speech recognition system for isolated digit recognition. The system records spoken digits, computes MFCC features, and uses DTW for template matching.

This work builds upon the implementations developed in Assignment 1 and Assignment 2. Assignment 3 introduces a DTW-based isolated digit recognition system that reuses the audio recording and endpointing functionality from Assignment 1, as well as the MFCC + delta + delta-delta feature extraction pipeline from Assignment 2. All new code specific to this assignment is located in the Assignment3 folder. Running main.py in that folder initiates the system, where the user is prompted to either record new digit samples or use existing recordings. All logs and results summary are under the logs folder and plots are in Assignment3. 
## Project Structure

```
Assignment3/
├── dtw/
│   └── dtw.py           # DTW implementation
├── features/
│   └── mfcc.py          # MFCC feature computation
├── main.py              # Main interface and testing
└── recordings/          # Directory for recorded audio files
```

## Features and Implementation Details

### 1. Audio Recording (`Assignment1.py`)
- **Function**: `record_audio()`
- **Parameters**:
  - `amplitude_threshold`: Detects silence (default: 80)
  - `zcr_threshold`: Zero-crossing rate threshold (default: 0.1)
  - `max_silence_len`: Silence duration to stop (default: 0.3s)
- **Usage**: Records audio with automatic endpoint detection

### 2. MFCC Feature Computation (`features/mfcc.py`)
- **Class**: `MFCC`
- **Features**:
  - 39-dimensional feature vectors (13 MFCC + deltas + double deltas)
  - Configurable parameters:
    - Sample rate: 16kHz
    - Mel filters: 40
    - Frequency range: 50Hz-7000Hz
- **Methods**:
  - `compute_features()`: Computes full feature vector
  - `_compute_mfcc()`: Computes basic MFCC features
  - `_compute_deltas()`: Computes delta features

### 3. DTW Implementation (`dtw/dtw.py`)
- **Class**: `DTW`
- **Features**:
  - Standard DTW
  - Time-synchronous DTW
  - Pruning support
- **Methods**:
  - `compute_distance()`: Standard DTW
  - `time_synchronous_dtw()`: Time-synchronous DTW
  - `recognize()`: Template matching

### 4. Main Interface (`main.py`)
- **Class**: `DigitRecognizer`
- **Features**:
  - Recording multiple instances of digits
  - Template setup and testing
  - Accuracy evaluation
  - Visualization of results
- **Methods**:
  - `record_digits()`: Records multiple instances
  - `setup_templates()`: Prepares templates
  - `test_recognition()`: Tests recognition accuracy

## How to Run

1. **Setup**:
   ```bash
   cd Assignment3
   
   # Install dependencies
   pip install -r requirements.txt
   
   ```

2. **Running the Program**:
   ```bash
   python main.py
   ```

3. **Recording Process**:
   - The program will prompt you to record 10 instances of each digit (0-9)
   - For each recording:
     1. Press Enter when ready
     2. Say the digit clearly
     3. Wait for automatic stop (after 0.3s of silence)
     4. Repeat for all instances

4. **Testing Process**:
   The program automatically runs several tests:
   - Single template recognition
   - Time-synchronous DTW
   - Different pruning thresholds
   - Different numbers of templates

5. **Results**:
   - Recognition accuracies are printed to console
   - Two plots are generated:
     - `pruning_results.png`: Accuracy vs pruning threshold
     - `template_results.png`: Accuracy vs number of templates
