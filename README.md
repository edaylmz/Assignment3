# Speech Recognition Assignment 3: DTW-based Digit Recognition

This project implements a Dynamic Time Warping (DTW) based speech recognition system for isolated digit recognition. The system records spoken digits, computes MFCC features, and uses DTW for template matching.

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

## Requirements

1. Python 3.6 or higher
2. Required packages (install using `pip install -r requirements.txt`):
   - numpy
   - scipy
   - matplotlib
   - librosa
   - pyaudio

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
   # Create required directories
   mkdir -p Assignment3/recordings
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Running the Program**:
   ```bash
   python Assignment3/main.py
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

## Testing Different Configurations

1. **Adjusting Recording Parameters**:
   Modify in `main.py`:
   ```python
   record_audio(
       amplitude_threshold=80,  # Adjust silence detection
       zcr_threshold=0.1,      # Adjust ZCR threshold
       max_silence_len=0.3     # Adjust silence duration
   )
   ```

2. **Testing Different MFCC Configurations**:
   Modify in `features/mfcc.py`:
   ```python
   mfcc = MFCC(
       n_filters=40,    # Number of Mel filters
       n_ceps=13,       # Number of cepstral coefficients
       low_freq=50,     # Lower frequency bound
       high_freq=7000   # Upper frequency bound
   )
   ```

3. **Testing DTW Parameters**:
   Modify in `dtw/dtw.py`:
   ```python
   dtw = DTW(pruning_threshold=0.2)  # Adjust pruning threshold
   ```

## Expected Results

1. **Recognition Accuracy**:
   - Single template: ~70-80%
   - Time-synchronous DTW: ~80-90%
   - Multiple templates: Up to ~95%

2. **Performance Metrics**:
   - Processing time per digit: ~100-200ms
   - Memory usage: ~100MB for 100 recordings

## Troubleshooting

1. **Recording Issues**:
   - If recording doesn't stop: Decrease `max_silence_len`
   - If recording stops too early: Increase `amplitude_threshold`
   - If background noise affects: Increase `zcr_threshold`

2. **Recognition Issues**:
   - Low accuracy: Try increasing number of templates
   - Slow performance: Enable pruning
   - False positives: Use time-synchronous DTW

## Contributing

Feel free to modify and improve the code. Key areas for improvement:
1. Feature extraction optimization
2. DTW algorithm efficiency
3. Template selection strategies
4. Noise robustness

## License

This project is part of a speech recognition course assignment. Use for educational purposes only.
