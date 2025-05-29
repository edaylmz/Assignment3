# Speech Recognition Assignment 3: DTW-based Digit Recognition

## Project Overview
This project implements a Dynamic Time Warping (DTW) based speech recognition system for isolated digit recognition (0-9). The system uses MFCC features and DTW with various configurations to achieve optimal recognition accuracy.

## Requirements

### Python Version
- Python 3.6 or higher

### Required Packages
```
numpy
scipy
matplotlib
librosa
pyaudio
```

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure
```
Assignment3/
├── dtw/
│   └── dtw.py           # DTW implementation with band-based pruning
├── features/
│   └── mfcc.py          # MFCC feature computation
├── recordings/          # Directory for storing recorded digits
├── logs/               # Directory for storing recognition logs
├── main.py             # Main program and testing interface
└── requirements.txt    # Project dependencies
```

## Features and Implementation Details

### 1. Audio Recording
- Uses PyAudio for real-time audio capture
- Implements silence detection using:
  - Amplitude threshold (default: 15)
  - Zero-crossing rate (ZCR) threshold (default: 2.0)
  - Maximum silence duration (default: 0.8s)
- Saves recordings as WAV files in the recordings directory

### 2. MFCC Feature Computation
- Computes 39-dimensional MFCC features:
  - 13 cepstral coefficients
  - 13 delta coefficients
  - 13 double-delta coefficients
- Uses librosa for efficient feature extraction
- Normalizes features for better recognition accuracy

### 3. DTW Implementation
- Implements standard DTW algorithm with band-based pruning
- Features:
  - Adaptive band width calculation based on sequence lengths
  - Feature normalization for improved matching
  - Time-synchronous DTW variant
  - Cost matrix visualization
- Band-based pruning parameters:
  - Base band ratio (default: 0.2)
  - Adaptive bandwidth factor (default: 0.15)
- Supports multiple templates per digit

### 4. Testing and Evaluation
The system performs comprehensive testing with:
1. Single template recognition
2. Time-synchronous DTW
3. Different band ratios (0.1 to 0.4)
4. Multiple template configurations (1 to 5 templates)

## How to Run

1. Setup the environment:
```bash
pip install -r requirements.txt
```

2. Run the program:
```bash
python Assignment3/main.py
```

3. Choose between:
   - Recording new digits
   - Using existing recordings

4. The program will:
   - Record or load digit samples
   - Perform recognition tests
   - Generate accuracy plots
   - Save detailed logs

## Testing Different Configurations

### 1. Band Ratio Testing
Tests recognition accuracy with different band ratios:
- Range: 0.1 to 0.4
- Results saved in 'band_ratio_results.png'
- Helps identify optimal pruning parameters

### 2. Template Count Testing
Tests recognition with different numbers of templates:
- Range: 1 to 5 templates per digit
- Results saved in 'template_results.png'
- Determines optimal template count

### 3. DTW Variants
Compares:
- Standard DTW
- Time-synchronous DTW
- Impact on recognition accuracy

## Results and Analysis

The program generates:
1. Recognition accuracy for each configuration
2. Plots showing:
   - Accuracy vs. Band Ratio
   - Accuracy vs. Template Count
3. DTW cost matrix visualizations
4. Detailed logs in the 'logs' directory
5. Summary of best performing configuration

## Troubleshooting

Common issues and solutions:

1. Microphone not working:
   - Check system audio settings
   - Verify PyAudio installation
   - Test with a simple audio recording program

2. Low recognition accuracy:
   - Ensure clear pronunciation
   - Check recording quality
   - Adjust silence detection parameters
   - Try different band ratios

3. Performance issues:
   - Reduce number of templates
   - Increase band ratio
   - Use time-synchronous DTW

4. Pruning not working:
   - Check band ratio values
   - Verify feature normalization
   - Monitor debug output for band widths
   - Try different adaptive bandwidth factors

