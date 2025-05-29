import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import wave
import struct
import sys
import logging
from datetime import datetime
sys.path.append('.')  # Add current directory to Python path
from Assignment1 import record_audio
from dtw.dtw import DTW
from features.mfcc import MFCC

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"recognition_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def read_wav_file(filename: str) -> np.ndarray:
    """Read WAV file and return audio samples"""
    with wave.open(filename, 'rb') as wf:
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
        samples = struct.unpack(str(n_frames) + 'h', frames)
        return np.array(samples)

class DigitRecognizer:
    def __init__(self, logger):
        self.mfcc = MFCC()
        self.dtw = DTW()
        self.templates: Dict[str, List[np.ndarray]] = {}
        self.recordings: Dict[str, List[np.ndarray]] = {}
        self.logger = logger
        
    def load_existing_recordings(self):
        """Load existing recordings from the recordings directory"""
        recordings_dir = "recordings"
        if not os.path.exists(recordings_dir):
            self.logger.error(f"Recordings directory not found: {recordings_dir}")
            return False
            
        digits = ['zero', 'one', 'two', 'three', 'four', 
                 'five', 'six', 'seven', 'eight', 'nine']
        
        for digit in digits:
            self.recordings[digit] = []
            digit_files = [f for f in os.listdir(recordings_dir) if f.startswith(digit)]
            
            if not digit_files:
                self.logger.error(f"No recordings found for digit: {digit}")
                return False
                
            for file in digit_files:
                file_path = os.path.join(recordings_dir, file)
                audio = read_wav_file(file_path)
                features = self.mfcc.compute_features(audio)
                self.recordings[digit].append(features)
                self.logger.info(f"Loaded recording: {file}")
        
        return True
        
    def record_digits(self, n_instances: int = 10):
        """Record multiple instances of each digit"""
        digits = ['zero', 'one', 'two', 'three', 'four', 
                 'five', 'six', 'seven', 'eight', 'nine']
        
        self.logger.info("\n=== Recording Instructions ===")
        self.logger.info("1. You will be asked to record each digit multiple times")
        self.logger.info("2. When prompted, press Enter to start recording")
        self.logger.info("3. Say the digit clearly")
        self.logger.info("4. The recording will stop automatically after silence")
        self.logger.info("5. Wait for the next prompt\n")
        
        input("Press Enter to start recording process...")
        
        for digit in digits:
            self.recordings[digit] = []
            self.logger.info(f"\n=== Recording digit: '{digit}' ===")
            
            for i in range(n_instances):
                self.logger.info(f"\nRecording instance {i+1}/{n_instances}")
                self.logger.info(f"Say '{digit}' when ready...")
                
                output_file = f"Assignment3/recordings/{digit}_{i+1}.wav"
                os.makedirs("Assignment3/recordings", exist_ok=True)
                
                # Record audio with adjusted parameters
                record_audio(
                    output_filename=output_file,
                    rate=16000,
                    chunk_size=1024,
                    channels=1,
                    amplitude_threshold=15,
                    zcr_threshold=2.0,
                    max_silence_len=0.8
                )
                
                # Read the recorded audio and compute features
                audio = read_wav_file(output_file)
                features = self.mfcc.compute_features(audio)
                self.recordings[digit].append(features)
                
                self.logger.info(f"Recording {i+1} completed for '{digit}'")
    
    def setup_templates(self, n_templates: int = 1):
        """Setup templates from recordings"""
        self.templates = {}
        for digit, recordings in self.recordings.items():
            self.templates[digit] = recordings[:n_templates]
    
    def test_recognition(self, n_tests: int = 5, use_time_sync: bool = False, use_pruning: bool = False, band_ratio: float = None):
        """Test recognition accuracy"""
        # Create DTW instance with pruning if enabled
        if use_pruning and band_ratio is not None:
            self.dtw = DTW(band_ratio=band_ratio)
        else:
            self.dtw = DTW()
            
        digits = list(self.templates.keys())
        correct = 0
        total = 0
        
        self.logger.info("\n=== Testing Recognition ===")
        
        for digit in digits:
            self.logger.info(f"\nTesting digit '{digit}'")
            
            for i in range(n_tests):
                self.logger.info(f"\nTest {i+1}/{n_tests}")
                test_features = self.recordings[digit][i + 1]  # Use the next recording as test
                
                # Try recognition with each template
                min_dist = float('inf')
                recognized_digit = None
                
                for template_digit, templates in self.templates.items():
                    # Use recognize method which properly applies pruning
                    template_idx, dist = self.dtw.recognize(templates, test_features, use_time_sync)
                    
                    if dist < min_dist:
                        min_dist = dist
                        recognized_digit = template_digit
                
                if recognized_digit == digit:
                    correct += 1
                total += 1
                
                self.logger.info(f"Recognized as: {recognized_digit}")
                self.logger.info(f"Actual digit: {digit}")
        
        accuracy = correct / total * 100
        self.logger.info(f"\nOverall accuracy: {accuracy:.2f}%")
        return accuracy

def main():
    # Setup logging
    logger = setup_logging()
    recognizer = DigitRecognizer(logger)
    
    # Ask user whether to record new digits or use existing recordings
    while True:
        choice = input("\nDo you want to:\n1. Record new digits\n2. Use existing recordings\nEnter your choice (1 or 2): ")
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    if choice == '1':
        logger.info("=== Starting Recording Process ===")
        logger.info("You will need to record 10 instances of each digit (0-9)")
        logger.info("Total recordings needed: 100")
        input("Press Enter to start recording...")
        recognizer.record_digits(n_instances=10)
    else:
        logger.info("=== Loading Existing Recordings ===")
        if not recognizer.load_existing_recordings():
            logger.error("Failed to load existing recordings. Exiting...")
            return
    
    # Test with single template
    logger.info("\n=== Testing with Single Template ===")
    recognizer.setup_templates(n_templates=1)
    accuracy_single = recognizer.test_recognition(n_tests=5, use_time_sync=False)
    
    # Test with time-synchronous DTW
    logger.info("\n=== Testing with Time-Synchronous DTW ===")
    accuracy_time_sync = recognizer.test_recognition(n_tests=5, use_time_sync=True)
    
    # Test with different band ratios
    logger.info("\n=== Testing with Different Band Ratios ===")
    band_ratios = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    accuracies_band = []
    
    for band_ratio in band_ratios:
        logger.info(f"\nTesting with band ratio: {band_ratio}")
        accuracy = recognizer.test_recognition(
            n_tests=5, 
            use_time_sync=True,
            use_pruning=True,
            band_ratio=band_ratio
        )
        accuracies_band.append(accuracy)
    
    # Plot band ratio results
    plt.figure(figsize=(10, 6))
    plt.plot(band_ratios, accuracies_band, 'bo-')
    plt.xlabel('Band Ratio')
    plt.ylabel('Recognition Accuracy (%)')
    plt.title('Recognition Accuracy vs Band Ratio')
    plt.grid(True)
    plt.savefig('band_ratio_results.png')
    plt.close()
    
    # Test with different numbers of templates
    logger.info("\n=== Testing with Different Numbers of Templates ===")
    template_counts = range(1, 6)  # 1 to 5 templates
    accuracies_templates = []
    
    for n_templates in template_counts:
        logger.info(f"\nTesting with {n_templates} template(s)...")
        recognizer.setup_templates(n_templates=n_templates)
        accuracy = recognizer.test_recognition(n_tests=5, use_time_sync=True)
        accuracies_templates.append(accuracy)
    
    # Plot template count results
    plt.figure(figsize=(10, 6))
    plt.plot(template_counts, accuracies_templates, 'ro-')
    plt.xlabel('Number of Templates')
    plt.ylabel('Recognition Accuracy (%)')
    plt.title('Recognition Accuracy vs Number of Templates')
    plt.grid(True)
    plt.savefig('template_results.png')
    plt.close()
    
    # Plot DTW cost matrices for a sample comparison
    logger.info("\n=== Generating DTW Cost Matrix Visualizations ===")
    sample_digit = list(recognizer.templates.keys())[0]
    sample_template = recognizer.templates[sample_digit][0]
    sample_test = recognizer.recordings[sample_digit][1]
    
    dtw = DTW(band_ratio=0.2)
    dtw.compute_distance(sample_template, sample_test, plot_matrix=True)
    dtw.time_synchronous_dtw(sample_template, sample_test, plot_matrix=True)
    
    logger.info("\n=== Results Summary ===")
    logger.info("\n1. Basic Recognition Tests:")
    logger.info(f"   - Single template accuracy: {accuracy_single:.2f}%")
    logger.info(f"   - Time-synchronous DTW accuracy: {accuracy_time_sync:.2f}%")
    
    logger.info("\n2. Band Ratio Analysis:")
    for ratio, acc in zip(band_ratios, accuracies_band):
        logger.info(f"   - Band ratio {ratio:.2f}: {acc:.2f}% accuracy")
    best_band_ratio = band_ratios[np.argmax(accuracies_band)]
    logger.info(f"   - Best performing band ratio: {best_band_ratio:.2f} ({max(accuracies_band):.2f}% accuracy)")
    
    logger.info("\n3. Template Count Analysis:")
    for count, acc in zip(template_counts, accuracies_templates):
        logger.info(f"   - {count} template(s): {acc:.2f}% accuracy")
    best_template_count = template_counts[np.argmax(accuracies_templates)]
    logger.info(f"   - Best performing template count: {best_template_count} ({max(accuracies_templates):.2f}% accuracy)")
    
    logger.info("\n4. Overall Best Configuration:")
    logger.info(f"   - Best accuracy achieved: {max(max(accuracies_band), max(accuracies_templates)):.2f}%")
    logger.info(f"   - Recommended settings:")
    logger.info(f"     * Band ratio: {best_band_ratio:.2f}")
    logger.info(f"     * Number of templates: {best_template_count}")
    logger.info(f"     * Time-synchronous DTW: {'Yes' if accuracy_time_sync > accuracy_single else 'No'}")
    
    logger.info("\n5. Visualization Files:")
    logger.info("   - Band ratio results: 'band_ratio_results.png'")
    logger.info("   - Template count results: 'template_results.png'")
    logger.info("   - DTW cost matrices: 'dtw_cost_matrix.png' and 'time_synchronous_dtw_cost_matrix.png'")

if __name__ == "__main__":
    main() 