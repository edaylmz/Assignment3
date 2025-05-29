import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

class DTW:
    def __init__(self, band_ratio: float = 0.2, alpha: float = 0.15):
        """
        Initialize DTW with adaptive band pruning
        
        Args:
            band_ratio: Base band ratio for pruning (default: 0.2)
            alpha: Adaptive bandwidth factor (default: 0.15)
        """
        self.band_ratio = band_ratio
        self.alpha = alpha
        
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to reduce noise and improve matching
        
        Args:
            features: Input feature sequence
            
        Returns:
            Normalized feature sequence
        """
        # Normalize each feature dimension
        return (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
    def compute_adaptive_band(self, len1: int, len2: int) -> int:
        """
        Compute adaptive bandwidth based on sequence lengths
        
        Args:
            len1: Length of first sequence
            len2: Length of second sequence
            
        Returns:
            Band width in frames
        """
        # Base band from ratio
        base_band = int(max(len1, len2) * self.band_ratio)
        
        # Adaptive component based on length difference
        adaptive_band = int(abs(len1 - len2) + self.alpha * min(len1, len2))
        
        # Use the larger of the two bands
        return max(base_band, adaptive_band)
        
    def compute_distance(self, template: np.ndarray, test: np.ndarray, plot_matrix: bool = False) -> Tuple[float, np.ndarray]:
        """
        Compute DTW distance between template and test sequences with adaptive pruning
        
        Args:
            template: Template feature sequence (N x 39)
            test: Test feature sequence (M x 39)
            plot_matrix: Whether to plot the cost matrix
            
        Returns:
            Tuple of (distance, accumulated cost matrix)
        """
        # Normalize features
        template_norm = self.normalize_features(template)
        test_norm = self.normalize_features(test)
        
        N, M = len(template_norm), len(test_norm)
        cost_matrix = np.full((N + 1, M + 1), np.inf)
        cost_matrix[0, 0] = 0
        
        # Compute adaptive band width
        band = self.compute_adaptive_band(N, M)
        
        # Compute cost matrix with adaptive band
        for i in range(1, N + 1):
            # Calculate valid j range based on adaptive band
            j_start = max(1, i - band)
            j_end = min(M + 1, i + band + 1)
            
            for j in range(j_start, j_end):
                # Compute Euclidean distance between normalized feature vectors
                dist = np.linalg.norm(template_norm[i-1] - test_norm[j-1])
                
                # Update cost matrix
                cost_matrix[i, j] = dist + min(
                    cost_matrix[i-1, j],    # insertion
                    cost_matrix[i, j-1],    # deletion
                    cost_matrix[i-1, j-1]   # match
                )
        
        # Plot cost matrix if requested
        if plot_matrix:
            self.plot_cost_matrix(cost_matrix, "DTW Cost Matrix")
        
        return cost_matrix[N, M], cost_matrix
    
    def time_synchronous_dtw(self, template: np.ndarray, test: np.ndarray, plot_matrix: bool = False) -> Tuple[float, np.ndarray]:
        """
        Compute time-synchronous DTW distance with adaptive pruning
        
        Args:
            template: Template feature sequence (N x 39)
            test: Test feature sequence (M x 39)
            plot_matrix: Whether to plot the cost matrix
            
        Returns:
            Tuple of (distance, accumulated cost matrix)
        """
        # Normalize features
        template_norm = self.normalize_features(template)
        test_norm = self.normalize_features(test)
        
        N, M = len(template_norm), len(test_norm)
        cost_matrix = np.full((N + 1, M + 1), np.inf)
        cost_matrix[0, 0] = 0
        
        # Compute adaptive band width
        band = self.compute_adaptive_band(N, M)
        
        # Compute cost matrix with time-sync constraints and adaptive band
        for i in range(1, N + 1):
            # Calculate valid j range based on band and time-sync constraints
            j_start = max(1, max(i - 1, i - band))
            j_end = min(M + 1, min(i + 2, i + band + 1))
            
            for j in range(j_start, j_end):
                dist = np.linalg.norm(template_norm[i-1] - test_norm[j-1])
                cost_matrix[i, j] = dist + min(
                    cost_matrix[i-1, j],    # insertion
                    cost_matrix[i, j-1],    # deletion
                    cost_matrix[i-1, j-1]   # match
                )
        
        # Plot cost matrix if requested
        if plot_matrix:
            self.plot_cost_matrix(cost_matrix, "Time-Synchronous DTW Cost Matrix")
        
        return cost_matrix[N, M], cost_matrix
    
    def plot_cost_matrix(self, cost_matrix: np.ndarray, title: str):
        """
        Plot DTW cost matrix for visualization
        
        Args:
            cost_matrix: Accumulated cost matrix
            title: Plot title
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(cost_matrix[1:, 1:], origin='lower', cmap='hot', interpolation='nearest')
        plt.title(title)
        plt.xlabel("Template Frame")
        plt.ylabel("Test Frame")
        plt.colorbar()
        plt.savefig(f"{title.lower().replace(' ', '_')}.png")
        plt.close()

    def recognize(self, templates: List[np.ndarray], test: np.ndarray, use_time_sync: bool = False) -> Tuple[int, float]:
        """
        Recognize test sequence using multiple templates with pruning
        
        Args:
            templates: List of template feature sequences
            test: Test feature sequence
            use_time_sync: Whether to use time-synchronous DTW
            
        Returns:
            Tuple of (template_index, distance)
        """
        distances = []
        min_dist = float('inf')
        best_template_idx = -1
        
        # Normalize test sequence once
        test_norm = self.normalize_features(test)
        
        for idx, template in enumerate(templates):
            # Normalize template
            template_norm = self.normalize_features(template)
            
            # Compute adaptive band width for this template
            band = self.compute_adaptive_band(len(template_norm), len(test_norm))
            
            # Apply pruning by computing cost matrix with band constraints
            if use_time_sync:
                dist, cost_matrix = self.time_synchronous_dtw(template_norm, test_norm)
            else:
                dist, cost_matrix = self.compute_distance(template_norm, test_norm)
            
            # Check if this template gives better match
            if dist < min_dist:
                min_dist = dist
                best_template_idx = idx
            
            distances.append(dist)
            
            # Debug: Print band width and distance for each template
            print(f"Template {idx}: Band width = {band}, Distance = {dist:.2f}")
        
        return best_template_idx, min_dist 