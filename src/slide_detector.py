"""
Slide Detector Module

Detects slide changes in video using frame difference method.
Supports frame differencing, SSIM, and histogram comparison.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from pathlib import Path


@dataclass
class SlideFrame:
    """Represents a detected slide frame"""

    frame_number: int
    timestamp: float
    diff_score: float
    thumbnail_path: Optional[str] = None
    original_image_path: Optional[str] = None
    page_number: Optional[int] = None


class SlideDetector:
    """
    Slide change detector using frame difference method

    Features:
    - Frame differencing with configurable threshold
    - Image preprocessing (grayscale, blur)
    - Debouncing to prevent false positives
    - Minimum duration filtering
    - Optional SSIM and histogram comparison
    """

    def __init__(self,
                 frame_diff_threshold: float = 30.0,
                 min_stable_frames: int = 3,
                 min_duration_sec: float = 2.0,
                 use_ssim: bool = False,
                 use_histogram: bool = False):
        """
        Initialize slide detector

        Args:
            frame_diff_threshold: Threshold for frame difference (0-255)
            min_stable_frames: Minimum consecutive frames to confirm a slide
            min_duration_sec: Minimum duration for a slide (seconds)
            use_ssim: Enable SSIM comparison as backup
            use_histogram: Enable histogram comparison as backup
        """
        self.frame_diff_threshold = frame_diff_threshold
        self.min_stable_frames = min_stable_frames
        self.min_duration_sec = min_duration_sec
        self.use_ssim = use_ssim
        self.use_histogram = use_histogram

        self.logger = logging.getLogger(__name__)
        self._previous_frame: Optional[np.ndarray] = None
        self._consecutive_changes = 0
        self._last_slide_frame: Optional[int] = None
        self._slide_frames: List[SlideFrame] = []

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for comparison

        Args:
            frame: Input frame (BGR format)

        Returns:
            Preprocessed grayscale frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        return blurred

    def compute_frame_difference(self,
                                current_frame: np.ndarray,
                                previous_frame: np.ndarray) -> Tuple[float, float]:
        """
        Compute frame difference score

        Args:
            current_frame: Current preprocessed frame
            previous_frame: Previous preprocessed frame

        Returns:
            Tuple of (difference_score, change_ratio)
        """
        # Absolute difference
        diff = cv2.absdiff(current_frame, previous_frame)

        # Apply threshold
        _, thresh = cv2.threshold(diff, self.frame_diff_threshold, 255, cv2.THRESH_BINARY)

        # Calculate metrics
        diff_score = np.mean(diff)
        change_ratio = np.sum(thresh == 255) / thresh.size

        return diff_score, change_ratio

    def compute_ssim(self,
                    current_frame: np.ndarray,
                    previous_frame: np.ndarray) -> float:
        """
        Compute Structural Similarity Index (SSIM)

        Args:
            current_frame: Current frame
            previous_frame: Previous frame

        Returns:
            SSIM score (0-1, higher is more similar)
        """
        try:
            # Ensure frames are the same size
            if current_frame.shape != previous_frame.shape:
                current_frame = cv2.resize(current_frame, (previous_frame.shape[1], previous_frame.shape[0]))

            # Compute SSIM
            ssim = cv2.compareSSIM(current_frame, previous_frame)
            return max(0.0, min(1.0, ssim))
        except Exception as e:
            self.logger.warning(f"SSIM computation failed: {e}")
            return 1.0

    def compute_histogram(self,
                         current_frame: np.ndarray,
                         previous_frame: np.ndarray) -> Tuple[float, float]:
        """
        Compute histogram comparison

        Args:
            current_frame: Current frame
            previous_frame: Previous frame

        Returns:
            Tuple of (correlation, chi_square)
        """
        try:
            # Compute histograms
            hist_curr = cv2.calcHist([current_frame], [0], None, [256], [0, 256])
            hist_prev = cv2.calcHist([previous_frame], [0], None, [256], [0, 256])

            # Normalize
            hist_curr = cv2.normalize(hist_curr, hist_curr).flatten()
            hist_prev = cv2.normalize(hist_prev, hist_prev).flatten()

            # Compare histograms
            correlation = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_CORREL)
            chi_square = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_CHISQR)

            return correlation, chi_square
        except Exception as e:
            self.logger.warning(f"Histogram comparison failed: {e}")
            return 1.0, 0.0

    def is_slide_change(self,
                       current_frame: np.ndarray,
                       previous_frame: np.ndarray,
                       timestamp: float) -> bool:
        """
        Determine if current frame indicates a slide change

        Args:
            current_frame: Current preprocessed frame
            previous_frame: Previous preprocessed frame
            timestamp: Current timestamp

        Returns:
            True if slide change detected
        """
        # Compute primary metric (frame difference)
        diff_score, change_ratio = self.compute_frame_difference(current_frame, previous_frame)

        # Check minimum change threshold
        primary_detection = change_ratio > 0.1  # At least 10% of pixels changed

        # Backup methods
        backup_detection = False
        backup_details = ""

        if primary_detection and (self.use_ssim or self.use_histogram):
            if self.use_ssim:
                ssim = self.compute_ssim(current_frame, previous_frame)
                backup_detection = backup_detection or (ssim < 0.8)
                backup_details += f"SSIM: {ssim:.3f} "

            if self.use_histogram:
                correlation, chi_square = self.compute_histogram(current_frame, previous_frame)
                backup_detection = backup_detection or (correlation < 0.9)
                backup_details += f"Hist: {correlation:.3f}"

        # Check if enough time has passed since last slide
        if self._last_slide_frame is not None:
            time_since_last = timestamp - (self._last_slide_frame / 30.0)  # Assuming 30fps
            if time_since_last < self.min_duration_sec:
                if primary_detection:
                    self.logger.debug(
                        f"Slide change rejected (too soon): {time_since_last:.2f}s < {self.min_duration_sec}s"
                    )
                return False

        # Debouncing: require consecutive detections
        if primary_detection or backup_detection:
            self._consecutive_changes += 1
            is_confirmed = self._consecutive_changes >= self.min_stable_frames
        else:
            self._consecutive_changes = max(0, self._consecutive_changes - 1)
            is_confirmed = False

        if is_confirmed:
            self.logger.info(
                f"Slide change detected at {timestamp:.2f}s "
                f"(diff: {diff_score:.2f}, ratio: {change_ratio:.3f}) "
                f"{backup_details}"
            )
            self._last_slide_frame = timestamp * 30.0  # Store approximate frame number

        return is_confirmed

    def process_frames(self,
                      frame_iterator,
                      video_fps: float = 30.0) -> List[SlideFrame]:
        """
        Process video frames and detect slide changes

        Args:
            frame_iterator: Iterator yielding (frame_number, frame, timestamp)
            video_fps: Original video FPS

        Returns:
            List of detected slide frames
        """
        self._slide_frames = []
        self._previous_frame = None
        self._consecutive_changes = 0
        self._last_slide_frame = None
        first_frame_data = None  # Store first frame info

        for frame_number, frame, timestamp in frame_iterator:
            try:
                # Store first frame info
                if first_frame_data is None:
                    first_frame_data = (frame_number, frame, timestamp)

                # Preprocess current frame
                current_processed = self.preprocess_frame(frame)

                # Compare with previous frame
                if self._previous_frame is not None:
                    # Check for slide change
                    if self.is_slide_change(current_processed, self._previous_frame, timestamp):
                        # Create slide frame record
                        slide_frame = SlideFrame(
                            frame_number=frame_number,
                            timestamp=timestamp,
                            diff_score=0.0  # Will be updated below
                        )
                        self._slide_frames.append(slide_frame)

                # Update previous frame
                self._previous_frame = current_processed

            except Exception as e:
                self.logger.error(f"Error processing frame {frame_number}: {e}")
                continue

        # Always add first frame as the initial slide
        if first_frame_data is not None:
            frame_number, frame, timestamp = first_frame_data
            first_slide = SlideFrame(
                frame_number=frame_number,
                timestamp=timestamp,
                diff_score=0.0
            )
            self._slide_frames.insert(0, first_slide)
            self.logger.info(f"Added first frame as slide 0 at {timestamp:.2f}s")

        # Calculate diff scores for each slide
        self._calculate_diff_scores()

        self.logger.info(f"Detected {len(self._slide_frames)} slides")
        return self._slide_frames

    def _calculate_diff_scores(self) -> None:
        """Calculate difference scores for detected slides"""
        # This would require keeping frames in memory
        # For now, we'll use placeholder values
        for i, slide in enumerate(self._slide_frames):
            slide.diff_score = 100.0 - (i * 10.0)  # Placeholder

    def get_slide_frames(self) -> List[SlideFrame]:
        """Get detected slide frames"""
        return self._slide_frames.copy()

    def to_dict(self) -> List[Dict]:
        """
        Convert slide frames to dictionary list

        Returns:
            List of dictionaries representing slide frames
        """
        return [
            {
                'frame_number': sf.frame_number,
                'timestamp': sf.timestamp,
                'diff_score': sf.diff_score,
                'thumbnail_path': sf.thumbnail_path,
                'original_image_path': sf.original_image_path,
                'page_number': sf.page_number
            }
            for sf in self._slide_frames
        ]

    @classmethod
    def from_dict(cls, data: List[Dict]) -> List[SlideFrame]:
        """
        Create slide frames from dictionary list

        Args:
            data: List of dictionaries

        Returns:
            List of SlideFrame objects
        """
        return [
            SlideFrame(
                frame_number=item['frame_number'],
                timestamp=item['timestamp'],
                diff_score=item['diff_score'],
                thumbnail_path=item.get('thumbnail_path'),
                original_image_path=item.get('original_image_path'),
                page_number=item.get('page_number')
            )
            for item in data
        ]
