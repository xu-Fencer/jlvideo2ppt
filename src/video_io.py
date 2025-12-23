"""
Video IO Module

Handles video file operations including:
- ffprobe metadata extraction
- Frame iteration with configurable FPS
- File upload handling
- Output directory initialization
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterator, Union
import cv2
import numpy as np
from tqdm import tqdm


def parse_time_string(time_str: str) -> Optional[float]:
    """
    Parse time string in format "HH:MM:SS" or "MM:SS" or "SS" to seconds

    Args:
        time_str: Time string (e.g., "1:30:45", "30:45", "45")

    Returns:
        Time in seconds, or None if invalid
    """
    if not time_str or not time_str.strip():
        return None

    try:
        parts = time_str.strip().split(':')
        if len(parts) == 1:
            # Seconds only
            return float(parts[0])
        elif len(parts) == 2:
            # MM:SS
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            # HH:MM:SS
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        else:
            return None
    except (ValueError, IndexError):
        return None


def format_seconds_to_time(seconds: float) -> str:
    """
    Format seconds to "HH:MM:SS" string

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 0:
        seconds = 0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


class VideoMetadata:
    """Container for video metadata"""

    def __init__(self):
        self.duration: float = 0.0
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.codec: str = ""
        self.bitrate: int = 0
        self.frame_count: int = 0
        self.size_bytes: int = 0

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary"""
        return {
            "duration": self.duration,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "bitrate": self.bitrate,
            "frame_count": self.frame_count,
            "size_bytes": self.size_bytes,
        }


class VideoIO:
    """Video input/output operations"""

    # Supported video formats
    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}

    def __init__(self, video_path: Union[str, Path]):
        """
        Initialize VideoIO with video path

        Args:
            video_path: Path to video file or URL
        """
        self._video_path_str = str(video_path)
        self.video_path = video_path
        self._metadata: Optional[VideoMetadata] = None
        self._cap = None

    @property
    def video_path(self) -> Path:
        """Get video path"""
        return self._video_path

    @property
    def is_url(self) -> bool:
        """Check if the input is a URL"""
        return self._video_path_str.startswith(('http://', 'https://'))

    @property
    def video_path_str(self) -> str:
        """Get video path as string (works for both file paths and URLs)"""
        return self._video_path_str

    @video_path.setter
    def video_path(self, path: Union[str, Path]):
        """Set video path and validate"""
        self._video_path_str = str(path)

        # Check if input is a URL
        if str(path).startswith(('http://', 'https://')):
            # For URLs, skip file existence and format checks
            # Convert to Path for API compatibility, but don't use for file operations
            self._video_path = Path(str(path))
        else:
            # For local files, perform existing validation
            self._video_path = Path(path)
            if not self._video_path.exists():
                raise FileNotFoundError(f"Video file not found: {self._video_path}")

            ext = self._video_path.suffix.lower()
            if ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported video format: {ext}. "
                               f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}")

    def extract_metadata(self) -> VideoMetadata:
        """
        Extract video metadata using ffprobe

        Returns:
            VideoMetadata object
        """
        if self._metadata:
            return self._metadata

        metadata = VideoMetadata()

        # Get file size (only for local files, not URLs)
        if not self.is_url:
            metadata.size_bytes = self._video_path.stat().st_size

        # Try ffprobe first (more accurate)
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                self.video_path_str
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            # Parse format info
            if 'format' in data:
                format_info = data['format']
                metadata.duration = float(format_info.get('duration', 0))
                metadata.bitrate = int(format_info.get('bit_rate', 0))

            # Parse video stream
            if 'streams' in data:
                for stream in data['streams']:
                    if stream.get('codec_type') == 'video':
                        metadata.codec = stream.get('codec_name', '')
                        metadata.width = int(stream.get('width', 0))
                        metadata.height = int(stream.get('height', 0))

                        # Calculate FPS
                        fps_str = stream.get('r_frame_rate', '0/1')
                        if '/' in fps_str:
                            num, den = fps_str.split('/')
                            metadata.fps = float(num) / float(den) if float(den) != 0 else 0.0
                        else:
                            metadata.fps = float(fps_str)

                        metadata.frame_count = int(stream.get('nb_frames', 0))
                        break

        except (subprocess.CalledProcessError, FileNotFoundError, KeyError, ValueError) as e:
            print(f"Warning: ffprobe failed ({e}), falling back to OpenCV")

        # Fallback to OpenCV if ffprobe failed
        if metadata.fps == 0 or metadata.width == 0:
            cap = cv2.VideoCapture(self.video_path_str)
            if cap.isOpened():
                metadata.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                metadata.fps = cap.get(cv2.CAP_PROP_FPS)
                metadata.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                metadata.duration = metadata.frame_count / metadata.fps if metadata.fps > 0 else 0
                cap.release()

        self._metadata = metadata
        return metadata

    def get_frame_iter(self,
                      start_time: float = 0.0,
                      end_time: Optional[float] = None,
                      target_fps: float = 1.0) -> Iterator[Tuple[int, np.ndarray, float]]:
        """
        Get iterator over video frames (memory-efficient, optimized with frame seeking)

        Args:
            start_time: Start time in seconds (default: 0.0)
            end_time: End time in seconds (None = end of video)
            target_fps: Target FPS for sampling

        Yields:
            Tuple of (frame_number, frame_array, timestamp)
        """
        import gc

        cap = cv2.VideoCapture(self.video_path_str)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path_str}")

        try:
            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / original_fps if original_fps > 0 else 0

            if original_fps <= 0:
                raise RuntimeError("Invalid video FPS")

            # Validate and clamp time range
            start_time = max(0.0, start_time)
            if end_time is not None:
                end_time = min(end_time, video_duration)
            else:
                end_time = video_duration

            if start_time >= end_time:
                raise ValueError(f"Invalid time range: start_time ({start_time:.2f}s) >= end_time ({end_time:.2f}s)")

            # Calculate frame interval and total frames to process
            frame_interval = max(1, int(original_fps / target_fps))
            start_frame = int(start_time * original_fps)
            end_frame = int(end_time * original_fps)
            total_sample_frames = (end_frame - start_frame) // frame_interval + 1

            frame_number = start_frame
            gc_counter = 0

            with tqdm(total=total_sample_frames, desc="Processing frames", unit="frame") as pbar:
                while frame_number < end_frame:
                    # Seek directly to target frame (optimized for sparse sampling)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()

                    if not ret or frame is None:
                        break

                    timestamp = frame_number / original_fps
                    yield frame_number, frame, timestamp
                    pbar.update(1)
                    gc_counter += 1

                    # Periodic garbage collection every 100 frames
                    if gc_counter >= 100:
                        gc.collect()
                        gc_counter = 0

                    # Move to next sample frame
                    frame_number += frame_interval

        finally:
            cap.release()
            gc.collect()

    @staticmethod
    def handle_upload(uploaded_file: Union[str, Tuple, None],
                     output_dir: Union[str, Path]) -> Optional[Path]:
        """
        Handle Gradio file upload

        Args:
            uploaded_file: Gradio uploaded file (str path or tuple)
            output_dir: Directory to save uploaded file

        Returns:
            Path to saved file or None if failed
        """
        if uploaded_file is None:
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # If it's already a string path
        if isinstance(uploaded_file, str):
            src_path = Path(uploaded_file)
            if src_path.exists():
                dst_path = output_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                return dst_path

        # If it's a tuple from Gradio (filename, file data)
        if isinstance(uploaded_file, tuple) and len(uploaded_file) == 2:
            filename, file_data = uploaded_file
            dst_path = output_dir / filename

            # Write file data
            with open(dst_path, 'wb') as f:
                if isinstance(file_data, bytes):
                    f.write(file_data)
                else:
                    f.write(file_data.encode() if isinstance(file_data, str) else file_data)

            return dst_path

        return None

    @staticmethod
    def init_output_dirs(output_base_dir: Union[str, Path]) -> Dict[str, Path]:
        """
        Initialize output directory structure

        Args:
            output_base_dir: Base output directory

        Returns:
            Dictionary of directory paths
        """
        base = Path(output_base_dir)
        base.mkdir(parents=True, exist_ok=True)

        dirs = {
            'base': base,
            'tmp': base / 'tmp',
            'thumbs': base / 'thumbs',
            'images': base / 'images',
            'presets': base / 'presets',
            'logs': base / 'logs',
        }

        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        return dirs

    @staticmethod
    def cleanup_temp_dirs(output_base_dir: Union[str, Path]) -> None:
        """
        Clean up temporary directories

        Args:
            output_base_dir: Base output directory
        """
        base = Path(output_base_dir)
        tmp_dir = base / 'tmp'
        thumbs_dir = base / 'thumbs'

        for dir_path in [tmp_dir, thumbs_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                dir_path.mkdir(parents=True, exist_ok=True)
