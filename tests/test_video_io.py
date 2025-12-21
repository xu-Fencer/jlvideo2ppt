"""Tests for video_io module"""

import pytest
from pathlib import Path
import tempfile
import numpy as np
import cv2

from src.video_io import VideoIO, VideoMetadata


class TestVideoMetadata:
    """Test VideoMetadata class"""

    def test_to_dict(self):
        """Test metadata to dictionary conversion"""
        metadata = VideoMetadata()
        metadata.duration = 10.5
        metadata.width = 1920
        metadata.height = 1080
        metadata.fps = 30.0

        result = metadata.to_dict()
        assert result['duration'] == 10.5
        assert result['width'] == 1920
        assert result['height'] == 1080
        assert result['fps'] == 30.0


class TestVideoIO:
    """Test VideoIO class"""

    def test_init_output_dirs(self):
        """Test output directory initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"
            dirs = VideoIO.init_output_dirs(output_dir)

            assert dirs['base'].exists()
            assert dirs['tmp'].exists()
            assert dirs['thumbs'].exists()
            assert dirs['presets'].exists()
            assert dirs['logs'].exists()

    def test_cleanup_temp_dirs(self):
        """Test temporary directory cleanup"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"
            VideoIO.init_output_dirs(output_dir)

            # Create some files
            (output_dir / "tmp" / "test.txt").write_text("test")
            (output_dir / "thumbs" / "test.jpg").write_bytes(b"test")

            # Cleanup
            VideoIO.cleanup_temp_dirs(output_dir)

            # Check directories exist but are empty
            assert (output_dir / "tmp").exists()
            assert (output_dir / "thumbs").exists()
            assert not any((output_dir / "tmp").iterdir())
            assert not any((output_dir / "thumbs").iterdir())


@pytest.fixture
def test_video_path():
    """Create a test video file"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name

    # Create a simple test video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, 30.0, (640, 480))

    # Write 30 frames
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (i * 8, 0, 0)  # Blue gradient
        cv2.putText(frame, f"Frame {i}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        out.write(frame)

    out.release()

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestVideoIOWithFile:
    """Test VideoIO with actual video file"""

    def test_extract_metadata(self, test_video_path):
        """Test metadata extraction"""
        video_io = VideoIO(test_video_path)
        metadata = video_io.extract_metadata()

        assert metadata.width == 640
        assert metadata.height == 480
        assert metadata.fps == 30.0
        assert metadata.frame_count == 30
        assert metadata.duration == 1.0  # 30 frames / 30 fps

    def test_get_frame_iter(self, test_video_path):
        """Test frame iteration"""
        video_io = VideoIO(test_video_path)

        frames = list(video_io.get_frame_iter(target_fps=10.0))

        assert len(frames) > 0
        assert len(frames) <= 10  # Should be approximately 10 frames

        # Check frame structure
        frame_number, frame, timestamp = frames[0]
        assert isinstance(frame_number, int)
        assert isinstance(frame, np.ndarray)
        assert isinstance(timestamp, float)
        assert frame.shape == (480, 640, 3)


if __name__ == "__main__":
    pytest.main([__file__])
