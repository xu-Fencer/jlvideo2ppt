"""Integration tests for pipeline module"""

import pytest
import tempfile
from pathlib import Path

from src.pipeline import VideoProcessingPipeline
from src.presets import PresetManager


class TestVideoProcessingPipeline:
    """Test VideoProcessingPipeline class"""

    def test_init(self):
        """Test pipeline initialization"""
        pipeline = VideoProcessingPipeline()

        assert pipeline.preset_manager is not None
        assert pipeline.video_path is None
        assert pipeline.video_metadata is None
        assert len(pipeline.slide_frames) == 0
        assert len(pipeline.slides_with_thumbnails) == 0
        assert pipeline.current_preset is None

    def test_load_preset(self):
        """Test loading a preset"""
        pipeline = VideoProcessingPipeline()

        # Load default preset
        success = pipeline.load_preset("default")
        assert success
        assert pipeline.current_preset is not None
        assert pipeline.current_preset.name == "Default"  # Note capital D

        # Load high_quality preset
        success = pipeline.load_preset("high_quality")
        assert success
        assert pipeline.current_preset.name == "High Quality"  # Note spaces and capital letters

    def test_get_status(self):
        """Test getting pipeline status"""
        pipeline = VideoProcessingPipeline()

        status = pipeline.get_status()

        assert 'video_path' in status
        assert 'video_metadata' in status
        assert 'total_slides' in status
        assert 'slides_with_thumbnails' in status
        assert 'selected_slides' in status
        assert 'preset' in status
        assert 'output_dir' in status

    def test_reset_state(self):
        """Test resetting pipeline state"""
        pipeline = VideoProcessingPipeline()

        # Set some state
        pipeline.video_path = Path("/test/video.mp4")
        pipeline.slide_frames = [1, 2, 3]

        # Reset
        pipeline.reset_state()

        assert pipeline.video_path is None
        assert len(pipeline.slide_frames) == 0

    def test_cleanup(self):
        """Test cleanup"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = VideoProcessingPipeline()

            # Initialize output directories
            from src.video_io import VideoIO
            output_dirs = VideoIO.init_output_dirs(tmpdir)
            pipeline.output_dirs = output_dirs

            # Create some temp files
            (output_dirs['tmp'] / "test.txt").write_text("test")
            (output_dirs['thumbs'] / "test.jpg").write_bytes(b"test")

            # Cleanup
            pipeline.cleanup()

            # Directories should still exist but be empty
            assert output_dirs['tmp'].exists()
            assert output_dirs['thumbs'].exists()
            assert not any(output_dirs['tmp'].iterdir())
            assert not any(output_dirs['thumbs'].iterdir())


if __name__ == "__main__":
    pytest.main([__file__])
