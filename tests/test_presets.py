"""Tests for presets module"""

import pytest
import tempfile
from pathlib import Path
import json

from src.presets import (
    PresetManager, Preset, DetectionParams, ThumbnailParams,
    SelectionParams, OCRParams, ExportParams
)


class TestPresetManager:
    """Test PresetManager class"""

    def test_list_presets(self):
        """Test listing presets"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PresetManager(tmpdir)
            presets = manager.list_presets()

            # Should include default presets
            assert 'default' in presets
            assert 'high_quality' in presets
            assert 'fast' in presets

    def test_save_and_load_preset(self):
        """Test saving and loading a preset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PresetManager(tmpdir)

            # Create a custom preset
            preset = Preset(
                name="test_preset",
                description="Test preset",
                detection=DetectionParams(
                    frame_diff_threshold=50.0,
                    min_stable_frames=5
                )
            )

            # Save
            success = manager.save_preset(preset)
            assert success

            # Load
            loaded = manager.load_preset("test_preset")
            assert loaded is not None
            assert loaded.name == "test_preset"
            assert loaded.description == "Test preset"
            assert loaded.detection.frame_diff_threshold == 50.0
            assert loaded.detection.min_stable_frames == 5

    def test_delete_preset(self):
        """Test deleting a preset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PresetManager(tmpdir)

            # Create and save a preset
            preset = Preset(name="to_delete")
            manager.save_preset(preset)

            # Verify it exists
            assert "to_delete" in manager.list_presets()

            # Delete
            success = manager.delete_preset("to_delete")
            assert success

            # Verify it's gone
            assert "to_delete" not in manager.list_presets()

    def test_validate_preset(self):
        """Test preset validation"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PresetManager(tmpdir)

            # Valid preset
            preset = Preset(
                name="test",
                detection=DetectionParams(
                    frame_diff_threshold=30.0,
                    min_stable_frames=3,
                    min_duration_sec=2.0
                )
            )

            # Should not raise
            manager.validate_preset(preset)

            # Invalid preset (missing required fields)
            invalid_data = {
                "name": "invalid",
                "detection": {}  # Missing required fields
            }

            # Should raise ValidationError
            with pytest.raises(Exception):
                manager.validate_preset(invalid_data)

    def test_preset_to_dict(self):
        """Test preset to dictionary conversion"""
        preset = Preset(
            name="test",
            version="1.0.0",
            description="Test",
            detection=DetectionParams(frame_diff_threshold=25.0),
            thumbnail=ThumbnailParams(max_size=1000),
            selection=SelectionParams(default_selection="new_only"),
            ocr=OCRParams(engine="easyocr"),
            export=ExportParams(format="pdf")
        )

        result = PresetManager.preset_to_dict(preset)

        assert result['name'] == "test"
        assert result['version'] == "1.0.0"
        assert result['description'] == "Test"
        assert result['detection']['frame_diff_threshold'] == 25.0
        assert result['thumbnail']['max_size'] == 1000
        assert result['selection']['default_selection'] == "new_only"
        assert result['ocr']['engine'] == "easyocr"
        assert result['export']['format'] == "pdf"

    def test_dict_to_preset(self):
        """Test dictionary to preset conversion"""
        data = {
            "name": "test",
            "version": "1.0.0",
            "description": "Test",
            "detection": {
                "frame_diff_threshold": 25.0,
                "min_stable_frames": 3,
                "min_duration_sec": 2.0
            },
            "thumbnail": {
                "max_size": 1000,
                "quality": 90
            },
            "selection": {
                "default_selection": "all",
                "multi_select": True
            },
            "ocr": {
                "engine": "pytesseract",
                "roi_x": 70.0,
                "roi_y": 70.0,
                "roi_w": 25.0,
                "roi_h": 20.0
            },
            "export": {
                "format": "pdf",
                "output_dir": "OUTPUT"
            }
        }

        preset = PresetManager.dict_to_preset(data)

        assert preset.name == "test"
        assert preset.version == "1.0.0"
        assert preset.description == "Test"
        assert preset.detection.frame_diff_threshold == 25.0
        assert preset.thumbnail.max_size == 1000
        assert preset.selection.default_selection == "all"
        assert preset.ocr.engine == "pytesseract"
        assert preset.export.format == "pdf"

    def test_export_and_import_preset(self):
        """Test exporting and importing presets"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PresetManager(tmpdir)

            # Create a preset
            preset = Preset(
                name="export_test",
                description="Export test",
                detection=DetectionParams(frame_diff_threshold=40.0)
            )

            manager.save_preset(preset)

            # Export to external file
            export_path = Path(tmpdir) / "exported_preset.json"
            success = manager.export_preset("export_test", str(export_path))
            assert success
            assert export_path.exists()

            # Delete original
            manager.delete_preset("export_test")
            assert "export_test" not in manager.list_presets()

            # Import from external file
            success = manager.import_preset(str(export_path))
            assert success
            assert "export_test" in manager.list_presets()

            # Verify imported preset
            loaded = manager.load_preset("export_test")
            assert loaded is not None
            assert loaded.detection.frame_diff_threshold == 40.0


class TestPreset:
    """Test Preset dataclass"""

    def test_create_preset_with_defaults(self):
        """Test creating preset with default values"""
        preset = Preset(name="test")

        assert preset.name == "test"
        assert preset.version == "1.0.0"
        assert preset.detection is not None
        assert preset.thumbnail is not None
        assert preset.selection is not None
        assert preset.ocr is not None
        assert preset.export is not None

    def test_create_preset_with_custom_values(self):
        """Test creating preset with custom values"""
        preset = Preset(
            name="custom",
            version="2.0.0",
            description="Custom preset",
            detection=DetectionParams(
                frame_diff_threshold=50.0,
                min_stable_frames=5,
                min_duration_sec=3.0
            ),
            thumbnail=ThumbnailParams(
                max_size=1200,
                quality=95
            )
        )

        assert preset.name == "custom"
        assert preset.version == "2.0.0"
        assert preset.description == "Custom preset"
        assert preset.detection.frame_diff_threshold == 50.0
        assert preset.thumbnail.max_size == 1200
        assert preset.thumbnail.quality == 95


if __name__ == "__main__":
    pytest.main([__file__])
