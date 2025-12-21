"""
Presets Module

Manages configuration presets:
- JSON schema validation
- Save/load presets from files
- Default value management
- Backward compatibility with old preset versions
- Gradio UI integration
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import jsonschema
from jsonschema import validate, ValidationError


@dataclass
class DetectionParams:
    """Slide detection parameters"""

    frame_diff_threshold: float = 30.0  # 范围: 0-100，支持小数
    min_stable_frames: int = 3
    min_duration_sec: float = 2.0
    frame_interval: float = 1.0  # 每多少秒检测一帧 (1.0 = 每秒1帧, 5.0 = 每5秒1帧)
    use_ssim: bool = False
    use_histogram: bool = False
    ssim_threshold: float = 0.8
    histogram_threshold: float = 0.9


@dataclass
class ThumbnailParams:
    """Thumbnail generation parameters"""

    max_size: int = 800
    quality: int = 85
    dpi: int = 150


@dataclass
class SelectionParams:
    """Slide selection parameters"""

    default_selection: str = 'all'  # 'all' or 'new_only'
    multi_select: bool = True
    sort_by_page: bool = False
    show_statistics: bool = True


@dataclass
class OCRParams:
    """OCR parameters"""

    engine: str = 'pytesseract'  # 'pytesseract' or 'easyocr'
    roi_x: float = 70.0
    roi_y: float = 70.0
    roi_w: float = 25.0
    roi_h: float = 20.0
    confidence_threshold: float = 30.0
    language: str = 'eng'


@dataclass
class ExportParams:
    """Export parameters"""

    format: str = 'pdf'  # 'png', 'jpeg', 'pdf', 'both'
    image_format: str = 'jpeg'
    image_quality: int = 95
    pdf_page_size: str = 'A4'
    pdf_margin: float = 0.5
    naming: str = 'index'  # 'index', 'page', 'timestamp'
    output_dir: str = 'OUTPUT'


@dataclass
class Preset:
    """Complete preset configuration"""

    name: str
    version: str = "1.0.0"
    detection: DetectionParams = None
    thumbnail: ThumbnailParams = None
    selection: SelectionParams = None
    ocr: OCRParams = None
    export: ExportParams = None
    description: str = ""

    def __post_init__(self):
        """Initialize nested dataclasses if None"""
        if self.detection is None:
            self.detection = DetectionParams()
        if self.thumbnail is None:
            self.thumbnail = ThumbnailParams()
        if self.selection is None:
            self.selection = SelectionParams()
        if self.ocr is None:
            self.ocr = OCRParams()
        if self.export is None:
            self.export = ExportParams()


class TabPresetManager:
    """Manage per-tab presets separately"""

    def __init__(self, presets_dir: str):
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Tab-specific directories
        self.tab_dirs = {
            'detection': self.presets_dir / 'detection',
            'ocr': self.presets_dir / 'ocr',
            'export': self.presets_dir / 'export'
        }
        for d in self.tab_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # Default presets for each tab
        self._default_presets = {
            'detection': {
                'default': {
                    'name': '默认',
                    'frame_diff_threshold': 30.0,
                    'min_stable_frames': 3,
                    'min_duration_sec': 2.0,
                    'target_fps': 1.0,
                    'use_ssim': False,
                    'use_histogram': False
                },
                'high_quality': {
                    'name': '高质量',
                    'frame_diff_threshold': 25.0,
                    'min_stable_frames': 5,
                    'min_duration_sec': 3.0,
                    'target_fps': 1.0,
                    'use_ssim': False,
                    'use_histogram': False
                },
                'fast': {
                    'name': '快速',
                    'frame_diff_threshold': 40.0,
                    'min_stable_frames': 2,
                    'min_duration_sec': 1.0,
                    'target_fps': 2.0,
                    'use_ssim': False,
                    'use_histogram': False
                }
            },
            'ocr': {
                'default': {
                    'name': '默认',
                    'engine': 'pytesseract',
                    'roi_x': 70.0,
                    'roi_y': 70.0,
                    'roi_w': 25.0,
                    'roi_h': 20.0,
                    'confidence_threshold': 30.0
                },
                'bottom_right': {
                    'name': '右下角页码',
                    'engine': 'pytesseract',
                    'roi_x': 85.0,
                    'roi_y': 90.0,
                    'roi_w': 12.0,
                    'roi_h': 8.0,
                    'confidence_threshold': 30.0
                },
                'bottom_center': {
                    'name': '底部居中页码',
                    'engine': 'pytesseract',
                    'roi_x': 40.0,
                    'roi_y': 90.0,
                    'roi_w': 20.0,
                    'roi_h': 8.0,
                    'confidence_threshold': 30.0
                }
            },
            'export': {
                'default': {
                    'name': '默认',
                    'format': 'pdf',
                    'image_format': 'jpeg',
                    'image_quality': 95,
                    'naming': 'index'
                },
                'high_quality_images': {
                    'name': '高质量图片',
                    'format': 'jpeg',
                    'image_format': 'jpeg',
                    'image_quality': 100,
                    'naming': 'page'
                }
            }
        }

    def list_presets(self, tab: str) -> List[str]:
        """List presets for a specific tab"""
        presets = set(self._default_presets.get(tab, {}).keys())
        tab_dir = self.tab_dirs.get(tab)
        if tab_dir and tab_dir.exists():
            for f in tab_dir.glob("*.json"):
                presets.add(f.stem)
        return sorted(list(presets))

    def load_preset(self, tab: str, name: str) -> Optional[Dict]:
        """Load a preset for a specific tab"""
        # Try file first
        tab_dir = self.tab_dirs.get(tab)
        if tab_dir:
            preset_path = tab_dir / f"{self._sanitize_filename(name)}.json"
            if preset_path.exists():
                with open(preset_path, 'r', encoding='utf-8') as f:
                    return json.load(f)

        # Fall back to defaults
        return self._default_presets.get(tab, {}).get(name)

    def save_preset(self, tab: str, name: str, data: Dict, overwrite: bool = False) -> bool:
        """Save a preset for a specific tab"""
        tab_dir = self.tab_dirs.get(tab)
        if not tab_dir:
            return False

        preset_path = tab_dir / f"{self._sanitize_filename(name)}.json"
        if preset_path.exists() and not overwrite:
            self.logger.warning(f"Preset '{name}' already exists in {tab}")
            return False

        data['name'] = name
        with open(preset_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True

    def delete_preset(self, tab: str, name: str) -> bool:
        """Delete a custom preset"""
        if name in self._default_presets.get(tab, {}):
            return False  # Can't delete defaults

        tab_dir = self.tab_dirs.get(tab)
        if tab_dir:
            preset_path = tab_dir / f"{self._sanitize_filename(name)}.json"
            if preset_path.exists():
                preset_path.unlink()
                return True
        return False

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize filename"""
        safe_name = ''.join(c for c in name if c.isalnum() or c in (' ', '-', '_', '中', '文')).strip()
        safe_name = safe_name.replace(' ', '_')
        return safe_name if safe_name else 'unnamed'


class PresetManager:
    """Manage configuration presets"""

    # JSON schema for preset validation
    SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "version": {"type": "string"},
            "description": {"type": "string"},
            "detection": {
                "type": "object",
                "properties": {
                    "frame_diff_threshold": {"type": "number", "minimum": 0, "maximum": 255},
                    "min_stable_frames": {"type": "integer", "minimum": 1},
                    "min_duration_sec": {"type": "number", "minimum": 0},
                    "target_fps": {"type": "number", "minimum": 0.1, "maximum": 30},
                    "use_ssim": {"type": "boolean"},
                    "use_histogram": {"type": "boolean"},
                    "ssim_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                    "histogram_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["frame_diff_threshold", "min_stable_frames", "min_duration_sec"]
            },
            "thumbnail": {
                "type": "object",
                "properties": {
                    "max_size": {"type": "integer", "minimum": 100, "maximum": 2000},
                    "quality": {"type": "integer", "minimum": 1, "maximum": 100},
                    "dpi": {"type": "integer", "minimum": 72, "maximum": 600}
                },
                "required": ["max_size", "quality"]
            },
            "selection": {
                "type": "object",
                "properties": {
                    "default_selection": {"type": "string", "enum": ["all", "new_only"]},
                    "multi_select": {"type": "boolean"},
                    "sort_by_page": {"type": "boolean"},
                    "show_statistics": {"type": "boolean"}
                },
                "required": ["default_selection", "multi_select"]
            },
            "ocr": {
                "type": "object",
                "properties": {
                    "engine": {"type": "string", "enum": ["pytesseract", "easyocr"]},
                    "roi_x": {"type": "number", "minimum": 0, "maximum": 100},
                    "roi_y": {"type": "number", "minimum": 0, "maximum": 100},
                    "roi_w": {"type": "number", "minimum": 0, "maximum": 100},
                    "roi_h": {"type": "number", "minimum": 0, "maximum": 100},
                    "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 100},
                    "language": {"type": "string"}
                },
                "required": ["engine", "roi_x", "roi_y", "roi_w", "roi_h"]
            },
            "export": {
                "type": "object",
                "properties": {
                    "format": {"type": "string", "enum": ["png", "jpeg", "pdf", "both"]},
                    "image_format": {"type": "string", "enum": ["png", "jpeg"]},
                    "image_quality": {"type": "integer", "minimum": 1, "maximum": 100},
                    "pdf_page_size": {"type": "string", "enum": ["A4", "Letter", "A3", "A5"]},
                    "pdf_margin": {"type": "number", "minimum": 0, "maximum": 2},
                    "naming": {"type": "string", "enum": ["index", "page", "timestamp"]},
                    "output_dir": {"type": "string"}
                },
                "required": ["format", "output_dir"]
            }
        },
        "required": ["name", "detection", "thumbnail", "selection", "ocr", "export"]
    }

    def __init__(self, presets_dir: str):
        """
        Initialize preset manager

        Args:
            presets_dir: Directory to store preset files
        """
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Default presets
        self._default_presets = {
            'default': Preset(
                name='Default',
                description='Default configuration'
            ),
            'high_quality': Preset(
                name='High Quality',
                description='Higher quality settings for better results',
                detection=DetectionParams(
                    frame_diff_threshold=25.0,
                    min_stable_frames=5,
                    min_duration_sec=3.0
                ),
                thumbnail=ThumbnailParams(
                    max_size=1200,
                    quality=95
                )
            ),
            'fast': Preset(
                name='Fast',
                description='Faster processing with lower quality',
                detection=DetectionParams(
                    frame_diff_threshold=40.0,
                    min_stable_frames=2,
                    min_duration_sec=1.0,
                    frame_interval=0.5  # 每0.5秒检测1帧 (相当于2 FPS)
                ),
                thumbnail=ThumbnailParams(
                    max_size=600,
                    quality=70
                )
            ),
            'long_video': Preset(
                name='Long Video',
                description='Optimized for long videos (>30 minutes)',
                detection=DetectionParams(
                    frame_diff_threshold=35.0,
                    min_stable_frames=3,
                    min_duration_sec=3.0,
                    frame_interval=5.0  # 每5秒检测1帧，适合长视频
                ),
                thumbnail=ThumbnailParams(
                    max_size=800,
                    quality=85
                )
            )
        }

    def save_preset(self, preset: Preset, overwrite: bool = False) -> bool:
        """
        Save preset to file

        Args:
            preset: Preset to save
            overwrite: Whether to overwrite existing preset

        Returns:
            True if successful
        """
        try:
            # Validate preset
            self.validate_preset(preset)

            # Generate filename
            safe_name = self._sanitize_filename(preset.name)
            preset_path = self.presets_dir / f"{safe_name}.json"

            # Check if exists
            if preset_path.exists() and not overwrite:
                self.logger.warning(f"Preset '{preset.name}' already exists")
                return False

            # Convert to dictionary
            preset_dict = self.preset_to_dict(preset)

            # Save to file
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved preset: {preset.name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save preset '{preset.name}': {e}")
            return False

    def load_preset(self, name: str) -> Optional[Preset]:
        """
        Load preset from file

        Args:
            name: Preset name

        Returns:
            Loaded preset or None if failed
        """
        try:
            # Try loading from file
            safe_name = self._sanitize_filename(name)
            preset_path = self.presets_dir / f"{safe_name}.json"

            if preset_path.exists():
                with open(preset_path, 'r', encoding='utf-8') as f:
                    preset_dict = json.load(f)

                preset = self.dict_to_preset(preset_dict)
                self.logger.info(f"Loaded preset: {name}")
                return preset

            # Try default presets
            if name in self._default_presets:
                self.logger.info(f"Using default preset: {name}")
                return self._default_presets[name]

            self.logger.error(f"Preset not found: {name}")
            return None

        except Exception as e:
            self.logger.error(f"Failed to load preset '{name}': {e}")
            return None

    def list_presets(self) -> List[str]:
        """
        List all available presets

        Returns:
            List of preset names
        """
        presets = set(self._default_presets.keys())

        # Add file-based presets
        for preset_path in self.presets_dir.glob("*.json"):
            name = preset_path.stem
            presets.add(name)

        return sorted(list(presets))

    def delete_preset(self, name: str) -> bool:
        """
        Delete preset

        Args:
            name: Preset name

        Returns:
            True if successful
        """
        try:
            # Don't allow deleting default presets
            if name in self._default_presets:
                self.logger.error(f"Cannot delete default preset: {name}")
                return False

            # Delete file
            safe_name = self._sanitize_filename(name)
            preset_path = self.presets_dir / f"{safe_name}.json"

            if preset_path.exists():
                preset_path.unlink()
                self.logger.info(f"Deleted preset: {name}")
                return True
            else:
                self.logger.warning(f"Preset not found: {name}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to delete preset '{name}': {e}")
            return False

    def validate_preset(self, preset: Preset) -> bool:
        """
        Validate preset against schema

        Args:
            preset: Preset to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        preset_dict = self.preset_to_dict(preset)
        validate(instance=preset_dict, schema=self.SCHEMA)
        return True

    @staticmethod
    def preset_to_dict(preset: Preset) -> Dict[str, Any]:
        """Convert preset to dictionary"""
        return {
            'name': preset.name,
            'version': preset.version,
            'description': preset.description,
            'detection': asdict(preset.detection),
            'thumbnail': asdict(preset.thumbnail),
            'selection': asdict(preset.selection),
            'ocr': asdict(preset.ocr),
            'export': asdict(preset.export)
        }

    @staticmethod
    def dict_to_preset(data: Dict[str, Any]) -> Preset:
        """Convert dictionary to preset"""
        # Handle backward compatibility
        data = PresetManager._migrate_preset(data)

        return Preset(
            name=data['name'],
            version=data.get('version', '1.0.0'),
            description=data.get('description', ''),
            detection=DetectionParams(**data.get('detection', {})),
            thumbnail=ThumbnailParams(**data.get('thumbnail', {})),
            selection=SelectionParams(**data.get('selection', {})),
            ocr=OCRParams(**data.get('ocr', {})),
            export=ExportParams(**data.get('export', {}))
        )

    @staticmethod
    def _migrate_preset(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate old preset format to new format

        Args:
            data: Preset dictionary

        Returns:
            Migrated dictionary
        """
        # Handle version 0.x presets
        if 'version' not in data or data['version'].startswith('0.'):
            # Old format had different structure
            if 'params' in data:
                old_params = data['params']

                # Map old parameters to new structure
                if 'slide_detection' in old_params:
                    sd = old_params['slide_detection']
                    data.setdefault('detection', {
                        'frame_diff_threshold': sd.get('threshold', 30.0),
                        'min_stable_frames': sd.get('min_stable_frames', 3),
                        'min_duration_sec': sd.get('min_duration', 2.0),
                    })

                if 'thumbnail' in old_params:
                    th = old_params['thumbnail']
                    data.setdefault('thumbnail', {
                        'max_size': th.get('size', 800),
                        'quality': th.get('quality', 85),
                    })

                if 'ocr' in old_params:
                    ocr = old_params['ocr']
                    data.setdefault('ocr', {
                        'engine': ocr.get('engine', 'pytesseract'),
                        'roi_x': ocr.get('roi_x', 70.0),
                        'roi_y': ocr.get('roi_y', 70.0),
                        'roi_w': ocr.get('roi_w', 25.0),
                        'roi_h': ocr.get('roi_h', 20.0),
                    })

                if 'export' in old_params:
                    ex = old_params['export']
                    data.setdefault('export', {
                        'format': ex.get('format', 'pdf'),
                        'output_dir': ex.get('output_dir', 'OUTPUT'),
                    })

            # Set default values for missing sections
            data.setdefault('detection', {})
            data.setdefault('thumbnail', {})
            data.setdefault('selection', {
                'default_selection': 'all',
                'multi_select': True,
            })
            data.setdefault('ocr', {})
            data.setdefault('export', {})

            # Update version
            data['version'] = '1.0.0'

        return data

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize filename"""
        # Remove invalid characters
        safe_name = ''.join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        # Replace spaces with underscores
        safe_name = safe_name.replace(' ', '_')
        return safe_name

    def export_preset(self, name: str, output_path: str) -> bool:
        """
        Export preset to external file

        Args:
            name: Preset name
            output_path: Output file path

        Returns:
            True if successful
        """
        preset = self.load_preset(name)
        if preset is None:
            return False

        try:
            preset_dict = self.preset_to_dict(preset)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(preset_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Exported preset '{name}' to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export preset: {e}")
            return False

    def import_preset(self, input_path: str, name: Optional[str] = None) -> bool:
        """
        Import preset from external file

        Args:
            input_path: Input file path
            name: Optional new name for the preset

        Returns:
            True if successful
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                preset_dict = json.load(f)

            preset = self.dict_to_preset(preset_dict)

            # Rename if specified
            if name:
                preset.name = name

            # Save preset
            return self.save_preset(preset, overwrite=False)

        except Exception as e:
            self.logger.error(f"Failed to import preset: {e}")
            return False
