"""
Pipeline Module

Main pipeline that orchestrates all processing steps:
- Video parsing and slide detection
- Thumbnail generation
- OCR page number recognition
- Export to various formats
- Gradio UI integration
- CLI interface
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import traceback

from .video_io import VideoIO, VideoMetadata
from .slide_detector import SlideDetector, SlideFrame
from .thumbs import ThumbnailGenerator
from .selection import SlideSelection
from .page_number import PageNumberRecognizer, PageNumberSorter, ROIRect
from .exporter import Exporter
from .presets import PresetManager, Preset, TabPresetManager
from dataclasses import asdict
from pathlib import Path as PathLib

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class VideoProcessingPipeline:
    """Main processing pipeline for video to slides conversion"""

    def __init__(self, preset_manager: Optional[PresetManager] = None):
        """
        Initialize pipeline

        Args:
            preset_manager: Preset manager instance
        """
        self.preset_manager = preset_manager or PresetManager('OUTPUT/presets')
        self.logger = self._setup_logger()
        self.reset_state()

    def _setup_logger(self) -> logging.Logger:
        """Setup pipeline logger"""
        logger = logging.getLogger('pipeline')
        logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        return logger

    def reset_state(self):
        """Reset pipeline state"""
        self.video_path: Optional[Path] = None
        self.video_metadata: Optional[VideoMetadata] = None
        self.slide_frames: List[SlideFrame] = []
        self.slides_with_thumbnails: List[Dict] = []
        self.selection: SlideSelection = SlideSelection()
        self.output_dirs: Dict[str, Path] = {}
        self.current_preset: Optional[Preset] = None
        self.processing_log: List[str] = []
        self._non_duplicate_slides: List[Dict] = []  # Slides without duplicate page numbers
        self._stop_processing = False  # Flag to stop processing

    def check_stop(self) -> bool:
        """
        Check if processing should be stopped

        Returns:
            True if processing should be stopped
        """
        return self._stop_processing

    def stop_processing(self):
        """Signal to stop processing"""
        self.logger.info("Stop signal received. Processing will terminate after current step.")
        self._stop_processing = True

    def continue_processing(self):
        """Reset stop flag to continue processing"""
        self._stop_processing = False

    def load_preset(self, preset_name: str) -> bool:
        """
        Load configuration preset

        Args:
            preset_name: Name of preset to load

        Returns:
            True if successful
        """
        preset = self.preset_manager.load_preset(preset_name)
        if preset:
            self.current_preset = preset
            self.logger.info(f"Loaded preset: {preset_name}")
            return True
        return False

    def parse_video(self,
                   video_path: str,
                   output_dir: str,
                   preset_name: str = 'default',
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> Tuple[bool, str, Optional[Dict]]:
        """
        Parse video and detect slides

        Args:
            video_path: Path to video file or URL
            output_dir: Output directory
            preset_name: Preset to use
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)

        Returns:
            Tuple of (success, message, result_dict)
        """
        try:
            # Reset state
            self.reset_state()
            self.continue_processing()  # Ensure processing flag is not set

            # Load preset
            if not self.load_preset(preset_name):
                return False, f"Failed to load preset: {preset_name}", None

            self.video_path = Path(video_path)
            self.output_dirs = VideoIO.init_output_dirs(output_dir)

            # Initialize video IO
            self.logger.info(f"Opening video: {video_path}")
            video_io = VideoIO(video_path)

            # Extract metadata
            self.video_metadata = video_io.extract_metadata()
            self.logger.info(
                f"Video metadata: {self.video_metadata.width}x{self.video_metadata.height}, "
                f"{self.video_metadata.fps:.2f}fps, {self.video_metadata.duration:.2f}s"
            )

            # Check if stopped
            if self.check_stop():
                return False, "Processing cancelled by user", None

            # Detect slides
            self.logger.info("Detecting slides...")
            detector = SlideDetector(
                frame_diff_threshold=self.current_preset.detection.frame_diff_threshold,
                min_stable_frames=self.current_preset.detection.min_stable_frames,
                min_duration_sec=self.current_preset.detection.min_duration_sec,
                use_ssim=self.current_preset.detection.use_ssim,
                use_histogram=self.current_preset.detection.use_histogram
            )

            # Get frame iterator
            frame_iter = video_io.get_frame_iter(
                start_time=start_time if start_time is not None else 0.0,
                end_time=end_time,
                target_fps=1.0 / self.current_preset.detection.frame_interval
            )

            # Process frames
            self.slide_frames = detector.process_frames(frame_iter, self.video_metadata.fps)

            # Check if stopped after slide detection
            if self.check_stop():
                return False, "Processing cancelled by user", None

            self.logger.info(f"Detected {len(self.slide_frames)} slides")

            # Generate thumbnails
            self.logger.info("Generating thumbnails...")
            thumb_gen = ThumbnailGenerator(
                thumbs_dir=self.output_dirs['thumbs'],
                original_dir=self.output_dirs['images'],
                max_size=self.current_preset.thumbnail.max_size,
                quality=self.current_preset.thumbnail.quality,
                max_workers=4  # Use 4 threads for parallel I/O
            )

            # Convert to dictionaries for thumbnail generation
            slide_dicts = [asdict(sf) for sf in self.slide_frames]

            # Extract frame positions for thumbnails (optimized with direct seeking)
            frame_positions = [sf.frame_number for sf in self.slide_frames]

            # Generate thumbnails using direct frame seeking (much faster than sequential reading)
            # Use video_io's video_path_str for proper URL handling
            self.slides_with_thumbnails = thumb_gen.generate_batch_parallel(
                slide_dicts, video_io.video_path_str, frame_positions
            )

            # Check if stopped after thumbnail generation
            if self.check_stop():
                return False, "Processing cancelled by user", None

            # Update selection
            self.selection.set_slides(self.slides_with_thumbnails)

            # Return result
            result = {
                'metadata': self.video_metadata.to_dict(),
                'slides': self.slides_with_thumbnails,
                'output_dir': str(self.output_dirs['base'])
            }

            return True, f"Successfully processed video: {len(self.slide_frames)} slides detected", result

        except Exception as e:
            error_msg = f"Failed to parse video: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return False, error_msg, None

    def recognize_page_numbers(self,
                              slides: Optional[List[Dict]] = None) -> Tuple[bool, str, List[Dict]]:
        """
        Recognize page numbers in slides

        Args:
            slides: Slides to process (uses pipeline slides if None)

        Returns:
            Tuple of (success, message, updated_slides)
        """
        try:
            if slides is None:
                slides = self.slides_with_thumbnails

            if not slides:
                return False, "No slides available for page number recognition", []

            self.logger.info("Recognizing page numbers...")

            # Initialize recognizer
            roi = ROIRect(
                x=self.current_preset.ocr.roi_x,
                y=self.current_preset.ocr.roi_y,
                w=self.current_preset.ocr.roi_w,
                h=self.current_preset.ocr.roi_h
            )

            recognizer = PageNumberRecognizer(
                ocr_engine=self.current_preset.ocr.engine,
                roi=roi,
                max_workers=4  # Use 4 processes for parallel OCR
            )

            # Load images (use only original high-res images)
            images = []
            for slide in slides:
                # 只使用original_image_path（原图），不使用缩略图
                img_path = slide.get('original_image_path')
                if img_path and Path(img_path).exists():
                    import cv2
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
                    else:
                        self.logger.warning(f"Could not load image: {img_path}")
                        images.append(None)
                else:
                    self.logger.warning(f"Original image not found: {img_path}")
                    images.append(None)

            # Recognize page numbers (use parallel processing by default)
            updated_slides = recognizer.update_slides_with_page_numbers(slides, images, use_parallel=True)

            # Sort by page number
            sorter = PageNumberSorter()
            sorted_slides = sorter.sort_slides(updated_slides)

            # Update pipeline state
            self.slides_with_thumbnails = sorted_slides
            self.selection.set_slides(sorted_slides)

            # Count successful recognitions
            recognized_count = sum(1 for s in sorted_slides if s.get('page_number') is not None)

            message = f"Page number recognition complete: {recognized_count}/{len(slides)} slides recognized"

            return True, message, sorted_slides

        except Exception as e:
            error_msg = f"Failed to recognize page numbers: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return False, error_msg, slides

    def export_slides(self,
                     slides: Optional[List[Dict]] = None,
                     selection_only: bool = False) -> Tuple[bool, str, Optional[Dict]]:
        """
        Export slides to files

        Args:
            slides: Slides to export (uses pipeline slides if None)
            selection_only: Export only selected slides

        Returns:
            Tuple of (success, message, export_results)
        """
        try:
            if slides is None:
                slides = self.slides_with_thumbnails

            if not slides:
                return False, "No slides available for export", None

            # Get slides to export
            if selection_only:
                slides_to_export = self.selection.get_selected_slides()
                if not slides_to_export:
                    return False, "No slides selected for export", None
            else:
                slides_to_export = slides

            self.logger.info(f"Exporting {len(slides_to_export)} slides...")

            # Initialize exporter
            # Use the current output directory (with timestamp) instead of preset default
            if self.output_dirs and 'base' in self.output_dirs:
                output_dir = str(self.output_dirs['base'])
            else:
                # Fallback to preset default if no output_dirs set
                output_dir = self.current_preset.export.output_dir

            exporter = Exporter(output_dir)

            # Export
            export_format = self.current_preset.export.format

            if export_format == 'pdf':
                pdf_path = exporter.export_pdf(
                    slides_to_export,
                    page_size=self.current_preset.export.pdf_page_size,
                    margin=self.current_preset.export.pdf_margin,
                    naming=self.current_preset.export.naming
                )
                results = {'pdf': [pdf_path]}

            elif export_format in ['png', 'jpeg']:
                image_paths = exporter.export_images(
                    slides_to_export,
                    format=export_format,
                    quality=self.current_preset.export.image_quality,
                    naming=self.current_preset.export.naming
                )
                results = {'images': image_paths}

            elif export_format == 'both':
                results = exporter.export_both(
                    slides_to_export,
                    image_format=self.current_preset.export.image_format,
                    image_quality=self.current_preset.export.image_quality,
                    pdf_page_size=self.current_preset.export.pdf_page_size,
                    pdf_margin=self.current_preset.export.pdf_margin,
                    naming=self.current_preset.export.naming
                )

            else:
                return False, f"Unsupported export format: {export_format}", None

            # Get summary
            summary = exporter.get_export_summary(results)

            self.logger.info(f"Export complete: {summary}")

            return True, summary, results

        except Exception as e:
            error_msg = f"Failed to export slides: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return False, error_msg, None

    def get_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status

        Returns:
            Status dictionary
        """
        return {
            'video_path': str(self.video_path) if self.video_path else None,
            'video_metadata': self.video_metadata.to_dict() if self.video_metadata else None,
            'total_slides': len(self.slide_frames),
            'slides_with_thumbnails': len(self.slides_with_thumbnails),
            'selected_slides': len(self.selection.selected_indices),
            'preset': self.current_preset.name if self.current_preset else None,
            'output_dir': str(self.output_dirs.get('base')) if self.output_dirs else None,
            'selection_stats': self.selection.get_statistics()
        }

    def cleanup(self):
        """Clean up temporary files"""
        if self.output_dirs and 'base' in self.output_dirs:
            try:
                VideoIO.cleanup_temp_dirs(self.output_dirs['base'])
                self.logger.info("Cleaned up temporary files")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temporary files: {e}")


def create_gradio_interface(pipeline: VideoProcessingPipeline):
    """Create Gradio interface"""
    import gradio as gr

    with gr.Blocks(title="JL Video to PPT Converter") as demo:
        gr.Markdown("# JL Video to PPT Converter")
        gr.Markdown("Convert video presentations to slide images/PDF")

        # State
        preset_manager = pipeline.preset_manager
        tab_preset_manager = TabPresetManager('OUTPUT/presets')
        current_slides = gr.State(value=[])
        selected_slides_for_export = gr.State(value=[])
        selected_slides_for_ocr = gr.State(value=[])
        selection_source = gr.State(value="")  # Track data source: "selection" or "ocr"

        # Tab 1: Video Parsing
        with gr.Tab("1. 解析视频", id="tab_parse"):
            gr.Markdown("## 上传视频并配置参数")

            with gr.Row():
                video_input = gr.File(
                    label="上传视频文件",
                    file_types=['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
                )
                video_path_input = gr.Textbox(
                    label="或输入本地文件路径/URL",
                    placeholder="/path/to/video.mp4 或 http://example.com/video.mp4",
                    info="支持本地文件路径或HTTP/HTTPS URL"
                )

            with gr.Row():
                output_dir = gr.Textbox(
                    label="输出目录",
                    value="OUTPUT",
                    info="默认为视频目录下的OUTPUT文件夹"
                )

            # Preset section for detection
            with gr.Row():
                detection_preset_dropdown = gr.Dropdown(
                    label="检测预设",
                    choices=tab_preset_manager.list_presets('detection'),
                    value='default',
                    scale=2
                )
                load_detection_preset_btn = gr.Button("加载预设", scale=1)
                detection_preset_name = gr.Textbox(
                    label="预设名称",
                    placeholder="输入名称保存当前参数",
                    scale=2
                )
                save_detection_preset_btn = gr.Button("保存为预设", scale=1)

            # Collapsible parameters section
            with gr.Accordion("高级检测参数", open=False):
                with gr.Row():
                    frame_diff_threshold = gr.Number(
                        label="帧差阈值 (0-100)",
                        value=30.0,
                        info="越小越敏感，范围: 0-100，支持小数点"
                    )
                    min_stable_frames = gr.Number(
                        label="最小稳定帧数",
                        value=3,
                        info="确认幻灯片变化所需的稳定帧数",
                        minimum=1
                    )
                with gr.Row():
                    min_duration_sec = gr.Number(
                        label="最小持续时间(秒)",
                        value=2.0,
                        info="幻灯片最小显示时长",
                        minimum=0.5
                    )
                    frame_interval = gr.Number(
                        label="每多少秒检测一帧",
                        value=1.0,
                        info="长视频建议: 1-5秒 (例: 5.0 = 每5秒检测1帧)",
                        minimum=0.5,
                        maximum=30.0,
                        step=0.5
                    )
                with gr.Row():
                    use_ssim = gr.Checkbox(label="使用SSIM算法", value=False)
                    use_histogram = gr.Checkbox(label="使用直方图算法", value=False)

            # Time range section (not part of preset)
            with gr.Accordion("检测范围 (可选)", open=False):
                gr.Markdown("设置检测的时间范围，支持格式: 时:分:秒 (如: 1:30:45, 30:45, 45)")

                with gr.Row():
                    start_time_input = gr.Textbox(
                        label="起始时间",
                        placeholder="例: 1:30:45 或 30:45",
                        info="留空表示从视频开始检测"
                    )
                    end_time_input = gr.Textbox(
                        label="终止时间",
                        placeholder="例: 10:00:00 或 5:00",
                        info="留空表示检测到视频结束"
                    )

                with gr.Row():
                    clear_time_btn = gr.Button("清除时间范围", variant="secondary")

            with gr.Row():
                parse_btn = gr.Button("开始解析", variant="primary", scale=1)
                stop_btn = gr.Button("停止处理", variant="stop", scale=1)

            with gr.Row():
                parse_status = gr.Textbox(
                    label="解析状态",
                    lines=5,
                    interactive=False
                )
                video_info = gr.JSON(
                    label="视频信息"
                )

            # Load detection preset
            def load_detection_preset_fn(preset_name):
                preset_data = tab_preset_manager.load_preset('detection', preset_name)
                if preset_data:
                    return (
                        preset_data.get('frame_diff_threshold', 30.0),
                        preset_data.get('min_stable_frames', 3),
                        preset_data.get('min_duration_sec', 2.0),
                        preset_data.get('frame_interval', 1.0),
                        preset_data.get('use_ssim', False),
                        preset_data.get('use_histogram', False),
                        f"已加载预设: {preset_name}"
                    )
                return 30.0, 3, 2.0, 1.0, False, False, f"预设 {preset_name} 不存在"

            load_detection_preset_btn.click(
                fn=load_detection_preset_fn,
                inputs=[detection_preset_dropdown],
                outputs=[frame_diff_threshold, min_stable_frames, min_duration_sec,
                        frame_interval, use_ssim, use_histogram, parse_status]
            )

            # Save detection preset
            def save_detection_preset_fn(name, threshold, stable, duration, interval, ssim, histogram):
                if not name.strip():
                    return "请输入预设名称", gr.update()
                data = {
                    'frame_diff_threshold': threshold,
                    'min_stable_frames': stable,
                    'min_duration_sec': duration,
                    'frame_interval': interval,
                    'use_ssim': ssim,
                    'use_histogram': histogram
                }
                if tab_preset_manager.save_preset('detection', name, data, overwrite=True):
                    new_choices = tab_preset_manager.list_presets('detection')
                    return f"已保存预设: {name}", gr.update(choices=new_choices, value=name)
                return "保存失败", gr.update()

            save_detection_preset_btn.click(
                fn=save_detection_preset_fn,
                inputs=[detection_preset_name, frame_diff_threshold, min_stable_frames,
                       min_duration_sec, frame_interval, use_ssim, use_histogram],
                outputs=[parse_status, detection_preset_dropdown]
            )

            def parse_video_gui(vfile, vpath, outdir, threshold, stable, duration, interval, ssim, histogram, start_time_str, end_time_str):
                """Parse video with proper timestamp directory generation"""
                import os
                from pathlib import Path as PathLib
                from datetime import datetime

                # Get video path
                video_path = vfile if vfile else vpath
                if not video_path:
                    return "Error: No video file provided", None, []

                # Generate timestamp directory based on CLI logic
                # Extract video name (handle both local files and URLs)
                is_url = video_path.startswith(('http://', 'https://'))
                if is_url:
                    # For URLs, extract filename from URL
                    url_filename = video_path.split('/')[-1].split('?')[0].split('#')[0]
                    video_name = url_filename
                    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']:
                        if video_name.lower().endswith(ext.lower()):
                            video_name = video_name[:-len(ext)]
                            break
                    video_name = video_name or 'video'
                else:
                    # For local files, use existing logic
                    video_path_obj = PathLib(video_path)
                    video_name = video_path_obj.stem  # filename without extension

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Calculate final output directory
                if outdir != "OUTPUT":
                    base_output = PathLib(outdir)
                    if not base_output.is_absolute():
                        base_output = PathLib.cwd() / base_output
                    final_output = base_output / f"{video_name}_{timestamp}"
                else:
                    final_output = PathLib(outdir) / f"{video_name}_{timestamp}"

                # Reset state
                pipeline.reset_state()
                pipeline.continue_processing()  # Ensure processing flag is not set

                # Create a temporary preset with current parameters
                from .presets import DetectionParams, ThumbnailParams
                pipeline.current_preset = Preset(
                    name='custom',
                    detection=DetectionParams(
                        frame_diff_threshold=threshold,
                        min_stable_frames=stable,
                        min_duration_sec=duration,
                        frame_interval=interval,
                        use_ssim=ssim,
                        use_histogram=histogram
                    ),
                    thumbnail=ThumbnailParams()
                )

                pipeline.video_path = PathLib(video_path)
                pipeline.output_dirs = VideoIO.init_output_dirs(str(final_output))

                # Initialize video IO
                pipeline.logger.info(f"Opening video: {video_path}")
                video_io = VideoIO(video_path)

                # Extract metadata
                pipeline.video_metadata = video_io.extract_metadata()
                pipeline.logger.info(
                    f"Video metadata: {pipeline.video_metadata.width}x{pipeline.video_metadata.height}, "
                    f"{pipeline.video_metadata.fps:.2f}fps, {pipeline.video_metadata.duration:.2f}s"
                )

                # Check if stopped
                if pipeline.check_stop():
                    return "Processing cancelled by user", None, []

                # Parse time range
                from .video_io import parse_time_string
                start_time = parse_time_string(start_time_str) if start_time_str else 0.0
                end_time = parse_time_string(end_time_str) if end_time_str else None

                # Validate time range
                if start_time is not None and start_time < 0:
                    return "Error: Start time cannot be negative", None, []
                if end_time is not None and end_time < 0:
                    return "Error: End time cannot be negative", None, []
                if start_time is not None and end_time is not None and start_time >= end_time:
                    return "Error: Start time must be less than end time", None, []

                # Detect slides
                detector = SlideDetector(
                    frame_diff_threshold=threshold,
                    min_stable_frames=stable,
                    min_duration_sec=duration,
                    use_ssim=ssim,
                    use_histogram=histogram
                )

                # Get frame iterator with time range
                frame_iter = video_io.get_frame_iter(
                    start_time=start_time if start_time is not None else 0.0,
                    end_time=end_time,
                    target_fps=1.0/interval
                )

                # Process frames
                pipeline.slide_frames = detector.process_frames(frame_iter, pipeline.video_metadata.fps)

                # Check if stopped after slide detection
                if pipeline.check_stop():
                    return "Processing cancelled by user", None, []

                pipeline.logger.info(f"Detected {len(pipeline.slide_frames)} slides")

                # Generate thumbnails and save original images
                thumb_gen = ThumbnailGenerator(
                    thumbs_dir=pipeline.output_dirs['thumbs'],
                    original_dir=pipeline.output_dirs['images'],
                    max_size=pipeline.current_preset.thumbnail.max_size,
                    quality=pipeline.current_preset.thumbnail.quality,
                    max_workers=4  # Use 4 threads for parallel I/O
                )

                # Convert to dictionaries for thumbnail generation
                slide_dicts = [asdict(sf) for sf in pipeline.slide_frames]

                # Extract frame positions for thumbnails (optimized with direct seeking)
                frame_positions = [sf.frame_number for sf in pipeline.slide_frames]

                # Generate thumbnails using direct frame seeking (much faster than sequential reading)
                # Use video_io's video_path_str for proper URL handling
                pipeline.slides_with_thumbnails = thumb_gen.generate_batch_parallel(
                    slide_dicts, video_io.video_path_str, frame_positions
                )

                # Check if stopped after thumbnail generation
                if pipeline.check_stop():
                    return "Processing cancelled by user", None, []

                # Update selection
                pipeline.selection.set_slides(pipeline.slides_with_thumbnails)

                message = f"Successfully processed video: {len(pipeline.slide_frames)} slides detected"

                # Return result
                result = {
                    'metadata': pipeline.video_metadata.to_dict(),
                    'slides': pipeline.slides_with_thumbnails,
                    'output_dir': str(pipeline.output_dirs['base'])
                }

                return (
                    message,
                    pipeline.get_status()['video_metadata'],
                    pipeline.slides_with_thumbnails
                )

            parse_btn.click(
                fn=parse_video_gui,
                inputs=[video_input, video_path_input, output_dir,
                       frame_diff_threshold, min_stable_frames, min_duration_sec,
                       frame_interval, use_ssim, use_histogram,
                       start_time_input, end_time_input],
                outputs=[parse_status, video_info, current_slides]
            )

            # Clear time range button
            def clear_time_range():
                """Clear time range inputs"""
                return "", ""

            clear_time_btn.click(
                fn=clear_time_range,
                inputs=[],
                outputs=[start_time_input, end_time_input]
            )

            # Stop processing button
            def stop_processing_fn():
                """Stop the current processing"""
                pipeline.stop_processing()
                return "已发送停止信号，正在等待当前步骤完成..."

            stop_btn.click(
                fn=stop_processing_fn,
                outputs=[parse_status]
            )

        # Tab 2: Slide Selection
        with gr.Tab("2. 筛选幻灯片", id="tab_selection"):
            gr.Markdown("## 选择要导出的幻灯片")
            gr.Markdown("点击缩略图选择/取消选择，全选/取消全选按钮控制所有选择")

            # State for selected indices
            selected_indices_state = gr.State(value=set())

            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label="幻灯片缩略图（点击选择）",
                    columns=4,
                    rows=3,
                    height="auto",
                    interactive=True,
                    allow_preview=False
                )

                with gr.Row():
                    select_all_btn = gr.Button("全选", variant="primary")
                    select_none_btn = gr.Button("取消全选")
                    invert_btn = gr.Button("反选")

                # Simple stats display
                stats_display = gr.Textbox(
                    label="已选择",
                    value="请先解析视频",
                    interactive=False
                )

                # Send buttons
                with gr.Row():
                    send_to_export_btn = gr.Button("发送到导出", variant="secondary")
                    clear_export_btn = gr.Button("清除导出选择", variant="secondary")
                    send_to_ocr_btn = gr.Button("发送到页码识别", variant="secondary")

            # Update gallery and selection state when slides change
            def update_selection_ui(slides):
                if not slides:
                    return [], "请先解析视频", set(), []

                pipeline.selection.set_slides(slides)

                # Generate gallery data
                gallery_data = pipeline.selection._to_gallery_data()

                # Initialize all as selected by default
                selected_indices = set(range(len(slides)))

                # Get selected slides for export
                selected_slides = pipeline.selection.get_selected_slides()

                return gallery_data, selected_slides

            # Update UI when slides change
            current_slides.change(
                update_selection_ui,
                inputs=[current_slides],
                outputs=[gallery, stats_display]
            )

            # Update stats when selection changes
            def update_selection_stats(current_selected):
                if current_selected is None:
                    current_selected = set()

                # Update pipeline selection
                pipeline.selection.set_selection(list(current_selected))

                # Get updated stats
                selected_count = len(current_selected)
                total_count = len(pipeline.selection.slides)
                stats = f"已选择: {selected_count}/{total_count} 张幻灯片"

                return stats

            # Handle gallery selection/deselection
            def handle_gallery_select(evt: gr.SelectData, current_selected):
                """Toggle selection when clicking on gallery item"""
                if current_selected is None:
                    current_selected = set()

                index = evt.index
                if index in current_selected:
                    current_selected.remove(index)
                else:
                    current_selected.add(index)

                # Get updated gallery and stats
                gallery_data = pipeline.selection._to_gallery_data()
                stats = update_selection_stats(current_selected)

                return gallery_data, stats, current_selected

            gallery.select(
                handle_gallery_select,
                inputs=[selected_indices_state],
                outputs=[gallery, stats_display, selected_indices_state]
            )

            # Button click handlers
            select_all_btn.click(
                lambda: (
                    pipeline.selection.select_all(),
                    set(range(len(pipeline.selection.slides)))
                ),
                outputs=[gallery, selected_indices_state]
            ).then(
                fn=update_selection_stats,
                inputs=[selected_indices_state],
                outputs=[stats_display]
            )

            select_none_btn.click(
                lambda: (
                    pipeline.selection.select_none(),
                    set()
                ),
                outputs=[gallery, selected_indices_state]
            ).then(
                fn=update_selection_stats,
                inputs=[selected_indices_state],
                outputs=[stats_display]
            )

            invert_btn.click(
                lambda: (
                    pipeline.selection.invert_selection(),
                    pipeline.selection.selected_indices.copy()
                ),
                outputs=[gallery, selected_indices_state]
            ).then(
                fn=update_selection_stats,
                inputs=[selected_indices_state],
                outputs=[stats_display]
            )

            # Send selected slides to export
            def send_to_export(current_selected, current_slides):
                if current_selected is None or not current_slides:
                    return "请先选择幻灯片"

                # Get selected slides
                selected_slides = [slide for i, slide in enumerate(current_slides) if i in current_selected]

                stats = f"已发送 {len(selected_slides)} 张幻灯片到导出页面"
                return stats

            send_to_export_btn.click(
                send_to_export,
                inputs=[selected_indices_state, current_slides],
                outputs=[stats_display]
            ).then(
                lambda: pipeline.selection.get_selected_slides(),
                outputs=[selected_slides_for_export]
            ).then(
                lambda: "selection",  # Set source to "selection"
                outputs=[selection_source]
            )

            # Clear export selection
            clear_export_btn.click(
                lambda: ("已清除导出选择", [], ""),
                outputs=[stats_display, selected_slides_for_export, selection_source]
            )

            # Send to OCR
            def send_to_ocr(current_selected, current_slides):
                if current_selected is None or not current_slides:
                    return "请先选择幻灯片"

                # Get selected slides
                selected_slides = [slide for i, slide in enumerate(current_slides) if i in current_selected]

                stats = f"已发送 {len(selected_slides)} 张幻灯片到页码识别"
                return stats

            send_to_ocr_btn.click(
                send_to_ocr,
                inputs=[selected_indices_state, current_slides],
                outputs=[stats_display]
            ).then(
                lambda: pipeline.selection.get_selected_slides(),
                outputs=[selected_slides_for_ocr]
            )

        # Tab 3: Page Number Recognition
        with gr.Tab("3. 页码识别", id="tab_ocr"):
            gr.Markdown("## OCR页码识别")
            gr.Markdown("上传图片预览ROI，或直接开始识别")

            # State for OCR results
            ocr_results = gr.State(value={})  # {slide_index: page_number}
            missing_pages = gr.State(value=[])  # List of missing page numbers

            # OCR Preset section
            with gr.Row():
                ocr_preset_dropdown = gr.Dropdown(
                    label="OCR预设",
                    choices=tab_preset_manager.list_presets('ocr'),
                    value='default',
                    scale=2
                )
                load_ocr_preset_btn = gr.Button("加载预设", scale=1)
                ocr_preset_name = gr.Textbox(
                    label="预设名称",
                    placeholder="输入名称保存当前参数",
                    scale=2
                )
                save_ocr_preset_btn = gr.Button("保存为预设", scale=1)

            # OCR Engine Settings - Full width
            with gr.Row():
                tesseract_path = gr.Textbox(
                    label="Tesseract路径",
                    value=r"d:\Program Files\Tesseract-OCR\tesseract.exe",
                    info="指定tesseract可执行文件路径（仅pytesseract需要）",
                    scale=3
                )
                ocr_engine = gr.Radio(
                    label="OCR引擎",
                    choices=['pytesseract', 'easyocr'],
                    value='pytesseract',
                    scale=1
                )

            # ROI Preview Upload - Full width
            with gr.Row():
                preview_image = gr.Image(
                    label="ROI预览（上传图片检查ROI范围）",
                    type="filepath",
                    interactive=True,
                    scale=1
                )
                preview_info = gr.Textbox(
                    label="预览信息",
                    value="请上传一张幻灯片图片来预览ROI区域",
                    lines=3,
                    interactive=False,
                    scale=1
                )

            # ROI Settings and Visualization - Split layout
            with gr.Row():
                # Left: ROI Parameters (2x2 grid)
                with gr.Column(scale=1):
                    gr.Markdown("### ROI参数设置")
                    with gr.Row():
                        roi_x = gr.Number(label="X (%)", value=70.0, precision=1, scale=1)
                        roi_y = gr.Number(label="Y (%)", value=70.0, precision=1, scale=1)
                    with gr.Row():
                        roi_w = gr.Number(label="宽度 (%)", value=25.0, precision=1, scale=1)
                        roi_h = gr.Number(label="高度 (%)", value=20.0, precision=1, scale=1)

                # Right: ROI Visualization
                with gr.Column(scale=1):
                    gr.Markdown("### ROI区域可视化")
                    roi_visualization = gr.Image(
                        label="红色框显示ROI位置",
                        type="filepath",
                        interactive=False
                    )

            # Action Buttons - Full width
            with gr.Row():
                ocr_btn = gr.Button("开始识别", variant="primary", scale=1)
                clear_ocr_btn = gr.Button("清除结果", variant="secondary", scale=1)
                send_to_export_btn = gr.Button("发送到导出", variant="secondary", scale=1)

            # Status and Results - Full width
            with gr.Row():
                ocr_status = gr.Textbox(
                    label="识别状态",
                    lines=5,
                    interactive=False,
                    scale=3
                )
                missing_pages_display = gr.JSON(
                    label="缺页列表",
                    value={},
                    scale=1
                )

            # Load OCR preset
            def load_ocr_preset_fn(preset_name):
                preset_data = tab_preset_manager.load_preset('ocr', preset_name)
                if preset_data:
                    return (
                        preset_data.get('engine', 'pytesseract'),
                        preset_data.get('roi_x', 70.0),
                        preset_data.get('roi_y', 70.0),
                        preset_data.get('roi_w', 25.0),
                        preset_data.get('roi_h', 20.0),
                        f"已加载预设: {preset_name}"
                    )
                return 'pytesseract', 70.0, 70.0, 25.0, 20.0, f"预设 {preset_name} 不存在"

            load_ocr_preset_btn.click(
                fn=load_ocr_preset_fn,
                inputs=[ocr_preset_dropdown],
                outputs=[ocr_engine, roi_x, roi_y, roi_w, roi_h, ocr_status]
            )

            # Save OCR preset
            def save_ocr_preset_fn(name, engine, x, y, w, h):
                if not name.strip():
                    return "请输入预设名称", gr.update()
                data = {
                    'engine': engine,
                    'roi_x': x,
                    'roi_y': y,
                    'roi_w': w,
                    'roi_h': h
                }
                if tab_preset_manager.save_preset('ocr', name, data, overwrite=True):
                    new_choices = tab_preset_manager.list_presets('ocr')
                    return f"已保存预设: {name}", gr.update(choices=new_choices, value=name)
                return "保存失败", gr.update()

            save_ocr_preset_btn.click(
                fn=save_ocr_preset_fn,
                inputs=[ocr_preset_name, ocr_engine, roi_x, roi_y, roi_w, roi_h],
                outputs=[ocr_status, ocr_preset_dropdown]
            )

            # Preview image change handler
            def update_preview(image_path, roi_x, roi_y, roi_w, roi_h):
                if not image_path:
                    return "请上传一张幻灯片图片来预览ROI区域", None

                if not CV2_AVAILABLE:
                    return "OpenCV not available. Install with: pip install opencv-python", None

                try:
                    # Read the image with OpenCV
                    img = cv2.imread(image_path)
                    if img is None:
                        return f"无法读取图片: {image_path}", None

                    height, width = img.shape[:2]

                    # Calculate ROI coordinates
                    roi_x_px = int(width * roi_x / 100)
                    roi_y_px = int(height * roi_y / 100)
                    roi_w_px = int(width * roi_w / 100)
                    roi_h_px = int(height * roi_h / 100)

                    # Draw ROI rectangle on the image
                    # Red color (BGR: 0, 0, 255), thickness 3
                    cv2.rectangle(img, (roi_x_px, roi_y_px), (roi_x_px + roi_w_px, roi_y_px + roi_h_px), (0, 0, 255), 3)

                    # Add ROI label
                    label = f"ROI: ({roi_x}%, {roi_y}%)"
                    cv2.putText(img, label, (roi_x_px, roi_y_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    # Save the visualized image to a temporary file
                    import tempfile
                    temp_dir = Path(tempfile.gettempdir()) / 'jlvideo2ppt'
                    temp_dir.mkdir(exist_ok=True)
                    temp_file = temp_dir / f'roi_preview_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.jpg'

                    cv2.imwrite(str(temp_file), img)

                    info = f"""图片尺寸: {width} x {height}
ROI区域 (像素):
  X: {roi_x_px} ({roi_x}%)
  Y: {roi_y_px} ({roi_y}%)
  宽度: {roi_w_px} ({roi_w}%)
  高度: {roi_h_px} ({roi_h}%)

注意：OCR将从此红色区域识别页码"""
                    return info, str(temp_file)
                except Exception as e:
                    return f"Error processing image: {e}", None

            preview_image.change(
                update_preview,
                inputs=[preview_image, roi_x, roi_y, roi_w, roi_h],
                outputs=[preview_info, roi_visualization]
            )

            # Update visualization when ROI parameters change
            def update_roi_visualization(current_image, roi_x, roi_y, roi_w, roi_h):
                if not current_image:
                    return None
                info, vis_img = update_preview(current_image, roi_x, roi_y, roi_w, roi_h)
                return vis_img

            roi_x.change(
                fn=update_roi_visualization,
                inputs=[preview_image, roi_x, roi_y, roi_w, roi_h],
                outputs=[roi_visualization]
            )

            roi_y.change(
                fn=update_roi_visualization,
                inputs=[preview_image, roi_x, roi_y, roi_w, roi_h],
                outputs=[roi_visualization]
            )

            roi_w.change(
                fn=update_roi_visualization,
                inputs=[preview_image, roi_x, roi_y, roi_w, roi_h],
                outputs=[roi_visualization]
            )

            roi_h.change(
                fn=update_roi_visualization,
                inputs=[preview_image, roi_x, roi_y, roi_w, roi_h],
                outputs=[roi_visualization]
            )

            def recognize_ocr_gui(tesseract_path, ocr_engine, selected_slides, roi_x, roi_y, roi_w, roi_h):
                """OCR page number recognition"""
                # Use selected slides if available, otherwise use all slides
                if selected_slides and len(selected_slides) > 0:
                    slides = selected_slides
                else:
                    slides = pipeline.slides_with_thumbnails

                if not slides:
                    return "No slides available for page number recognition", {}, []

                if not CV2_AVAILABLE:
                    return "OpenCV not available. Install with: pip install opencv-python", {}, []

                # Set tesseract path if using pytesseract
                if ocr_engine == 'pytesseract' and tesseract_path:
                    import pytesseract
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path

                # Initialize recognizer
                roi = ROIRect(
                    x=roi_x,
                    y=roi_y,
                    w=roi_w,
                    h=roi_h
                )

                recognizer = PageNumberRecognizer(
                    ocr_engine=ocr_engine,
                    roi=roi,
                    max_workers=4  # Use 4 processes for parallel OCR
                )

                # Load images (use only original high-res images)
                images = []
                for slide in slides:
                    # 只使用original_image_path（原图），不使用缩略图
                    img_path = slide.get('original_image_path')

                    if img_path and PathLib(img_path).exists():
                        img = cv2.imread(img_path)
                        if img is not None:
                            images.append(img)
                        else:
                            pipeline.logger.warning(f"Failed to load image: {img_path}")
                            images.append(None)
                    else:
                        pipeline.logger.warning(f"Original image not found: {img_path}")
                        images.append(None)

                # Recognize page numbers (use parallel processing by default)
                updated_slides = recognizer.update_slides_with_page_numbers(slides, images, use_parallel=True)

                # Save pages to files
                pages_dir = None
                if pipeline.output_dirs and 'base' in pipeline.output_dirs:
                    pages_dir = PathLib(pipeline.output_dirs['base']) / 'pages'
                    pages_dir.mkdir(exist_ok=True)

                # Process recognized pages
                page_counts = {}  # Track page number occurrences
                ocr_results_dict = {}
                non_duplicate_slides = []  # Slides that are not duplicates

                for i, slide in enumerate(updated_slides):
                    page_number = slide.get('page_number')
                    if page_number is not None:
                        # Track occurrences
                        if page_number not in page_counts:
                            page_counts[page_number] = 0
                        page_counts[page_number] += 1

                        # Generate filename
                        if page_counts[page_number] == 1:
                            filename = f"slide_{page_number}"
                            # Only add non-duplicates for export
                            non_duplicate_slides.append(slide)
                        else:
                            filename = f"slide_{page_number}_重复{page_counts[page_number] - 1}"

                        # Save if pages_dir exists
                        if pages_dir:
                            source_path = slide.get('original_image_path')
                            if source_path and PathLib(source_path).exists():
                                from PIL import Image as PILImage
                                try:
                                    with PILImage.open(source_path) as img:
                                        img.save(pages_dir / f"{filename}.jpg", "JPEG", quality=95)
                                except Exception as e:
                                    pipeline.logger.warning(f"Failed to save page {filename}: {e}")
                            else:
                                pipeline.logger.warning(f"Original image not found for page {filename}")

                        ocr_results_dict[i] = page_number

                # Store non-duplicate slides for export
                pipeline._non_duplicate_slides = non_duplicate_slides

                # Generate missing pages list
                if page_counts:
                    min_page = min(page_counts.keys())
                    max_page = max(page_counts.keys())
                    missing = []
                    for page_num in range(min_page, max_page + 1):
                        if page_num not in page_counts:
                            missing.append(page_num)
                else:
                    missing = []

                # Update pipeline state
                pipeline.slides_with_thumbnails = updated_slides
                pipeline.selection.set_slides(updated_slides)

                # Count successful recognitions
                recognized_count = len(page_counts)
                duplicate_count = sum(1 for v in page_counts.values() if v > 1)

                message = f"识别完成！成功识别 {recognized_count} 个页码"
                if duplicate_count > 0:
                    message += f"\n发现 {duplicate_count} 个重复页码"
                if missing:
                    message += f"\n缺失页码: {missing}"
                else:
                    message += "\n所有页码连续"

                if pages_dir:
                    message += f"\n页码图片已保存到: {pages_dir}"

                message += f"\n可导出页面数: {len(non_duplicate_slides)}"

                return message, ocr_results_dict, missing

            ocr_btn.click(
                fn=recognize_ocr_gui,
                inputs=[tesseract_path, ocr_engine, selected_slides_for_ocr, roi_x, roi_y, roi_w, roi_h],
                outputs=[ocr_status, ocr_results, missing_pages]
            )

            # Update missing pages display
            missing_pages.change(
                fn=lambda missing: {f"页面 {m}": "❌ 缺失" for m in missing},
                inputs=[missing_pages],
                outputs=[missing_pages_display]
            )

            # Clear OCR results
            clear_ocr_btn.click(
                lambda: ("已清除识别结果", {}, [], ""),
                outputs=[ocr_status, ocr_results, missing_pages, selection_source]
            ).then(
                lambda: {},
                outputs=[missing_pages_display]
            )

            # Send to export (only non-duplicate pages)
            def send_ocr_to_export():
                """Send non-duplicate recognized pages to export"""
                if hasattr(pipeline, '_non_duplicate_slides') and pipeline._non_duplicate_slides:
                    slides = pipeline._non_duplicate_slides
                    # Sort by page number
                    sorted_slides = sorted(slides, key=lambda x: x.get('page_number', 0))
                    return f"已发送 {len(sorted_slides)} 张非重复页面到导出", sorted_slides, "ocr"
                else:
                    return "没有可导出的页面，请先运行OCR识别", [], ""

            send_to_export_btn.click(
                fn=send_ocr_to_export,
                outputs=[ocr_status, selected_slides_for_export, selection_source]
            )

        # Tab 4: Export
        with gr.Tab("4. 导出", id="tab_export"):
            gr.Markdown("## 导出幻灯片")
            gr.Markdown("导出前请先在筛选页面选择幻灯片并点击'发送到导出'")

            # Display current selection source
            selection_source_display = gr.Textbox(
                label="选择来源",
                value="未收到任何页面的选择",
                interactive=False
            )

            # Export Preset section
            with gr.Row():
                export_preset_dropdown = gr.Dropdown(
                    label="导出预设",
                    choices=tab_preset_manager.list_presets('export'),
                    value='default',
                    scale=2
                )
                load_export_preset_btn = gr.Button("加载预设", scale=1)
                export_preset_name = gr.Textbox(
                    label="预设名称",
                    placeholder="输入名称保存当前参数",
                    scale=2
                )
                save_export_preset_btn = gr.Button("保存为预设", scale=1)

            with gr.Row():
                export_format = gr.Radio(
                    label="导出格式",
                    choices=['pdf', 'jpeg', 'both'],
                    value='pdf'
                )
                naming_strategy = gr.Radio(
                    label="命名策略",
                    choices=['index', 'page', 'timestamp'],
                    value='index'
                )

            with gr.Row():
                export_btn = gr.Button("开始导出", variant="primary")
                export_status = gr.Textbox(
                    label="导出状态",
                    lines=5,
                    interactive=False
                )

            # Load export preset
            def load_export_preset_fn(preset_name):
                preset_data = tab_preset_manager.load_preset('export', preset_name)
                if preset_data:
                    return (
                        preset_data.get('format', 'pdf'),
                        preset_data.get('naming', 'index'),
                        f"已加载预设: {preset_name}"
                    )
                return 'pdf', 'index', f"预设 {preset_name} 不存在"

            load_export_preset_btn.click(
                fn=load_export_preset_fn,
                inputs=[export_preset_dropdown],
                outputs=[export_format, naming_strategy, export_status]
            )

            # Save export preset
            def save_export_preset_fn(name, fmt, naming):
                if not name.strip():
                    return "请输入预设名称", gr.update()
                data = {
                    'format': fmt,
                    'naming': naming
                }
                if tab_preset_manager.save_preset('export', name, data, overwrite=True):
                    new_choices = tab_preset_manager.list_presets('export')
                    return f"已保存预设: {name}", gr.update(choices=new_choices, value=name)
                return "保存失败", gr.update()

            save_export_preset_btn.click(
                fn=save_export_preset_fn,
                inputs=[export_preset_name, export_format, naming_strategy],
                outputs=[export_status, export_preset_dropdown]
            )

            # Update selection source display
            def update_selection_source(selected_slides, source):
                if selected_slides and len(selected_slides) > 0:
                    if source == "ocr":
                        return f"使用页码识别页面选择: {len(selected_slides)} 张幻灯片"
                    else:
                        return f"使用筛选页面选择: {len(selected_slides)} 张幻灯片"
                else:
                    return "使用当前页面选择"

            # Monitor selection source changes
            selected_slides_for_export.change(
                update_selection_source,
                inputs=[selected_slides_for_export, selection_source],
                outputs=[selection_source_display]
            )

            def export_gui(fmt, naming, selected_slides=None):
                """Export slides"""

                # Use selected slides from selection tab if available, otherwise use pipeline selection
                if selected_slides and len(selected_slides) > 0:
                    slides_to_export = selected_slides
                else:
                    slides_to_export = pipeline.selection.get_selected_slides()

                if not slides_to_export:
                    return "No slides selected for export. Please select slides in Tab 2 first."

                # Check if any slide has page_number
                has_page_numbers = any(slide.get('page_number') is not None for slide in slides_to_export)

                # Auto-switch to 'page' naming if page numbers are detected
                if has_page_numbers and naming == 'index':
                    naming = 'page'

                # Initialize exporter
                if pipeline.output_dirs and 'base' in pipeline.output_dirs:
                    output_dir = str(pipeline.output_dirs['base'])
                else:
                    output_dir = pipeline.current_preset.export.output_dir

                exporter = Exporter(output_dir)

                # Export
                if fmt == 'pdf':
                    pdf_path = exporter.export_pdf(
                        slides_to_export,
                        page_size=pipeline.current_preset.export.pdf_page_size,
                        margin=pipeline.current_preset.export.pdf_margin,
                        naming=naming
                    )
                    results = {'pdf': [pdf_path]}

                elif fmt == 'jpeg':
                    # Create separate images directory for JPEG exports
                    images_dir = PathLib(output_dir) / 'exported_images'
                    images_dir.mkdir(exist_ok=True)
                    jpeg_exporter = Exporter(str(images_dir))
                    image_paths = jpeg_exporter.export_images(
                        slides_to_export,
                        format='jpeg',
                        quality=pipeline.current_preset.export.image_quality,
                        naming=naming
                    )
                    results = {'images': image_paths}

                elif fmt == 'both':
                    # Export both PDF and JPEG images in the same directory
                    # First export PDF
                    pdf_path = exporter.export_pdf(
                        slides_to_export,
                        page_size=pipeline.current_preset.export.pdf_page_size,
                        margin=pipeline.current_preset.export.pdf_margin,
                        naming=naming
                    )

                    # Then export JPEG images to the same directory
                    jpeg_paths = exporter.export_images(
                        slides_to_export,
                        format='jpeg',
                        quality=pipeline.current_preset.export.image_quality,
                        naming=naming
                    )

                    results = {'pdf': [pdf_path], 'images': jpeg_paths}

                # Get summary
                summary = exporter.get_export_summary(results)

                return summary

            export_btn.click(
                fn=export_gui,
                inputs=[export_format, naming_strategy, selected_slides_for_export],
                outputs=[export_status]
            )

    return demo


def main():
    """Main entry point for CLI"""
    import argparse

    parser = argparse.ArgumentParser(description="Convert video to slides")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("-o", "--output", default="OUTPUT", help="Output directory")
    parser.add_argument("-p", "--preset", default="default", help="Preset to use")
    parser.add_argument("--gui", action="store_true", help="Launch Gradio UI")

    args = parser.parse_args()

    if args.gui:
        # Launch Gradio UI
        pipeline = VideoProcessingPipeline()
        demo = create_gradio_interface(pipeline)
        demo.launch()
    else:
        # Run CLI mode
        pipeline = VideoProcessingPipeline()
        success, message, _ = pipeline.parse_video(args.video_path, args.output, args.preset)

        if success:
            print(message)
            print("\nPipeline status:")
            print(pipeline.get_status())
        else:
            print(f"Error: {message}")
            sys.exit(1)


if __name__ == "__main__":
    main()
