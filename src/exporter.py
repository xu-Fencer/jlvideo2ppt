"""
Exporter Module

Handles export to various formats:
- Image formats (PNG, JPEG)
- PDF generation with ReportLab
- Naming strategies
- Error handling and recovery
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.error("PIL not available. Install with: pip install pillow")

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.error("ReportLab not available. Install with: pip install reportlab")


class Exporter:
    """Export slides to various formats"""

    # Supported image formats
    IMAGE_FORMATS = {
        'png': 'PNG',
        'jpg': 'JPEG',
        'jpeg': 'JPEG',
    }

    # Page sizes (width, height) in points (1/72 inch)
    PAGE_SIZES = {
        'A4': A4,
        'Letter': letter,
        'A3': (841.89, 1199.11),
        'A5': (419.53, 595.28),
    }

    def __init__(self, output_dir: str):
        """
        Initialize exporter

        Args:
            output_dir: Output directory path
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Ensure required dependencies
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL not available. Install with: pip install pillow")

    def export_images(self,
                     slides: List[Dict],
                     format: str = 'png',
                     quality: int = 95,
                     naming: str = 'index') -> List[str]:
        """
        Export slides as images

        Args:
            slides: List of slide metadata
            format: Image format ('png' or 'jpeg')
            quality: JPEG quality (1-100)
            naming: Naming strategy ('index', 'page', 'timestamp')

        Returns:
            List of exported file paths
        """
        if format.lower() not in self.IMAGE_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Use: {', '.join(self.IMAGE_FORMATS.keys())}")

        exported_files = []

        for i, slide in enumerate(slides):
            try:
                # Get source image (prefer original image over thumbnail for better quality)
                source_path = slide.get('original_image_path') or slide.get('thumbnail_path')
                if not source_path or not Path(source_path).exists():
                    self.logger.warning(f"Slide {i}: No image available, skipping")
                    continue

                # Generate filename
                filename = self._generate_filename(
                    slide=slide,
                    index=i,
                    format=format,
                    naming=naming
                )
                output_path = self.output_dir / filename

                # Open and process image
                with Image.open(source_path) as img:
                    # Convert to RGB if necessary (for JPEG)
                    if format.lower() in ['jpg', 'jpeg'] and img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Save image
                    save_kwargs = {}
                    if format.lower() in ['jpg', 'jpeg']:
                        save_kwargs['quality'] = quality
                        save_kwargs['optimize'] = True

                    img.save(output_path, self.IMAGE_FORMATS[format.lower()], **save_kwargs)
                    exported_files.append(str(output_path))

                    self.logger.debug(f"Exported slide {i} to {filename}")

            except Exception as e:
                self.logger.error(f"Failed to export slide {i}: {e}")
                continue

        self.logger.info(f"Exported {len(exported_files)} images to {self.output_dir}")
        return exported_files

    def export_pdf(self,
                  slides: List[Dict],
                  page_size: str = 'A4',
                  margin: float = 0.5,
                  naming: str = 'index') -> str:
        """
        Export slides as PDF

        Args:
            slides: List of slide metadata
            page_size: Ignored (kept for backward compatibility)
            margin: Page margin in inches (for padding)
            naming: Naming strategy

        Returns:
            Path to exported PDF file
        """
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("ReportLab not available. Install with: pip install reportlab")

        # Generate PDF filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"slides_{timestamp}.pdf"
        pdf_path = self.output_dir / pdf_filename

        try:
            # Convert margin to points
            margin_points = margin * inch

            # Create a temporary list to store all slides with their image paths and dimensions
            valid_slides = []
            for i, slide in enumerate(slides):
                try:
                    # Get source image (prefer original image over thumbnail for better quality)
                    source_path = slide.get('original_image_path') or slide.get('thumbnail_path')
                    if not source_path or not Path(source_path).exists():
                        self.logger.warning(f"Slide {i}: No image available, skipping")
                        continue

                    # Get image dimensions
                    with Image.open(source_path) as img:
                        img_width, img_height = img.size

                    valid_slides.append({
                        'path': source_path,
                        'width': img_width,
                        'height': img_height,
                        'page_number': slide.get('page_number')
                    })

                except Exception as e:
                    self.logger.error(f"Failed to process slide {i}: {e}")
                    continue

            if not valid_slides:
                raise ValueError("No valid slides found for PDF export")

            # Start creating PDF
            c = canvas.Canvas(str(pdf_path))

            for i, slide_info in enumerate(valid_slides):
                # Set page size before drawing this slide
                c.setPageSize((slide_info['width'], slide_info['height']))

                # Draw image at (0, 0) - full page (no page number overlay)
                c.drawImage(
                    slide_info['path'],
                    0, 0,
                    width=slide_info['width'],
                    height=slide_info['height'],
                    preserveAspectRatio=True
                )

                # Start new page for next slide (except last)
                if i < len(valid_slides) - 1:
                    c.showPage()

                self.logger.debug(f"Added slide {i} to PDF")

            # Save PDF
            c.save()
            self.logger.info(f"Exported PDF to {pdf_path}")
            return str(pdf_path)

        except Exception as e:
            self.logger.error(f"Failed to export PDF: {e}")
            # Clean up partial PDF
            if pdf_path.exists():
                pdf_path.unlink()
            raise

    def export_both(self,
                   slides: List[Dict],
                   image_format: str = 'jpeg',
                   image_quality: int = 95,
                   pdf_page_size: str = 'A4',
                   pdf_margin: float = 0.5,
                   naming: str = 'index') -> Dict[str, List[str]]:
        """
        Export slides as both images and PDF

        Args:
            slides: List of slide metadata
            image_format: Image format
            image_quality: JPEG quality
            pdf_page_size: PDF page size
            pdf_margin: PDF margin
            naming: Naming strategy

        Returns:
            Dictionary with 'images' and 'pdf' keys
        """
        results = {}

        # Export images
        try:
            image_dir = self.output_dir / 'images'
            image_dir.mkdir(exist_ok=True)
            exporter = Exporter(str(image_dir))
            results['images'] = exporter.export_images(
                slides,
                format=image_format,
                quality=image_quality,
                naming=naming
            )
        except Exception as e:
            self.logger.error(f"Image export failed: {e}")
            results['images'] = []

        # Export PDF
        try:
            results['pdf'] = [self.export_pdf(
                slides,
                page_size=pdf_page_size,
                margin=pdf_margin,
                naming=naming
            )]
        except Exception as e:
            self.logger.error(f"PDF export failed: {e}")
            results['pdf'] = []

        return results

    def _generate_filename(self,
                          slide: Dict,
                          index: int,
                          format: str,
                          naming: str) -> str:
        """
        Generate filename based on naming strategy

        Args:
            slide: Slide metadata
            index: Slide index
            format: File format extension

        Returns:
            Generated filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if naming == 'page' and slide.get('page_number') is not None:
            # Use page number (e.g., slide_0001.jpeg)
            filename = f"slide_{slide['page_number']:04d}.{format}"
        elif naming == 'timestamp' and slide.get('timestamp') is not None:
            # Use timestamp
            filename = f"slide_{slide['timestamp']:.2f}s.{format}"
        elif naming == 'index':
            # Use index
            filename = f"slide_{index:04d}.{format}"
        else:
            # Fallback to index
            filename = f"slide_{index:04d}.{format}"

        return filename

    def get_export_summary(self, export_results: Dict[str, List[str]]) -> str:
        """
        Get summary of export results

        Args:
            export_results: Results from export methods

        Returns:
            Summary string
        """
        summary_lines = ["Export Summary:", "=" * 50]

        if 'images' in export_results and export_results['images']:
            summary_lines.append(f"Images: {len(export_results['images'])} files")
            for img_path in export_results['images']:
                summary_lines.append(f"  - {img_path}")

        if 'pdf' in export_results and export_results['pdf']:
            summary_lines.append(f"\nPDF: {len(export_results['pdf'])} file")
            for pdf_path in export_results['pdf']:
                summary_lines.append(f"  - {pdf_path}")

        summary_lines.append(f"\nOutput directory: {self.output_dir}")

        return "\n".join(summary_lines)

    @staticmethod
    def validate_slides(slides: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Validate slides before export

        Args:
            slides: List of slide metadata

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not slides:
            errors.append("No slides to export")
            return False, errors

        for i, slide in enumerate(slides):
            # Check for thumbnail
            if not slide.get('thumbnail_path'):
                errors.append(f"Slide {i}: No thumbnail path")
                continue

            thumb_path = Path(slide['thumbnail_path'])
            if not thumb_path.exists():
                errors.append(f"Slide {i}: Thumbnail file not found: {thumb_path}")

        is_valid = len(errors) == 0
        return is_valid, errors


class ExportStatistics:
    """Track export statistics"""

    def __init__(self):
        """Initialize statistics tracker"""
        self.reset()

    def reset(self):
        """Reset statistics"""
        self.total_slides = 0
        self.exported_slides = 0
        self.failed_slides = 0
        self.export_time = 0.0
        self.file_sizes = []

    def update(self, **kwargs):
        """Update statistics"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_summary(self) -> Dict:
        """Get statistics summary"""
        return {
            'total_slides': self.total_slides,
            'exported_slides': self.exported_slides,
            'failed_slides': self.failed_slides,
            'success_rate': (
                self.exported_slides / self.total_slides
                if self.total_slides > 0 else 0
            ),
            'export_time': self.export_time,
            'avg_time_per_slide': (
                self.export_time / self.exported_slides
                if self.exported_slides > 0 else 0
            ),
            'total_file_size': sum(self.file_sizes),
        }

    def __str__(self) -> str:
        """String representation of statistics"""
        summary = self.get_summary()
        return (
            f"Exported {summary['exported_slides']}/{summary['total_slides']} slides "
            f"({summary['success_rate']:.1%} success rate) in "
            f"{summary['export_time']:.2f}s"
        )
