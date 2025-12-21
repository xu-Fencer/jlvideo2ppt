"""
Thumbnail Module

Handles thumbnail generation and caching:
- Batch thumbnail creation from video frames
- JPEG compression with configurable quality
- Thumbnail caching and path management
- Cleanup utilities
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from PIL import Image
from tqdm import tqdm
import hashlib
import logging


class ThumbnailGenerator:
    """Generate and cache video frame thumbnails"""

    def __init__(self,
                 thumbs_dir: Union[str, Path],
                 original_dir: Optional[Union[str, Path]] = None,
                 max_size: int = 800,
                 quality: int = 85):
        """
        Initialize thumbnail generator

        Args:
            thumbs_dir: Directory to store thumbnails
            original_dir: Directory to store original resolution images
            max_size: Maximum dimension (width or height) for thumbnails
            quality: JPEG quality (1-100)
        """
        self.thumbs_dir = Path(thumbs_dir)
        self.original_dir = Path(original_dir) if original_dir else None
        self.max_size = max_size
        self.quality = quality
        self.logger = logging.getLogger(__name__)

        # Ensure directories exist
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        if self.original_dir:
            self.original_dir.mkdir(parents=True, exist_ok=True)

    def generate_thumbnail(self,
                          frame: np.ndarray,
                          slide_idx: int,
                          timestamp: float) -> Dict[str, str]:
        """
        Generate a thumbnail and optionally save original image

        Args:
            frame: Video frame (BGR format)
            slide_idx: Slide index
            timestamp: Frame timestamp

        Returns:
            Dictionary with 'thumbnail_path' and optionally 'original_path'
        """
        # Generate filename based on slide index and timestamp
        filename = f"slide_{slide_idx:04d}_{timestamp:.2f}s.jpg"
        thumb_path = self.thumbs_dir / filename
        original_path = None

        result = {'thumbnail_path': None}

        # Check if thumbnail already exists
        if thumb_path.exists():
            result['thumbnail_path'] = str(thumb_path)
            if self.original_dir:
                original_path = self.original_dir / filename
                result['original_path'] = str(original_path) if original_path.exists() else None
            return result

        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # Save original image if directory is specified
            if self.original_dir:
                original_path = self.original_dir / filename
                pil_image.save(
                    original_path,
                    "JPEG",
                    quality=95,  # Higher quality for original
                    optimize=True,
                    dpi=(150, 150)
                )
                result['original_path'] = str(original_path)
                self.logger.debug(f"Saved original image: {filename}")

            # Create and save thumbnail
            pil_image.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)
            pil_image.save(
                thumb_path,
                "JPEG",
                quality=self.quality,
                optimize=True,
                dpi=(150, 150)
            )

            result['thumbnail_path'] = str(thumb_path)
            self.logger.debug(f"Generated thumbnail: {filename}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to generate thumbnail {filename}: {e}")
            # Return placeholder or raise
            raise

    def generate_batch(self,
                      slide_frames: List[Dict],
                      frame_iterator) -> List[Dict]:
        """
        Generate thumbnails for multiple slides

        Args:
            slide_frames: List of slide frame metadata
            frame_iterator: Video frame iterator

        Returns:
            Updated slide frames with thumbnail paths
        """
        self.logger.info(f"Generating {len(slide_frames)} thumbnails...")

        # Convert iterator to list for multiple passes
        all_frames = list(frame_iterator)

        updated_slides = []

        for idx, slide in enumerate(tqdm(slide_frames, desc="Generating thumbnails")):
            try:
                # Find the frame closest to the slide timestamp
                target_time = slide['timestamp']
                best_frame = None
                best_diff = float('inf')

                for frame_number, frame, timestamp in all_frames:
                    time_diff = abs(timestamp - target_time)
                    if time_diff < best_diff:
                        best_diff = time_diff
                        best_frame = frame

                    # If we passed the target time, we can break
                    if timestamp > target_time + 1.0:  # 1 second tolerance
                        break

                if best_frame is not None:
                    # Generate thumbnail and original image
                    result = self.generate_thumbnail(best_frame, idx, target_time)
                    slide['thumbnail_path'] = result['thumbnail_path']
                    slide['original_image_path'] = result.get('original_path')
                else:
                    self.logger.warning(f"Could not find frame for slide at {target_time:.2f}s")

                updated_slides.append(slide)

            except Exception as e:
                self.logger.error(f"Error generating thumbnail for slide {idx}: {e}")
                slide['thumbnail_path'] = None
                slide['original_image_path'] = None
                updated_slides.append(slide)

        return updated_slides

    @staticmethod
    def get_thumbnail_info(thumb_path: Union[str, Path]) -> Optional[Dict]:
        """
        Get thumbnail file information

        Args:
            thumb_path: Path to thumbnail

        Returns:
            Dictionary with thumbnail info or None
        """
        thumb_path = Path(thumb_path)
        if not thumb_path.exists():
            return None

        stat = thumb_path.stat()
        with Image.open(thumb_path) as img:
            return {
                'path': str(thumb_path),
                'size': img.size,
                'mode': img.mode,
                'file_size': stat.st_size,
                'modified': stat.st_mtime
            }

    def cleanup_old_thumbnails(self, max_age_days: int = 7) -> int:
        """
        Clean up old thumbnail files

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of files deleted
        """
        import time
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        deleted_count = 0

        for thumb_path in self.thumbs_dir.glob("*.jpg"):
            if current_time - thumb_path.stat().st_mtime > max_age_seconds:
                try:
                    thumb_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete {thumb_path}: {e}")

        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old thumbnails")

        return deleted_count

    def get_thumbnail_size(self, thumb_path: Union[str, Path]) -> Tuple[int, int]:
        """
        Get thumbnail dimensions

        Args:
            thumb_path: Path to thumbnail

        Returns:
            Tuple of (width, height)
        """
        thumb_path = Path(thumb_path)
        if not thumb_path.exists():
            return (0, 0)

        with Image.open(thumb_path) as img:
            return img.size

    @staticmethod
    def create_thumbnail_hash(frame: np.ndarray) -> str:
        """
        Create hash of frame for caching

        Args:
            frame: Video frame

        Returns:
            MD5 hash string
        """
        # Resize frame to small size for hashing
        small = cv2.resize(frame, (64, 64))
        # Convert to bytes
        frame_bytes = small.tobytes()
        # Create hash
        return hashlib.md5(frame_bytes).hexdigest()

    def batch_resize_thumbnails(self,
                               source_dir: Union[str, Path],
                               target_dir: Optional[Union[str, Path]] = None,
                               max_size: Optional[int] = None,
                               quality: Optional[int] = None) -> None:
        """
        Batch resize existing thumbnails

        Args:
            source_dir: Source directory with thumbnails
            target_dir: Target directory (defaults to source_dir)
            max_size: New max size (uses instance default if None)
            quality: New quality (uses instance default if None)
        """
        source_dir = Path(source_dir)
        target_dir = Path(target_dir) if target_dir else source_dir
        max_size = max_size if max_size is not None else self.max_size
        quality = quality if quality is not None else self.quality

        target_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(list(source_dir.glob("*.jpg")),
                            desc="Resizing thumbnails"):
            try:
                with Image.open(img_path) as img:
                    # Only resize if larger than target
                    if img.width > max_size or img.height > max_size:
                        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                    # Save with new quality
                    target_path = target_dir / img_path.name
                    img.save(
                        target_path,
                        "JPEG",
                        quality=quality,
                        optimize=True
                    )

            except Exception as e:
                self.logger.error(f"Failed to resize {img_path}: {e}")

    def optimize_thumbnails(self) -> Dict[str, int]:
        """
        Optimize all thumbnails in the directory

        Returns:
            Dictionary with optimization statistics
        """
        stats = {
            'total': 0,
            'optimized': 0,
            'skipped': 0,
            'errors': 0
        }

        for img_path in tqdm(list(self.thumbs_dir.glob("*.jpg")),
                            desc="Optimizing thumbnails"):
            stats['total'] += 1
            try:
                with Image.open(img_path) as img:
                    original_size = img_path.stat().st_size

                    # Re-save with optimization
                    img.save(
                        img_path,
                        "JPEG",
                        quality=self.quality,
                        optimize=True,
                        dpi=(150, 150)
                    )

                    new_size = img_path.stat().st_size
                    if new_size < original_size:
                        stats['optimized'] += 1
                    else:
                        stats['skipped'] += 1

            except Exception as e:
                self.logger.error(f"Failed to optimize {img_path}: {e}")
                stats['errors'] += 1

        self.logger.info(f"Optimization complete: {stats}")
        return stats
