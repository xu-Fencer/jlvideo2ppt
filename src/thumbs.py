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
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


class ThumbnailGenerator:
    """Generate and cache video frame thumbnails"""

    def __init__(self,
                 thumbs_dir: Union[str, Path],
                 original_dir: Optional[Union[str, Path]] = None,
                 max_size: int = 800,
                 quality: int = 85,
                 max_workers: Optional[int] = None):
        """
        Initialize thumbnail generator

        Args:
            thumbs_dir: Directory to store thumbnails
            original_dir: Directory to store original resolution images
            max_size: Maximum dimension (width or height) for thumbnails
            quality: JPEG quality (1-100)
            max_workers: Maximum number of worker threads for parallel I/O
        """
        self.thumbs_dir = Path(thumbs_dir)
        self.original_dir = Path(original_dir) if original_dir else None
        self.max_size = max_size
        self.quality = quality
        self.max_workers = max_workers if max_workers is not None else min(8, multiprocessing.cpu_count() * 2)
        self.logger = logging.getLogger(__name__)

        # Ensure directories exist
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        if self.original_dir:
            self.original_dir.mkdir(parents=True, exist_ok=True)

    def _resize_optimized(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """
        Optimized two-stage image resizing using OpenCV

        Args:
            image: Input image (BGR format)
            target_size: Maximum dimension (width or height)

        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)

        # Only resize if needed
        if max_dim <= target_size:
            return image

        # Two-stage scaling for quality and speed
        # Stage 1: Fast downsample to 2x target size (using INTER_AREA for downsampling)
        stage1_scale = (target_size * 2) / max_dim
        if stage1_scale < 0.5:  # Only do two-stage if significant downsampling needed
            intermediate_w = int(w * stage1_scale)
            intermediate_h = int(h * stage1_scale)
            stage1 = cv2.resize(image, (intermediate_w, intermediate_h), interpolation=cv2.INTER_AREA)

            # Stage 2: High-quality resize to target size (using INTER_LANCZOS4)
            stage2_scale = target_size / max_dim
            if stage2_scale < 1:
                return cv2.resize(stage1, (w, h), fx=stage2_scale, fy=stage2_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            else:
                return stage1
        else:
            # Single-stage scaling for moderate images
            scale = target_size / max_dim
            return cv2.resize(image, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_LANCZOS4)

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
            # Save original image if directory is specified (use OpenCV for speed)
            if self.original_dir:
                original_path = self.original_dir / filename
                cv2.imwrite(
                    str(original_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                )
                result['original_path'] = str(original_path)
                self.logger.debug(f"Saved original image: {filename}")

            # Optimized thumbnail generation using OpenCV with two-stage scaling
            h, w = frame.shape[:2]
            max_dim = max(h, w)

            # Only resize if image is larger than target
            if max_dim > self.max_size:
                thumbnail = self._resize_optimized(frame, self.max_size)
            else:
                # Just copy if already small enough
                thumbnail = frame.copy()

            # Save thumbnail using OpenCV (faster than PIL)
            cv2.imwrite(
                str(thumb_path),
                thumbnail,
                [cv2.IMWRITE_JPEG_QUALITY, self.quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
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
        Generate thumbnails for multiple slides (memory-efficient version)

        Args:
            slide_frames: List of slide frame metadata
            frame_iterator: Video frame iterator

        Returns:
            Updated slide frames with thumbnail paths
        """
        self.logger.info(f"Generating {len(slide_frames)} thumbnails...")

        # Sort slide frames by timestamp for single-pass processing
        indexed_slides = [(idx, slide) for idx, slide in enumerate(slide_frames)]
        indexed_slides.sort(key=lambda x: x[1]['timestamp'])

        # Create a map of target timestamps to slide indices
        pending_slides = {slide['timestamp']: (idx, slide) for idx, slide in indexed_slides}
        target_times = sorted(pending_slides.keys())

        updated_slides = [None] * len(slide_frames)
        current_target_idx = 0
        last_frame = None
        last_timestamp = -1

        # Process frames one at a time (memory efficient)
        for frame_number, frame, timestamp in tqdm(frame_iterator, desc="Generating thumbnails"):
            # Check if we've passed all target times
            if current_target_idx >= len(target_times):
                # Release the last frame
                if last_frame is not None:
                    del last_frame
                    last_frame = None
                break

            # Find slides that match this frame's timestamp
            while current_target_idx < len(target_times):
                target_time = target_times[current_target_idx]

                # If this frame is close enough to the target
                if abs(timestamp - target_time) < 0.5:  # 0.5 second tolerance
                    idx, slide = pending_slides[target_time]
                    try:
                        result = self.generate_thumbnail(frame, idx, target_time)
                        slide['thumbnail_path'] = result['thumbnail_path']
                        slide['original_image_path'] = result.get('original_path')
                    except Exception as e:
                        self.logger.error(f"Error generating thumbnail for slide {idx}: {e}")
                        slide['thumbnail_path'] = None
                        slide['original_image_path'] = None
                    updated_slides[idx] = slide
                    current_target_idx += 1

                # If we've passed the target time, use the last frame or current frame
                elif timestamp > target_time:
                    idx, slide = pending_slides[target_time]
                    # Use the closer frame (last or current)
                    use_frame = frame
                    if last_frame is not None and abs(last_timestamp - target_time) < abs(timestamp - target_time):
                        use_frame = last_frame

                    try:
                        result = self.generate_thumbnail(use_frame, idx, target_time)
                        slide['thumbnail_path'] = result['thumbnail_path']
                        slide['original_image_path'] = result.get('original_path')
                    except Exception as e:
                        self.logger.error(f"Error generating thumbnail for slide {idx}: {e}")
                        slide['thumbnail_path'] = None
                        slide['original_image_path'] = None
                    updated_slides[idx] = slide
                    current_target_idx += 1
                else:
                    # Haven't reached the target time yet
                    break

            # Keep only the last frame for potential use
            if last_frame is not None:
                del last_frame
            last_frame = frame.copy()  # Make a copy since frame may be reused
            last_timestamp = timestamp

            # Release current frame reference
            del frame

        # Clean up
        if last_frame is not None:
            del last_frame

        # Handle any remaining slides (use last available frame info)
        for idx, slide in indexed_slides:
            if updated_slides[idx] is None:
                self.logger.warning(f"Could not find frame for slide at {slide['timestamp']:.2f}s")
                slide['thumbnail_path'] = None
                slide['original_image_path'] = None
                updated_slides[idx] = slide

        return updated_slides

    def generate_batch_parallel(self,
                                slide_frames: List[Dict],
                                frame_iterator) -> List[Dict]:
        """
        Generate thumbnails for multiple slides using parallel I/O

        Args:
            slide_frames: List of slide frame metadata
            frame_iterator: Video frame iterator

        Returns:
            Updated slide frames with thumbnail paths
        """
        self.logger.info(f"Generating {len(slide_frames)} thumbnails with parallel I/O...")

        # Sort slide frames by timestamp for single-pass processing
        indexed_slides = [(idx, slide) for idx, slide in enumerate(slide_frames)]
        indexed_slides.sort(key=lambda x: x[1]['timestamp'])

        # Create a map of target timestamps to slide indices
        pending_slides = {slide['timestamp']: (idx, slide) for idx, slide in indexed_slides}
        target_times = sorted(pending_slides.keys())

        updated_slides = [None] * len(slide_frames)
        current_target_idx = 0
        last_frame = None
        last_timestamp = -1

        # Collect frames to process
        frames_to_process = []

        # Process frames one at a time (memory efficient)
        for frame_number, frame, timestamp in tqdm(frame_iterator, desc="Collecting frames"):
            # Check if we've passed all target times
            if current_target_idx >= len(target_times):
                # Release the last frame
                if last_frame is not None:
                    del last_frame
                    last_frame = None
                break

            # Find slides that match this frame's timestamp
            while current_target_idx < len(target_times):
                target_time = target_times[current_target_idx]

                # If this frame is close enough to the target
                if abs(timestamp - target_time) < 0.5:  # 0.5 second tolerance
                    idx, slide = pending_slides[target_time]
                    frames_to_process.append((idx, frame.copy(), target_time))
                    updated_slides[idx] = slide
                    current_target_idx += 1

                # If we've passed the target time, use the last frame or current frame
                elif timestamp > target_time:
                    idx, slide = pending_slides[target_time]
                    # Use the closer frame (last or current)
                    use_frame = frame
                    if last_frame is not None and abs(last_timestamp - target_time) < abs(timestamp - target_time):
                        use_frame = last_frame

                    frames_to_process.append((idx, use_frame.copy(), target_time))
                    updated_slides[idx] = slide
                    current_target_idx += 1
                else:
                    # Haven't reached the target time yet
                    break

            # Keep only the last frame for potential use
            if last_frame is not None:
                del last_frame
            last_frame = frame.copy()  # Make a copy since frame may be reused
            last_timestamp = timestamp

            # Release current frame reference
            del frame

        # Clean up
        if last_frame is not None:
            del last_frame

        # Handle any remaining slides (use last available frame info)
        for idx, slide in indexed_slides:
            if updated_slides[idx] is None:
                self.logger.warning(f"Could not find frame for slide at {slide['timestamp']:.2f}s")
                slide['thumbnail_path'] = None
                slide['original_image_path'] = None
                updated_slides[idx] = slide

        # Parallel I/O for thumbnail generation
        if frames_to_process:
            self.logger.info(f"Parallel writing {len(frames_to_process)} thumbnails with {self.max_workers} threads")

            def process_single_thumbnail(args):
                idx, frame, timestamp = args
                try:
                    result = self.generate_thumbnail(frame, idx, timestamp)
                    return idx, result
                except Exception as e:
                    self.logger.error(f"Error generating thumbnail for slide {idx}: {e}")
                    return idx, {'thumbnail_path': None, 'original_path': None}

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                futures = [executor.submit(process_single_thumbnail, args) for args in frames_to_process]

                # Collect results
                for future in tqdm(futures, desc="Writing thumbnails (parallel)"):
                    idx, result = future.result()
                    updated_slides[idx]['thumbnail_path'] = result['thumbnail_path']
                    updated_slides[idx]['original_image_path'] = result.get('original_path')

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
