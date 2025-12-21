"""
Page Number Recognition Module

Handles OCR-based page number detection:
- ROI (Region of Interest) configuration
- OCR using pytesseract or easyocr
- Number extraction using regex
- Conflict detection and resolution
- Sorting interface
"""

import re
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import traceback
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logging.warning("pytesseract not available. Install with: pip install pytesseract")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("easyocr not available. Install with: pip install easyocr")


@dataclass
class ROIRect:
    """Region of Interest rectangle in percentage coordinates"""

    x: float  # X coordinate (0-100%)
    y: float  # Y coordinate (0-100%)
    w: float  # Width (0-100%)
    h: float  # Height (0-100%)

    def to_absolute(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert percentage coordinates to absolute pixel coordinates"""
        return (
            int(self.x * img_width / 100),
            int(self.y * img_height / 100),
            int(self.w * img_width / 100),
            int(self.h * img_height / 100)
        )

    @classmethod
    def from_dict(cls, data: Dict) -> 'ROIRect':
        """Create ROIRect from dictionary"""
        return cls(x=data['x'], y=data['y'], w=data['w'], h=data['h'])

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {'x': self.x, 'y': self.y, 'w': self.w, 'h': self.h}


@dataclass
class OCRResult:
    """OCR recognition result"""

    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)


class PageNumberRecognizer:
    """Recognize page numbers in slides using OCR"""

    # Default ROI (bottom-right corner)
    DEFAULT_ROI = ROIRect(x=70.0, y=70.0, w=25.0, h=20.0)

    # Regex patterns for page numbers
    PAGE_NUMBER_PATTERNS = [
        r'\b\d+\b',  # Simple numbers
        r'\bPage\s+\d+\b',  # "Page 123"
        r'\bp\.\s*\d+\b',  # "p. 123"
        r'\b第\s*\d+\s*页\b',  # Chinese "第X页"
        r'\b\d+\s*/\s*\d+\b',  # "1/10" format
    ]

    def __init__(self,
                 ocr_engine: str = 'pytesseract',
                 roi: Optional[ROIRect] = None,
                 max_workers: Optional[int] = None):
        """
        Initialize page number recognizer

        Args:
            ocr_engine: 'pytesseract' or 'easyocr'
            roi: Region of Interest for page number location
            max_workers: Maximum number of worker processes for parallel OCR
        """
        self.ocr_engine = ocr_engine
        self.roi = roi if roi else self.DEFAULT_ROI
        self.max_workers = max_workers if max_workers is not None else min(4, multiprocessing.cpu_count())
        self.logger = logging.getLogger(__name__)

        # Initialize OCR engine
        if ocr_engine == 'pytesseract':
            if not PYTESSERACT_AVAILABLE:
                raise RuntimeError("pytesseract not available. Install with: pip install pytesseract")
            self._init_tesseract()
        elif ocr_engine == 'easyocr':
            if not EASYOCR_AVAILABLE:
                raise RuntimeError("easyocr not available. Install with: pip install easyocr")
            self._init_easyocr()
        else:
            raise ValueError(f"Unsupported OCR engine: {ocr_engine}")

    def _init_tesseract(self):
        """Initialize Tesseract OCR"""
        try:
            # Try to get version
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract version: {version}")

            # Set default config for better number recognition
            self.tesseract_config = r'--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789./-'
        except Exception as e:
            self.logger.error(f"Failed to initialize Tesseract: {e}")
            raise

    def _init_easyocr(self):
        """Initialize EasyOCR"""
        try:
            # Initialize EasyOCR - use basic parameters, optimization happens in readtext
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            self.logger.info("EasyOCR initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize EasyOCR: {e}")
            raise

    def extract_roi(self, image: np.ndarray) -> np.ndarray:
        """
        Extract ROI from image

        Args:
            image: Input image (BGR format)

        Returns:
            ROI image
        """
        height, width = image.shape[:2]
        x, y, w, h = self.roi.to_absolute(width, height)

        # Ensure ROI is within image bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))

        roi_image = image[y:y+h, x:x+w]
        self.logger.debug(f"Extracted ROI: ({x}, {y}, {w}, {h})")

        return roi_image

    def preprocess_roi(self, roi_image: np.ndarray) -> np.ndarray:
        """
        Preprocess ROI for better OCR accuracy

        Args:
            roi_image: ROI image

        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image.copy()

        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Noise removal
        denoised = cv2.medianBlur(thresh, 3)

        return denoised

    def recognize_with_tesseract(self, roi_image: np.ndarray) -> List[OCRResult]:
        """
        Recognize text using Tesseract

        Args:
            roi_image: ROI image

        Returns:
            List of OCRResult objects
        """
        results = []

        try:
            # Get detailed data
            data = pytesseract.image_to_data(
                roi_image,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )

            # Process results
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else -1

                if text and conf > 30:  # Confidence threshold
                    results.append(OCRResult(
                        text=text,
                        confidence=conf,
                        bbox=(
                            data['left'][i],
                            data['top'][i],
                            data['width'][i],
                            data['height'][i]
                        )
                    ))

        except Exception as e:
            self.logger.error(f"Tesseract OCR failed: {e}")

        return results

    def recognize_with_easyocr(self, roi_image: np.ndarray) -> List[OCRResult]:
        """
        Recognize text using EasyOCR

        Args:
            roi_image: ROI image

        Returns:
            List of OCRResult objects
        """
        results = []

        try:
            # EasyOCR expects RGB
            if len(roi_image.shape) == 3:
                roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
            else:
                roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2RGB)

            # Run OCR with optimized parameters
            detections = self.easyocr_reader.readtext(
                roi_rgb,
                # Higher confidence threshold for page numbers
                text_threshold=0.7,
                low_text=0.4,
                link_threshold=0.4,
                # Only allow numbers and common page number characters
                allowlist='0123456789./- Page',
                # Use paragraph mode for better grouping
                paragraph=False,  # Changed to False to get individual results
                # Detailed output (bbox, text, confidence)
                detail=1
            )

            for detection in detections:
                # EasyOCR returns: [bbox, text, confidence] or just [text] if detail=0
                if len(detection) == 3:
                    bbox, text, conf = detection

                    # Higher confidence threshold for page numbers (70%)
                    if conf > 0.7:
                        results.append(OCRResult(
                            text=text.strip(),
                            confidence=conf * 100,
                            bbox=tuple(map(int, bbox[0] + bbox[2]))  # Convert to (x, y, w, h)
                        ))
                    else:
                        self.logger.debug(f"Low confidence detection rejected: '{text}' (conf: {conf:.2f})")
                elif len(detection) == 2:
                    # Sometimes EasyOCR returns [bbox, text] without confidence
                    bbox, text = detection
                    results.append(OCRResult(
                        text=text.strip(),
                        confidence=50.0,  # Default confidence
                        bbox=tuple(map(int, bbox[0] + bbox[2]))
                    ))

            self.logger.debug(f"EasyOCR found {len(results)} high-confidence results")

        except Exception as e:
            self.logger.error(f"EasyOCR failed: {e}")
            self.logger.error(traceback.format_exc())

        return results

    def extract_page_numbers(self, text: str) -> List[int]:
        """
        Extract page numbers from text using regex

        Args:
            text: Input text

        Returns:
            List of extracted page numbers
        """
        page_numbers = []

        for pattern in self.PAGE_NUMBER_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract numbers from match
                numbers = re.findall(r'\d+', match)
                for num_str in numbers:
                    try:
                        num = int(num_str)
                        if 0 < num < 10000:  # Reasonable page number range
                            page_numbers.append(num)
                    except ValueError:
                        continue

        # Remove duplicates while preserving order
        unique_pages = []
        seen = set()
        for page in page_numbers:
            if page not in seen:
                unique_pages.append(page)
                seen.add(page)

        return unique_pages

    def recognize_page_number(self, image: np.ndarray) -> Optional[int]:
        """
        Recognize page number in image

        Args:
            image: Input image (BGR format)

        Returns:
            Detected page number or None
        """
        # Extract ROI
        roi_image = self.extract_roi(image)

        # Preprocess
        processed = self.preprocess_roi(roi_image)

        # Run OCR
        if self.ocr_engine == 'pytesseract':
            ocr_results = self.recognize_with_tesseract(processed)
        else:
            ocr_results = self.recognize_with_easyocr(processed)

        # Extract text
        all_text = ' '.join([result.text for result in ocr_results])

        # Extract page numbers
        page_numbers = self.extract_page_numbers(all_text)

        # Return first valid page number
        if page_numbers:
            self.logger.info(f"Detected page number: {page_numbers[0]} (from text: '{all_text}')")
            return page_numbers[0]

        self.logger.debug(f"No page number detected in text: '{all_text}'")
        return None

    def batch_recognize(self, images: List[np.ndarray]) -> List[Optional[int]]:
        """
        Recognize page numbers for multiple images

        Args:
            images: List of input images

        Returns:
            List of detected page numbers
        """
        results = []

        from tqdm import tqdm
        for img in tqdm(images, desc="Recognizing page numbers"):
            page_num = self.recognize_page_number(img)
            results.append(page_num)

        return results

    def batch_recognize_parallel(self, images: List[np.ndarray]) -> List[Optional[int]]:
        """
        Recognize page numbers for multiple images using parallel processing

        Args:
            images: List of input images

        Returns:
            List of detected page numbers
        """
        if not images:
            return []

        self.logger.info(f"Starting parallel OCR with {self.max_workers} threads for {len(images)} images")

        # Use ThreadPoolExecutor (OCR calls external tesseract process, so I/O bound)
        # ThreadPoolExecutor avoids Windows pickle serialization issues with ProcessPoolExecutor
        from tqdm import tqdm

        def recognize_single(img):
            """Wrapper to recognize single image with exception handling"""
            try:
                return self.recognize_page_number(img)
            except Exception as e:
                self.logger.warning(f"OCR failed: {e}")
                return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(recognize_single, images),
                total=len(images),
                desc="Recognizing page numbers (parallel)"
            ))

        return results

    def update_slides_with_page_numbers(self,
                                        slides: List[Dict],
                                        images: List[np.ndarray],
                                        use_parallel: bool = True) -> List[Dict]:
        """
        Update slide metadata with recognized page numbers

        Args:
            slides: List of slide metadata
            images: List of slide images
            use_parallel: Whether to use parallel processing

        Returns:
            Updated slide metadata
        """
        if len(slides) != len(images):
            self.logger.warning(f"Mismatch: {len(slides)} slides, {len(images)} images")

        # Choose processing method
        if use_parallel and len(images) > 1:
            page_numbers = self.batch_recognize_parallel(images)
        else:
            page_numbers = self.batch_recognize(images)

        for i, (slide, page_num) in enumerate(zip(slides, page_numbers)):
            slide['page_number'] = page_num
            if page_num:
                self.logger.debug(f"Slide {i}: page {page_num}")
            else:
                self.logger.debug(f"Slide {i}: no page number")

        return slides


class PageNumberSorter:
    """Sort slides by page numbers"""

    def __init__(self):
        """Initialize sorter"""
        self.logger = logging.getLogger(__name__)

    def detect_conflicts(self, slides: List[Dict]) -> List[Dict]:
        """
        Detect page number conflicts

        Args:
            slides: List of slide metadata

        Returns:
            List of conflict information
        """
        conflicts = []
        page_counts = {}

        for i, slide in enumerate(slides):
            page_num = slide.get('page_number')
            if page_num is not None:
                if page_num not in page_counts:
                    page_counts[page_num] = []
                page_counts[page_num].append(i)

        # Find duplicates
        for page_num, indices in page_counts.items():
            if len(indices) > 1:
                conflicts.append({
                    'page_number': page_num,
                    'slide_indices': indices,
                    'count': len(indices)
                })

        return conflicts

    def resolve_conflicts(self, slides: List[Dict], strategy: str = 'first') -> List[Dict]:
        """
        Resolve page number conflicts

        Args:
            slides: List of slide metadata
            strategy: 'first', 'last', 'timestamp', 'duplicate'

        Returns:
            Updated slide metadata
        """
        conflicts = self.detect_conflicts(slides)

        if not conflicts:
            return slides

        self.logger.info(f"Resolving {len(conflicts)} page number conflicts using strategy: {strategy}")

        if strategy == 'duplicate':
            # Keep duplicates with suffix
            for conflict in conflicts:
                for idx, slide_idx in enumerate(conflict['slide_indices']):
                    slides[slide_idx]['page_number'] = conflict['page_number'] + idx * 0.1

        elif strategy == 'first':
            # Keep first occurrence
            for conflict in conflicts:
                for slide_idx in conflict['slide_indices'][1:]:
                    slides[slide_idx]['page_number'] = None

        elif strategy == 'timestamp':
            # Sort by timestamp
            for conflict in conflicts:
                indices = conflict['slide_indices']
                indices.sort(key=lambda i: slides[i]['timestamp'])
                for slide_idx in indices[1:]:
                    slides[slide_idx]['page_number'] = None

        elif strategy == 'last':
            # Keep last occurrence
            for conflict in conflicts:
                for slide_idx in conflict['slide_indices'][:-1]:
                    slides[slide_idx]['page_number'] = None

        return slides

    def sort_slides(self, slides: List[Dict], conflicts: bool = True) -> List[Dict]:
        """
        Sort slides by page number

        Args:
            slides: List of slide metadata
            conflicts: Whether to resolve conflicts

        Returns:
            Sorted slide metadata
        """
        if conflicts:
            slides = self.resolve_conflicts(slides, strategy='first')

        # Separate slides with and without page numbers
        slides_with_pages = [s for s in slides if s.get('page_number') is not None]
        slides_without_pages = [s for s in slides if s.get('page_number') is None]

        # Sort slides with page numbers
        slides_with_pages.sort(key=lambda x: x['page_number'])

        # Combine: slides with pages first, then without
        sorted_slides = slides_with_pages + slides_without_pages

        self.logger.info(f"Sorted {len(sorted_slides)} slides")
        return sorted_slides
