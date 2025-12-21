"""
Selection Module

Manages slide selection state and UI for Gradio interface:
- Track selected/unselected slides
- Gallery display with thumbnails
- Select all/none operations
- Sort by page number or timestamp
- Statistics and filtering
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from pathlib import Path
import gradio as gr
import logging


class SlideSelection:
    """Manages slide selection state"""

    def __init__(self):
        """Initialize selection manager"""
        self.slides: List[Dict] = []
        self.selected_indices: Set[int] = set()
        self.sort_by_page: bool = False
        self.logger = logging.getLogger(__name__)

    def set_slides(self, slides: List[Dict]) -> None:
        """
        Set slides data

        Args:
            slides: List of slide dictionaries
        """
        self.slides = slides.copy()
        # Initialize all as selected by default
        self.selected_indices = set(range(len(slides)))
        self.logger.info(f"Set {len(slides)} slides for selection")

    def get_slides(self) -> List[Dict]:
        """
        Get current slides

        Returns:
            List of slide dictionaries
        """
        return self.slides.copy()

    def get_selected_slides(self) -> List[Dict]:
        """
        Get only selected slides

        Returns:
            List of selected slide dictionaries
        """
        return [self.slides[i] for i in self.selected_indices if i < len(self.slides)]

    def select_all(self) -> List[Dict]:
        """
        Select all slides

        Returns:
            Updated gallery data
        """
        self.selected_indices = set(range(len(self.slides)))
        self.logger.info("Selected all slides")
        return self._to_gallery_data()

    def select_none(self) -> List[Dict]:
        """
        Deselect all slides

        Returns:
            Updated gallery data
        """
        self.selected_indices = set()
        self.logger.info("Deselected all slides")
        return self._to_gallery_data()

    def toggle_selection(self, index: int) -> List[Dict]:
        """
        Toggle selection of a slide

        Args:
            index: Slide index

        Returns:
            Updated gallery data
        """
        if index in self.selected_indices:
            self.selected_indices.remove(index)
        else:
            self.selected_indices.add(index)
        return self._to_gallery_data()

    def set_selection(self, indices: List[int]) -> List[Dict]:
        """
        Set selection for specific indices

        Args:
            indices: List of indices to select (can be strings or integers)

        Returns:
            Updated gallery data
        """
        # Convert to integers if they are strings
        int_indices = []
        for idx in indices:
            if isinstance(idx, str):
                try:
                    int_indices.append(int(idx))
                except ValueError:
                    continue
            elif isinstance(idx, int):
                int_indices.append(idx)
            else:
                continue

        self.selected_indices = set(int_indices)
        self.logger.info(f"Selected {len(int_indices)} slides")
        return self._to_gallery_data()

    def get_selection_state(self) -> Tuple[List[bool], int, int]:
        """
        Get selection state for UI

        Returns:
            Tuple of (selected_mask, selected_count, total_count)
        """
        selected_mask = [i in self.selected_indices for i in range(len(self.slides))]
        selected_count = len(self.selected_indices)
        total_count = len(self.slides)
        return selected_mask, selected_count, total_count

    def sort_by_timestamp(self) -> List[Dict]:
        """
        Sort slides by timestamp

        Returns:
            Updated gallery data
        """
        # Store selection state
        old_selection = {self.slides[i]['timestamp'] for i in self.selected_indices}

        # Sort slides by timestamp
        self.slides.sort(key=lambda x: x['timestamp'])

        # Restore selection based on timestamps
        self.selected_indices = {
            i for i, slide in enumerate(self.slides)
            if slide['timestamp'] in old_selection
        }

        self.sort_by_page = False
        self.logger.info("Sorted slides by timestamp")
        return self._to_gallery_data()

    def sort_by_page_number(self) -> List[Dict]:
        """
        Sort slides by page number

        Returns:
            Updated gallery data
        """
        # Only sort if page numbers are available
        slides_with_pages = [s for s in self.slides if s.get('page_number') is not None]
        if not slides_with_pages:
            self.logger.warning("No page numbers available for sorting")
            return self._to_gallery_data()

        # Store selection state
        old_selection = {self.slides[i].get('page_number') for i in self.selected_indices}

        # Sort slides by page number (placing slides without page numbers at the end)
        self.slides.sort(key=lambda x: (x.get('page_number') is None, x.get('page_number')))

        # Restore selection based on page numbers
        self.selected_indices = {
            i for i, slide in enumerate(self.slides)
            if slide.get('page_number') in old_selection
        }

        self.sort_by_page = True
        self.logger.info("Sorted slides by page number")
        return self._to_gallery_data()

    def invert_selection(self) -> List[Dict]:
        """
        Invert current selection

        Returns:
            Updated gallery data
        """
        all_indices = set(range(len(self.slides)))
        self.selected_indices = all_indices - self.selected_indices
        self.logger.info("Inverted selection")
        return self._to_gallery_data()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get selection statistics

        Returns:
            Dictionary with statistics
        """
        total = len(self.slides)
        selected = len(self.selected_indices)
        pages_with_numbers = sum(1 for s in self.slides if s.get('page_number') is not None)

        return {
            'total_slides': total,
            'selected_slides': selected,
            'unselected_slides': total - selected,
            'selection_ratio': selected / total if total > 0 else 0,
            'slides_with_page_numbers': pages_with_numbers,
            'slides_without_page_numbers': total - pages_with_numbers,
            'sort_mode': 'page_number' if self.sort_by_page else 'timestamp'
        }

    def _to_gallery_data(self) -> List[Tuple[str, str]]:
        """
        Convert slides to gallery format (path, caption)

        Returns:
            List of (thumbnail_path, caption) tuples
        """
        gallery_data = []
        for i, slide in enumerate(self.slides):
            # Get thumbnail path
            thumb_path = slide.get('thumbnail_path', '')
            if not thumb_path or not Path(thumb_path).exists():
                # Placeholder for missing thumbnail
                thumb_path = "https://via.placeholder.com/400x300?text=No+Thumbnail"

            # Create caption
            caption_parts = [f"Slide {i+1}"]
            if slide.get('timestamp') is not None:
                caption_parts.append(f"@{slide['timestamp']:.2f}s")
            if slide.get('page_number') is not None:
                caption_parts.append(f"Page {slide['page_number']}")

            # Add selection indicator
            if i in self.selected_indices:
                caption_parts.append("✓ Selected")

            caption = " | ".join(caption_parts)

            gallery_data.append((thumb_path, caption))

        return gallery_data

    def export_selection_list(self, output_path: Optional[str] = None) -> str:
        """
        Export selection as list of file paths

        Args:
            output_path: Optional path to save the list

        Returns:
            String with selection list
        """
        selected_slides = self.get_selected_slides()

        lines = []
        for i, slide in enumerate(selected_slides):
            if slide.get('thumbnail_path'):
                lines.append(slide['thumbnail_path'])

        result = '\n'.join(lines)

        if output_path:
            Path(output_path).write_text(result, encoding='utf-8')
            self.logger.info(f"Exported selection list to {output_path}")

        return result

    def to_dict(self) -> Dict:
        """
        Serialize selection state to dictionary

        Returns:
            Selection state dictionary
        """
        return {
            'slides': self.slides,
            'selected_indices': list(self.selected_indices),
            'sort_by_page': self.sort_by_page
        }

    def from_dict(self, data: Dict) -> None:
        """
        Load selection state from dictionary

        Args:
            data: Selection state dictionary
        """
        self.slides = data.get('slides', [])
        self.selected_indices = set(data.get('selected_indices', []))
        self.sort_by_page = data.get('sort_by_page', False)
        self.logger.info(f"Loaded selection state: {len(self.slides)} slides")


class SelectionUI:
    """Gradio UI components for selection"""

    def __init__(self, selection: SlideSelection):
        """
        Initialize UI components

        Args:
            selection: SlideSelection instance
        """
        self.selection = selection
        self.logger = logging.getLogger(__name__)

    def create_ui(self):
        """Create Gradio UI components"""
        with gr.Tab("筛选幻灯片", id="tab_selection"):
            gr.Markdown("## 选择要导出的幻灯片")

            # Statistics display
            with gr.Row():
                stats_text = gr.Textbox(
                    label="统计信息",
                    lines=3,
                    interactive=False
                )

            # Gallery
            with gr.Row():
                gallery = gr.Gallery(
                    label="幻灯片缩略图",
                    columns=4,
                    rows=2,
                    height="auto",
                    interactive=True,
                    show_share_button=False
                )

            # Controls
            with gr.Row():
                select_all_btn = gr.Button("全选", variant="primary")
                select_none_btn = gr.Button("取消全选")
                invert_btn = gr.Button("反选")
                sort_time_btn = gr.Button("按时间排序")
                sort_page_btn = gr.Button("按页码排序")

            # Hidden state for tracking
            selected_indices_state = gr.State(value=[])
            gallery_state = gr.State(value=[])

        return {
            'gallery': gallery,
            'stats_text': stats_text,
            'select_all_btn': select_all_btn,
            'select_none_btn': select_none_btn,
            'invert_btn': invert_btn,
            'sort_time_btn': sort_time_btn,
            'sort_page_btn': sort_page_btn,
            'selected_indices_state': selected_indices_state,
            'gallery_state': gallery_state
        }

    def update_gallery(self) -> Tuple[List[Tuple], str, List[int]]:
        """
        Update gallery display

        Returns:
            Tuple of (gallery_data, stats_text, selected_indices)
        """
        gallery_data = self.selection._to_gallery_data()
        stats = self.selection.get_statistics()

        stats_text = (
            f"总计: {stats['total_slides']} 张幻灯片\n"
            f"已选: {stats['selected_slides']} 张\n"
            f"未选: {stats['unselected_slides']} 张\n"
            f"有页码: {stats['slides_with_page_numbers']} 张\n"
            f"排序方式: {stats['sort_mode']}"
        )

        selected_indices = list(self.selection.selected_indices)

        return gallery_data, stats_text, selected_indices

    def handle_gallery_select(self,
                             evt: gr.SelectData,
                             gallery_data: List[Tuple],
                             selected_indices: List[int]) -> Tuple[List[Tuple], List[int], str]:
        """
        Handle gallery selection event

        Args:
            evt: SelectData event
            gallery_data: Current gallery data
            selected_indices: Current selected indices

        Returns:
            Tuple of (updated_gallery_data, updated_selected_indices, message)
        """
        # Toggle selection
        index = evt.index

        if index in selected_indices:
            selected_indices.remove(index)
            message = f"取消选择幻灯片 {index + 1}"
        else:
            selected_indices.append(index)
            message = f"已选择幻灯片 {index + 1}"

        # Update selection state
        self.selection.selected_indices = set(selected_indices)

        # Update gallery data with new selection states
        updated_gallery_data = self.selection._to_gallery_data()

        self.logger.info(f"Gallery selection changed: {message}")
        return updated_gallery_data, selected_indices, message
