#!/usr/bin/env python3
"""
Main entry point for JL Video to PPT Converter

Usage:
    python main.py                          # Launch Gradio UI
    python main.py video.mp4               # Process video via CLI
    python main.py video.mp4 --output ./out  # Process with custom output dir
    python main.py video.mp4 --preset high_quality  # Use custom preset
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import VideoProcessingPipeline, create_gradio_interface
import argparse


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="JL Video to PPT Converter - Convert video presentations to slides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch Gradio UI
  python main.py

  # Process video via CLI
  python main.py presentation.mp4

  # Process with custom output directory
  python main.py presentation.mp4 --output ./slides_output

  # Use high quality preset
  python main.py presentation.mp4 --preset high_quality

  # Process with time range (detect from 1:30 to 5:00)
  python main.py presentation.mp4 --start-time 1:30 --end-time 5:00

  # Process and export only
  python main.py presentation.mp4 --export-only
        """
    )

    parser.add_argument("video_path", nargs="?", help="Path to video file")
    parser.add_argument(
        "-o", "--output",
        default="OUTPUT",
        help="Output directory (default: OUTPUT)"
    )
    parser.add_argument(
        "-p", "--preset",
        default="default",
        help="Preset to use (default: default)"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch Gradio UI"
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip parsing, only export previously processed slides"
    )
    parser.add_argument(
        "--start-time",
        help="Start time for detection (format: HH:MM:SS or MM:SS or SS)"
    )
    parser.add_argument(
        "--end-time",
        help="End time for detection (format: HH:MM:SS or MM:SS or SS)"
    )

    args = parser.parse_args()

    # If no arguments or --gui specified, launch UI
    if args.gui or len(sys.argv) == 1:
        print("Launching Gradio UI...")
        print("Access the interface at: http://localhost:7930")
        pipeline = VideoProcessingPipeline()
        demo = create_gradio_interface(pipeline)
        demo.launch(
            server_name="0.0.0.0",
            server_port=7930,
            show_error=True
        )
        return

    # Otherwise, process video via CLI
    if not args.video_path:
        parser.print_help()
        sys.exit(1)

    # Check if input is a URL (skip file existence check for URLs)
    is_url = args.video_path.startswith(('http://', 'https://'))
    if not is_url and not Path(args.video_path).exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    # Generate output directory name based on video filename and timestamp
    if is_url:
        # For URLs, extract filename from URL
        # Remove query parameters and fragments, then extract filename without extension
        url_filename = args.video_path.split('/')[-1].split('?')[0].split('#')[0]
        # Remove video extensions to get clean name
        video_name = url_filename
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']:
            if video_name.lower().endswith(ext.lower()):
                video_name = video_name[:-len(ext)]
                break
        video_name = video_name or 'video'
    else:
        # For local files, use existing logic
        video_path = Path(args.video_path)
        video_name = video_path.stem  # filename without extension

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If user specified a custom output directory, append the timestamped folder to it
    if args.output != "OUTPUT":
        base_output = Path(args.output)
        if not base_output.is_absolute():
            base_output = Path.cwd() / base_output
        final_output = base_output / f"{video_name}_{timestamp}"
    else:
        final_output = Path(args.output) / f"{video_name}_{timestamp}"

    final_output.mkdir(parents=True, exist_ok=True)

    print(f"Processing video: {args.video_path}")
    print(f"Output directory: {final_output}")
    print(f"Preset: {args.preset}")
    print("-" * 50)

    # Initialize pipeline
    pipeline = VideoProcessingPipeline()

    # Parse video
    if not args.export_only:
        # Parse time range arguments
        from src.video_io import parse_time_string
        start_time = parse_time_string(args.start_time) if args.start_time else None
        end_time = parse_time_string(args.end_time) if args.end_time else None

        # Validate time range
        if start_time is not None and start_time < 0:
            print(f"Error: Start time cannot be negative")
            sys.exit(1)
        if end_time is not None and end_time < 0:
            print(f"Error: End time cannot be negative")
            sys.exit(1)
        if start_time is not None and end_time is not None and start_time >= end_time:
            print(f"Error: Start time must be less than end time")
            sys.exit(1)

        # Display time range info
        if start_time is not None or end_time is not None:
            from src.video_io import format_seconds_to_time
            start_str = format_seconds_to_time(start_time) if start_time is not None else "00:00"
            end_str = format_seconds_to_time(end_time) if end_time is not None else "end"
            print(f"Detection time range: {start_str} to {end_str}")

        success, message, result = pipeline.parse_video(
            args.video_path,
            str(final_output),
            args.preset,
            start_time,
            end_time
        )

        if not success:
            print(f"Error: {message}")
            sys.exit(1)

        print(f"\n[SUCCESS] {message}")
        print(f"\nVideo Information:")
        if result and 'metadata' in result:
            metadata = result['metadata']
            print(f"  Resolution: {metadata['width']}x{metadata['height']}")
            print(f"  Duration: {metadata['duration']:.2f}s")
            print(f"  FPS: {metadata['fps']:.2f}")
            print(f"  Total slides: {len(result['slides'])}")
    else:
        print("Skipping parsing (export-only mode)")

    # Export slides
    print("\n" + "-" * 50)
    print("Exporting slides...")

    success, message, results = pipeline.export_slides()

    if not success:
        print(f"Error: {message}")
        sys.exit(1)

    print(f"\n[SUCCESS] Export complete!")
    print(f"\n{message}")

    print("\n" + "=" * 50)
    print("Processing complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
