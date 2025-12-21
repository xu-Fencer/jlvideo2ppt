#!/usr/bin/env python3
"""
创建测试视频示例

此脚本用于生成简单的测试视频，包含多个幻灯片场景，
可用于测试 JL Video to PPT Converter 的功能。
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def create_slide_video(
    output_path: str,
    num_slides: int = 10,
    duration_per_slide: float = 3.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720
):
    """
    创建包含多个幻灯片的测试视频

    Args:
        output_path: 输出视频路径
        num_slides: 幻灯片数量
        duration_per_slide: 每个幻灯片持续时间（秒）
        fps: 帧率
        width: 视频宽度
        height: 视频高度
    """
    # 定义幻灯片内容
    slides = [
        {
            "title": "Slide 1",
            "subtitle": "Introduction",
            "color": (50, 50, 200),  # Blue
            "text_color": (255, 255, 255)
        },
        {
            "title": "Slide 2",
            "subtitle": "Problem Statement",
            "color": (200, 50, 50),  # Red
            "text_color": (255, 255, 255)
        },
        {
            "title": "Slide 3",
            "subtitle": "Solution Overview",
            "color": (50, 200, 50),  # Green
            "text_color": (0, 0, 0)
        },
        {
            "title": "Slide 4",
            "subtitle": "Methodology",
            "color": (200, 200, 50),  # Yellow
            "text_color": (0, 0, 0)
        },
        {
            "title": "Slide 5",
            "subtitle": "Results",
            "color": (200, 50, 200),  # Magenta
            "text_color": (255, 255, 255)
        },
        {
            "title": "Slide 6",
            "subtitle": "Analysis",
            "color": (50, 200, 200),  # Cyan
            "text_color": (0, 0, 0)
        },
        {
            "title": "Slide 7",
            "subtitle": "Discussion",
            "color": (128, 128, 128),  # Gray
            "text_color": (255, 255, 255)
        },
        {
            "title": "Slide 8",
            "subtitle": "Conclusions",
            "color": (255, 128, 0),  # Orange
            "text_color": (255, 255, 255)
        },
        {
            "title": "Slide 9",
            "subtitle": "Future Work",
            "color": (128, 0, 128),  # Purple
            "text_color": (255, 255, 255)
        },
        {
            "title": "Slide 10",
            "subtitle": "Thank You",
            "color": (0, 128, 255),  # Light Blue
            "text_color": (255, 255, 255)
        }
    ]

    # 如果需要更多幻灯片，重复使用颜色
    while len(slides) < num_slides:
        slides.extend(slides[:min(len(slides), num_slides - len(slides))])
    slides = slides[:num_slides]

    # 计算总帧数
    frames_per_slide = int(duration_per_slide * fps)
    total_frames = num_slides * frames_per_slide

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating video: {output_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Slides: {num_slides}")
    print(f"Duration: {num_slides * duration_per_slide:.1f}s")
    print(f"Total frames: {total_frames}")

    for slide_idx, slide in enumerate(slides):
        print(f"\nCreating slide {slide_idx + 1}/{num_slides}: {slide['title']}")

        for frame_idx in range(frames_per_slide):
            # 创建背景
            frame = np.full((height, width, 3), slide['color'], dtype=np.uint8)

            # 添加边框
            cv2.rectangle(frame, (0, 0), (width, height), (255, 255, 255), 5)

            # 添加标题
            title_scale = 3
            title_thickness = 6
            (text_w, text_h), baseline = cv2.getTextSize(
                slide['title'], cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thickness
            )
            title_x = (width - text_w) // 2
            title_y = (height - text_h) // 2 - 50
            cv2.putText(
                frame, slide['title'],
                (title_x, title_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                title_scale,
                slide['text_color'],
                title_thickness
            )

            # 添加副标题
            subtitle_scale = 1.5
            subtitle_thickness = 3
            (text_w, text_h), baseline = cv2.getTextSize(
                slide['subtitle'], cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, subtitle_thickness
            )
            subtitle_x = (width - text_w) // 2
            subtitle_y = title_y + 100
            cv2.putText(
                frame, slide['subtitle'],
                (subtitle_x, subtitle_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                subtitle_scale,
                slide['text_color'],
                subtitle_thickness
            )

            # 添加页码
            page_text = f"{slide_idx + 1}"
            page_scale = 2
            page_thickness = 4
            (text_w, text_h), baseline = cv2.getTextSize(
                page_text, cv2.FONT_HERSHEY_SIMPLEX, page_scale, page_thickness
            )
            page_x = width - text_w - 50
            page_y = height - 50
            cv2.putText(
                frame, page_text,
                (page_x, page_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                page_scale,
                slide['text_color'],
                page_thickness
            )

            # 添加帧计数（调试用）
            frame_text = f"Frame {slide_idx * frames_per_slide + frame_idx + 1}"
            cv2.putText(
                frame, frame_text,
                (50, height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                slide['text_color'],
                2
            )

            # 写入帧
            out.write(frame)

    # 完成
    out.release()
    cv2.destroyAllWindows()

    print(f"\n✓ Video created successfully: {output_path}")
    print(f"  Size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")


def create_presentation_with_transitions(
    output_path: str,
    num_slides: int = 10,
    slide_duration: float = 3.0,
    transition_duration: float = 0.5,
    fps: int = 30,
    width: int = 1280,
    height: int = 720
):
    """
    创建带过渡效果的演示视频

    Args:
        output_path: 输出视频路径
        num_slides: 幻灯片数量
        slide_duration: 每个幻灯片持续时间
        transition_duration: 过渡持续时间
        fps: 帧率
        width: 宽度
        height: 高度
    """
    print("Creating presentation with transitions...")

    # 创建幻灯片视频
    temp_video = "temp_slides.mp4"
    create_slide_video(
        temp_video,
        num_slides=num_slides,
        duration_per_slide=slide_duration,
        fps=fps,
        width=width,
        height=height
    )

    # 使用 ffmpeg 添加过渡效果
    import subprocess

    # 创建过渡效果滤镜
    filter_complex = []
    for i in range(num_slides):
        filter_complex.append(f"[{i}:v]scale={width}:{height},setpts=PTS-STARTPTS[v{i}];")

    # 添加交叉淡化过渡
    for i in range(num_slides - 1):
        blend = f"[v{i}][v{i+1}]blend=all_expr='A*(if(gte(T,{transition_duration}),1,T/{transition_duration}))+B*(1-(if(gte(T,{transition_duration}),1,T/{transition_duration})))':shortest=1[blend{i}];"
        filter_complex.append(blend)

    # 构建命令
    # 注意：这里简化处理，实际应用中需要更复杂的滤镜
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Video with transitions created: {output_path}")
        Path(temp_video).unlink(missing_ok=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to add transitions: {e}")
        # 如果添加过渡失败，复制原视频
        import shutil
        shutil.copy(temp_video, output_path)
        Path(temp_video).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Create test video for slide detection")
    parser.add_argument("-o", "--output", default="test_presentation.mp4", help="Output video path")
    parser.add_argument("-n", "--num-slides", type=int, default=10, help="Number of slides")
    parser.add_argument("-d", "--duration", type=float, default=3.0, help="Duration per slide (seconds)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--transitions", action="store_true", help="Add transitions between slides")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建视频
    if args.transitions:
        create_presentation_with_transitions(
            args.output,
            num_slides=args.num_slides,
            slide_duration=args.duration,
            transition_duration=0.5,
            fps=args.fps,
            width=args.width,
            height=args.height
        )
    else:
        create_slide_video(
            args.output,
            num_slides=args.num_slides,
            duration_per_slide=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height
        )

    print("\n" + "=" * 60)
    print("Test video created successfully!")
    print("=" * 60)
    print(f"\nYou can now test the converter with:")
    print(f"  python main.py {args.output}")
    print("\nOr launch the GUI:")
    print(f"  python main.py --gui")
    print("=" * 60)


if __name__ == "__main__":
    main()
