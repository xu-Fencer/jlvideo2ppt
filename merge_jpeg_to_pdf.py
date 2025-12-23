#!/usr/bin/env python3
"""
合并当前目录下的所有JPEG文件为一个PDF

按文件名排序后逐个添加到PDF中
"""

import sys
from pathlib import Path
from datetime import datetime
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


def merge_jpeg_to_pdf(
    input_dir: str = ".",
    output_filename: str = None,
    output_dir: str = "."
) -> str:
    """
    合并当前目录下的所有JPEG文件为一个PDF

    Args:
        input_dir: 输入目录（包含JPEG文件）
        output_filename: 输出PDF文件名（可选）
        output_dir: 输出目录

    Returns:
        输出PDF文件的路径
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有JPEG文件（使用不区分大小写的模式避免重复）
    jpeg_files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
        jpeg_files.extend(input_path.glob(ext))

    # 去重（因为glob可能返回重复文件）并排序
    jpeg_files = sorted(list(set(jpeg_files)))

    if not jpeg_files:
        print(f"在目录 {input_path} 中未找到JPEG文件")
        sys.exit(1)

    print(f"找到 {len(jpeg_files)} 个JPEG文件")
    print(f"文件列表:")
    for i, f in enumerate(jpeg_files, 1):
        print(f"  {i}. {f.name}")

    # 生成输出文件名
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"merged_{timestamp}.pdf"

    if not output_filename.endswith('.pdf'):
        output_filename += '.pdf'

    output_pdf_path = output_path / output_filename

    try:
        # 创建PDF
        c = canvas.Canvas(str(output_pdf_path))

        for i, jpeg_file in enumerate(jpeg_files):
            try:
                # 打开图片获取尺寸
                with Image.open(jpeg_file) as img:
                    img_width, img_height = img.size

                # 设置页面尺寸为图片尺寸
                c.setPageSize((img_width, img_height))

                # 添加图片到PDF
                c.drawImage(
                    str(jpeg_file),
                    0, 0,
                    width=img_width,
                    height=img_height,
                    preserveAspectRatio=True
                )

                # 添加新页面（除了最后一个图片）
                if i < len(jpeg_files) - 1:
                    c.showPage()

                print(f"已添加: {jpeg_file.name} ({img_width}x{img_height})")

            except Exception as e:
                print(f"处理文件 {jpeg_file.name} 时出错: {e}")
                continue

        # 保存PDF
        c.save()

        print(f"\n✅ PDF合并完成!")
        print(f"📄 输出文件: {output_pdf_path}")
        print(f"📊 包含 {len(jpeg_files)} 页")

        return str(output_pdf_path)

    except Exception as e:
        print(f"❌ 合并PDF失败: {e}")
        # 清理失败的PDF
        if output_pdf_path.exists():
            output_pdf_path.unlink()
        sys.exit(1)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="合并当前目录下的所有JPEG文件为一个PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 合并当前目录下的所有JPEG文件
  python merge_jpeg_to_pdf.py

  # 合并指定目录下的JPEG文件
  python merge_jpeg_to_pdf.py -i /path/to/images

  # 指定输出文件名
  python merge_jpeg_to_pdf.py -o my_presentation.pdf

  # 指定输出目录
  python merge_jpeg_to_pdf.py -d /path/to/output
        """
    )

    parser.add_argument(
        '-i', '--input',
        default='.',
        help='输入目录路径（默认: 当前目录）'
    )

    parser.add_argument(
        '-o', '--output',
        help='输出PDF文件名（默认: merged_YYYYMMDD_HHMMSS.pdf）'
    )

    parser.add_argument(
        '-d', '--dir',
        default='.',
        help='输出目录路径（默认: 当前目录）'
    )

    args = parser.parse_args()

    # 检查依赖
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        print("❌ 缺少依赖: reportlab")
        print("请运行: pip install reportlab")
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        print("❌ 缺少依赖: Pillow")
        print("请运行: pip install Pillow")
        sys.exit(1)

    # 执行合并
    merge_jpeg_to_pdf(
        input_dir=args.input,
        output_filename=args.output,
        output_dir=args.dir
    )


if __name__ == "__main__":
    main()
