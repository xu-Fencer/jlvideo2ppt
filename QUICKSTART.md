# 快速开始指南

欢迎使用 JL Video to PPT Converter！本指南将帮助您快速上手。

## 安装

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 安装系统依赖

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg tesseract-ocr
```

**Windows:**
- 下载并安装 ffmpeg: https://ffmpeg.org/download.html
- 下载并安装 Tesseract: https://github.com/UB-Mannheim/tesseract/wiki

**macOS:**
```bash
brew install ffmpeg tesseract
```

## 快速体验

### 1. 创建测试视频

```bash
cd examples
python create_test_video.py -o ../test_video.mp4
```

这将创建一个包含 10 张幻灯片的测试视频。

### 2. 启动图形界面

```bash
# 在项目根目录运行
python main.py --gui
```

然后在浏览器中打开 `http://localhost:7930`

### 3. 处理视频

1. **上传视频**: 点击"上传视频文件"或输入路径
2. **选择预设**: 建议先用 "fast" 预设测试
3. **开始解析**: 点击"开始解析"按钮
4. **筛选幻灯片**: 在第二个标签页中选择要保留的幻灯片
5. **识别页码**（可选）: 在第三个标签页中识别页码
6. **导出**: 在第四个标签页中选择格式并导出

## 命令行模式

```bash
# 基本用法
python main.py test_video.mp4

# 指定输出目录
python main.py test_video.mp4 --output ./slides_output

# 使用高质量预设
python main.py test_video.mp4 --preset high_quality
```

## 查看结果

处理完成后，结果将保存在时间戳目录中：

```
OUTPUT/
└── video_20241221_143022/    # 时间戳目录（每次处理不同）
    ├── images/               # 原始分辨率图片
    ├── thumbs/               # 缩略图缓存
    ├── pages/                # OCR识别的页码图片
    │   ├── slide_1.jpg       # 页码1
    │   ├── slide_2.jpg       # 页码2
    │   ├── slide_2_重复1.jpg # 重复页码2
    │   └── slide_3.jpg       # 页码3
    ├── exported_images/      # 导出的图片文件（如果选择JPEG格式）
    ├── slides_*.pdf          # 导出的 PDF 文件
    ├── thumbs/               # 缩略图缓存
    ├── tmp/                  # 临时文件
    └── logs/                 # 处理日志
```

## 下一步

- 📖 阅读 [USAGE.md](USAGE.md) 了解详细用法
- ❓ 查看 [FAQ.md](FAQ.md) 常见问题
- 🔧 查看 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) 了解技术细节

## 获取帮助

如有问题，请：
1. 查看 [FAQ.md](FAQ.md)
2. 提交 [GitHub Issue](https://github.com/your-repo/issues)
3. 查看日志文件 `OUTPUT/logs/run-*.log`

---

**祝您使用愉快！** 🎉
