# JL Video to PPT Converter

从视频演示文稿中自动检测幻灯片并导出为图片或PDF格式的工具。

> 献给可爱的 JL 老师

## 功能特性

- **智能幻灯片检测**：基于帧差法自动识别视频中的幻灯片变化，支持第一帧保留
- **URL视频支持**：支持本地视频文件和HTTP/HTTPS在线视频URL
- **时间范围检测**：支持设置检测起止时间（格式：HH:MM:SS），灵活控制检测范围
- **高性能处理**：使用帧定位直接跳转，大幅减少视频读取时间（优化后减少50%）
- **高清原图导出**：OCR识别使用高清原图，大幅提升页码识别准确率
- **OCR页码识别**：自动检测和识别幻灯片页码，支持ROI区域预览和可视化
- **交互式筛选**：Gallery点击选择界面，支持全选/反选，智能统计
- **页码管理**：自动保存识别页码到pages文件夹，智能处理重复页码，检测缺页
- **智能导出**：自动检测页码并使用页码命名，支持PDF纯图片导出（无页码叠加）
- **多源数据支持**：区分筛选页面和OCR页面的数据来源，避免混淆
- **时间戳目录**：每次处理自动创建独立的时间戳目录，避免文件冲突
- **多种导出格式**：支持PNG、JPEG、PDF等多种导出格式
- **预设系统**：保存和加载常用配置参数
- **Gradio界面**：友好的Web UI，支持本地运行和在线分享（端口7930）

## 系统要求

- Python 3.8+
- ffmpeg
- Tesseract OCR (可选，用于页码识别)

## 安装依赖

```bash
pip install -r requirements.txt
```

### 系统依赖安装

#### Windows

```bash
# 安装 ffmpeg
# 下载并添加到 PATH: https://ffmpeg.org/download.html

# 安装 Tesseract
# 下载并安装: https://github.com/UB-Mannheim/tesseract/wiki
```

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg tesseract-ocr
```

#### macOS

```bash
brew install ffmpeg tesseract
```

## 使用方法

### 1. Gradio界面启动

```bash
python main.py
```
访问 http://127.0.0.1:7930 使用Web界面

### 2. 上传视频

- Tab 1（解析）：上传本地视频文件或输入视频路径/URL
- 支持HTTP/HTTPS在线视频URL（如 http://example.com/video.mp4）
- 选择输出目录（默认为视频目录/OUTPUT）
- 可选：设置检测时间范围（格式：HH:MM:SS，如 1:30:45 或 30:45）
- 配置检测参数或使用预设
- 点击"解析"开始处理

### 3. 筛选幻灯片

- Tab 2（筛选）：查看所有检测到的幻灯片缩略图
- 点击缩略图选择/取消选择
- 使用全选/反选功能
- 发送到页码识别或导出功能

### 4. 识别页码

- Tab 3（页码识别）：配置ROI区域（页码位置）
- 上传图片预览ROI并可视化（红色框标注）
- 运行OCR自动识别页码（使用高清原图，提升准确率）
- 自动保存页码到pages文件夹，智能处理重复页码
- 检测并显示缺页列表
- 发送到导出功能（仅非重复页码）

### 5. 导出结果

- Tab 4（导出）：选择导出格式（PNG/JPEG/PDF）
- 智能命名：自动检测页码并使用页码命名（如slide_0001.jpeg）
- 区分数据来源：显示来自筛选页面还是OCR页面
- PDF导出为纯图片（无页码叠加）

### 6. 命令行使用

```bash
# 基本用法
python main.py /path/to/video.mp4

# 指定输出目录
python main.py /path/to/video.mp4 -o /path/to/output

# 使用预设
python main.py /path/to/video.mp4 --preset my_preset

# 设置检测时间范围
python main.py /path/to/video.mp4 --start-time 1:30:45 --end-time 5:00:00

# 处理在线视频URL
python main.py "http://example.com/video.mp4"

# 查看帮助
python main.py --help
```

### 7. 独立工具：JPEG合并为PDF

```bash
# 安装依赖
pip install reportlab Pillow

# 合并当前目录下的JPEG文件
python merge_jpeg_to_pdf.py

# 合并指定目录
python merge_jpeg_to_pdf.py -i /path/to/images -o my_presentation.pdf
```

## 配置参数

### 检测参数

- **抽帧FPS**：每秒提取帧数（默认1fps）
- **帧差阈值**：检测幻灯片变化的阈值（默认30）
- **最小稳定帧数**：确认新幻灯片所需的稳定帧数（默认3）
- **最小持续时间**：幻灯片最小显示时间（秒，默认2s）
- **起始时间**：检测开始时间（格式：HH:MM:SS，默认0:00:00）
- **结束时间**：检测结束时间（格式：HH:MM:SS，默认视频结束）
- **备用算法**：可选SSIM或直方图算法作为辅助判断

### 高清原图参数

- **原图质量**：JPEG质量系数（默认95）
- **原图保存**：自动保存高清原图用于OCR识别

### 缩略图参数

- **尺寸**：缩略图最大边长（默认800px）
- **压缩质量**：JPEG质量系数（默认85）

### 页码识别参数

- **ROI区域**：页码位置的矩形区域（百分比坐标）
- **OCR引擎**：pytesseract或easyocr
- **后处理**：数字正则表达式匹配

## 预设系统

预设保存所有可配置参数到JSON文件，位于 `OUTPUT/presets/`目录。

- **保存预设**：输入名称并点击"保存预设"
- **加载预设**：从下拉列表选择并点击"加载预设"
- **设默认**：将当前配置设为默认预设

## 目录结构

```
jlvideo2ppt/
├── src/                    # 源代码目录
│   ├── video_io.py        # 视频IO处理
│   ├── slide_detector.py   # 幻灯片检测
│   ├── thumbs.py          # 缩略图生成（含高清原图保存）
│   ├── selection.py       # 筛选界面
│   ├── page_number.py     # 页码识别
│   ├── exporter.py        # 导出功能
│   ├── presets.py         # 预设管理
│   └── pipeline.py        # 流程管线
├── OUTPUT/                 # 输出目录
│  ├─presets/              # 预设存储
│  └─<video_name>_timestamp/
│     ├─logs/              # 日志目录
│     ├─thumbs/            # 缩略图
│     ├─images/            # 高清原图（用于OCR）
│     ├─pages/             # 页码识别结果
│     └─export/            # 导出结果
├── tests/                 # 测试目录
├── main.py               # 入口文件
├── merge_jpeg_to_pdf.py  # JPEG合并PDF工具
├── build_exe.bat         # Windows打包脚本
├── build_exe.sh          # Linux/Mac打包脚本
├── requirements.txt      # 依赖列表
└── README.md            # 说明文档
```

## 常见问题

### Q: 视频格式不支持？

A: 确保安装了ffmpeg，支持的格式包括：MP4、AVI、MOV、MKV、WMV、FLV、WebM等。

### Q: OCR识别不准确？

A: 1) OCR现在使用高清原图识别，大幅提升准确率；2) 调整ROI区域确保只包含页码部分；3) 更换OCR引擎；4) 手动编辑识别结果。

### Q: 幻灯片检测不准确？

A: 1) 调整帧差阈值；2) 增加最小稳定帧数；3) 调整最小持续时间；4) 使用时间范围限制检测区域。

### Q: 如何处理长视频中的一部分？

A: 使用时间范围参数，设置起始时间和结束时间，格式为 HH:MM:SS（如 --start-time 1:30:00 --end-time 5:00:00）。

### Q: 如何处理在线视频？

A: 直接在输入框中粘贴HTTP/HTTPS URL即可（如 http://example.com/video.mp4）。

### Q: 如何清理临时文件？

A: 删除 `OUTPUT/<video>_timestamp/`目录中的 `thumbs/`、`tmp/`文件夹，或在UI中点击"清理缓存"。

## 许可证

MIT License

## 贡献

欢迎提交Issues和Pull Requests！

## 更新日志

### v1.1.0

- **URL视频支持**：支持HTTP/HTTPS在线视频URL输入
- **时间范围检测**：支持设置检测起止时间（HH:MM:SS格式）
- **性能优化**：使用帧定位直接跳转，视频读取时间减少50%
- **高清OCR**：OCR识别改用高清原图，显著提升识别准确率
- **独立工具**：新增merge_jpeg_to_pdf.py用于JPEG合并PDF
- **exe打包**：提供build_exe.bat/sh脚本用于打包exe

### v1.0.0

- 初始版本发布
- 支持基本幻灯片检测和导出
- Gradio UI界面
- OCR页码识别
- 预设系统
