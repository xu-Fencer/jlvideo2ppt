# 常见问题解答 (FAQ)

## 安装与配置

### Q1: 如何安装依赖？

**A:** 按照以下步骤安装：

```bash
# 1. 克隆或下载项目
# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 安装系统依赖
# Ubuntu/Debian:
sudo apt-get install ffmpeg tesseract-ocr

# Windows:
# 下载并安装 ffmpeg: https://ffmpeg.org/download.html
# 下载并安装 Tesseract: https://github.com/UB-Mannheim/tesseract/wiki

# macOS:
brew install ffmpeg tesseract
```

### Q2: 提示 "ffprobe not found" 怎么办？

**A:** 需要安装 ffmpeg 并确保在 PATH 中：

```bash
# 检查是否安装
ffprobe -version

# 如果未安装，重新安装 ffmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# Windows: 从 https://ffmpeg.org/download.html 下载并添加到 PATH
```

### Q3: OCR 识别失败怎么办？

**A:** 检查 Tesseract 是否正确安装：

```bash
# 检查 Tesseract 版本
tesseract --version

# 如果未安装：
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# Windows: 从 https://github.com/UB-Mannheim/tesseract/wiki 下载
```

## 使用问题

### Q4: 上传视频后提示 "Unsupported video format"

**A:** 请检查：
1. 文件扩展名是否为支持的格式（.mp4, .avi, .mov, .mkv, .wmv, .flv, .webm）
2. 文件是否损坏
3. 尝试用 ffmpeg 转换格式：
   ```bash
   ffmpeg -i input.avi -c:v libx264 output.mp4
   ```

### Q5: 解析过程中出现 "Failed to open video"

**A:** 可能的解决方案：
1. 检查视频路径是否正确
2. 确认文件没有被其他程序占用
3. 尝试使用绝对路径
4. 检查文件权限

### Q6: 幻灯片检测不准确

**A:** 调整检测参数：
1. **降低帧差阈值**（如 20-25）检测更多变化
2. **增加最小稳定帧数**（如 5-7）减少误检测
3. **启用 SSIM** 备用检测方法
4. 尝试不同的预设（high_quality 或 fast）

### Q7: 页码识别结果为空或不准确

**A:** 优化 OCR 设置：
1. **调整 ROI 区域**：
   - 确保 ROI 只包含页码部分
   - 避免包含其他文本或图形
2. **尝试不同引擎**：
   - 切换 pytesseract 和 easyocr
3. **提高图像质量**：
   - 使用更高质量的缩略图
   - 确保页码清晰可见
4. **手动编辑**：在 UI 中手动修改识别的页码

### Q8: 导出失败

**A:** 检查以下项目：
1. **磁盘空间**：确保有足够空间存储输出文件
2. **权限**：确认对输出目录有写入权限
3. **路径长度**：Windows 路径不能超过 260 字符
4. **文件占用**：确保输出目录未被其他程序占用

### Q9: 处理速度很慢

**A:** 优化性能：
1. **使用 "fast" 预设**降低质量要求
2. **降低目标 FPS**（如设为 0.5）
3. **减小缩略图尺寸**（如 600px）
4. **关闭 SSIM 和直方图**备用检测
5. **增加最小持续时间**减少检测次数

### Q10: 内存不足错误

**A:** 减少资源占用：
1. **关闭其他应用程序**
2. **处理较短的视频片段**
3. **降低缩略图质量**
4. **使用更小的目标 FPS**

## 功能相关

### Q11: 如何批量处理多个视频？

**A:** 使用批处理脚本：

```bash
# 创建 batch_process.py
import subprocess
import glob
import os

videos = glob.glob("*.mp4")
for video in videos:
    output_dir = f"output/{os.path.splitext(video)[0]}"
    subprocess.run(["python", "main.py", video, "--output", output_dir])
    print(f"Processed: {video}")
```

### Q12: 可以自定义输出文件名格式吗？

**A:** 目前支持三种命名策略：
- `index`：按索引（slide_0001.jpeg）
- `page`：按页码（slide_0001.jpeg）
- `timestamp`：按时间戳（slide_5.50s.jpeg）

**智能命名**：如果识别了页码，系统会自动切换到页码命名策略，即使选择了index策略。例如识别页码1和3，会生成slide_0001.jpeg和slide_0003.jpeg。

如需自定义，可以修改 `src/exporter.py` 中的 `_generate_filename` 方法。

### Q13: 如何备份和恢复预设？

**A:** 预设文件存储在 `OUTPUT/presets/` 目录：
```bash
# 备份
cp -r OUTPUT/presets ./presets_backup

# 恢复
cp -r ./presets_backup ./OUTPUT/presets
```

### Q14: 支持哪些页码格式？

**A:** 当前支持：
- 纯数字：123
- 带前缀：Page 123, p. 123
- 中文：第123页
- 分数格式：1/10

不支持复杂格式（如 "Chapter 5, Page 3"）。

### Q15: 可以导出到 PowerPoint 吗？

**A:** 目前不支持直接导出到 PPT/PPTX 格式。可用的导出格式：
- PDF
- PNG
- JPEG

可以手动将导出的图片插入到 PowerPoint 中。

### Q16: OCR识别的页码保存在哪里？

**A:** 页码图片自动保存在 `OUTPUT/时间戳目录/pages/` 文件夹中：
- 第一次出现的页码：`slide_页码.jpg`（如 slide_1.jpg）
- 重复出现的页码：`slide_页码_重复N.jpg`（如 slide_2_重复1.jpg）
- 系统会自动检测并显示缺页列表

### Q17: 如何预览OCR的ROI区域？

**A:** 在OCR页面：
1. 上传一张幻灯片图片到"ROI预览"区域
2. 系统会自动显示预览信息（像素坐标和百分比）
3. 下方会显示"ROI区域可视化"，用红色框标注ROI位置
4. 调整ROI参数时，可视化图像会自动更新

### Q18: 筛选页面和OCR页面的数据如何区分？

**A:** 在导出页面，"选择来源"会明确显示：
- `使用筛选页面选择: x 张幻灯片` - 来自手动筛选
- `使用页码识别页面选择: x 张幻灯片` - 来自OCR识别（仅非重复页码）

### Q19: 为什么每次处理都会创建新的目录？

**A:** 系统自动创建时间戳目录（如 `video_20241221_143022`），这样可以：
- 避免不同处理结果相互覆盖
- 便于管理和比较多次处理的结果
- 保持输出文件整洁有序

## 技术问题

### Q20: 在服务器上运行失败

**A:** 配置注意事项：
1. 使用 `--gui` 参数启动
2. 设置 `server_name="0.0.0.0"` 允许外部访问
3. 配置防火墙允许相应端口
4. 如需认证，可使用反向代理（如 Nginx）

### Q21: 支持 GPU 加速吗？

**A:** 部分支持：
- **EasyOCR**：可配置使用 GPU（`easyocr.Reader(['en'], gpu=True)`）
- **OpenCV**：CPU 版本已足够
- **ffmpeg**：依赖系统安装，可能支持 GPU

整体处理瓶颈在视频解码，GPU 加速效果有限。

### Q22: 可以处理 4K 视频吗？

**A:** 可以，但建议：
1. 使用 "fast" 预设
2. 降低目标 FPS（如 0.5）
3. 减小缩略图尺寸（如 600px）
4. 确保有足够内存和磁盘空间

### Q23: 最小系统要求？

**A:** 推荐配置：
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8+
- **RAM**: 8GB+（处理 4K 视频需要 16GB+）
- **磁盘**: 至少 2GB 可用空间
- **ffmpeg**: 最新版本

### Q24: 如何查看详细日志？

**A:** 日志存储在 `OUTPUT/logs/` 目录：
```bash
# 查看最新日志
tail -f OUTPUT/logs/run-*.log

# 启用详细日志（修改 pipeline.py）
logging.getLogger().setLevel(logging.DEBUG)
```

## 错误排查

### 错误：ModuleNotFoundError

**A:** 确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

### 错误：PermissionError

**A:** 修复权限：
```bash
# Linux/macOS
chmod +x main.py

# 确保输出目录可写
chmod 755 OUTPUT
```

### 错误：No module named 'cv2'

**A:** 安装 OpenCV：
```bash
pip install opencv-python
```

### 错误：TesseractNotFoundError

**A:** 设置 Tesseract 路径：
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### 错误：ImportError for ReportLab

**A:** 安装 ReportLab：
```bash
pip install reportlab
```

## 性能调优

### 优化处理速度

1. **预处理视频**：
   ```bash
   # 降低分辨率和帧率
   ffmpeg -i input.mp4 -vf "scale=1280:-2" -r 15 processed.mp4
   ```

2. **使用 SSD 存储临时文件**

3. **关闭不必要的后台程序**

### 优化检测精度

1. **使用高质量预设**

2. **调整 ROI 精确匹配页码位置**

3. **预处理图像**：
   - 提高对比度
   - 减少噪声

## 其他问题

### Q25: 如何报告 Bug 或请求功能？

**A:** 请访问项目 GitHub 页面：
1. 搜索现有 Issue
2. 创建新 Issue（如果是 Bug 或功能请求）
3. 提供详细信息：
   - 操作系统和版本
   - Python 版本
   - 错误日志
   - 复现步骤

### Q26: 可以用于商业用途吗？

**A:** 请查看项目 LICENSE 文件。默认为 MIT 许可证，允许商业使用。

### Q27: 如何贡献代码？

**A:** 欢迎贡献！
1. Fork 项目
2. 创建功能分支
3. 提交代码
4. 创建 Pull Request
5. 编写测试用例

### Q28: 项目的开发路线图？

**A:** 计划功能：
- [ ] 支持更多视频格式
- [ ] 自动页码区域检测
- [ ] 批量导出到 PPTX
- [ ] Web 界面优化
- [ ] 云端处理支持

## 联系我们

如果以上 FAQ 没有解答您的问题，请：
1. 查看项目 README.md
2. 查看 USAGE.md 使用指南
3. 提交 GitHub Issue
4. 参与讨论

---

**注意**: 本 FAQ 会持续更新，请定期查看最新版本。
