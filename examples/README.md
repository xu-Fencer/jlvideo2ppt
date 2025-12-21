# 示例和演示

本目录包含用于测试和演示 JL Video to PPT Converter 的示例代码和脚本。

## 创建测试视频

### 快速开始

```bash
# 创建一个包含 10 张幻灯片的测试视频
python create_test_video.py -o test_presentation.mp4

# 创建 20 张幻灯片的视频
python create_test_video.py -n 20 -o long_presentation.mp4

# 创建带过渡效果的视频
python create_test_video.py --transitions -o smooth_presentation.mp4

# 创建高清视频
python create_test_video.py --width 1920 --height 1080 -o hd_presentation.mp4
```

### 参数说明

- `-o, --output`: 输出视频文件路径
- `-n, --num-slides`: 幻灯片数量（默认 10）
- `-d, --duration`: 每个幻灯片持续时间（秒，默认 3.0）
- `--fps`: 帧率（默认 30）
- `--width`: 视频宽度（默认 1280）
- `--height`: 视频高度（默认 720）
- `--transitions`: 添加幻灯片之间的过渡效果

### 示例输出

运行 `python create_test_video.py -o demo.mp4` 将创建一个包含以下内容的视频：

```
Slide 1: Introduction (Blue background)
Slide 2: Problem Statement (Red background)
Slide 3: Solution Overview (Green background)
Slide 4: Methodology (Yellow background)
Slide 5: Results (Magenta background)
Slide 6: Analysis (Cyan background)
Slide 7: Discussion (Gray background)
Slide 8: Conclusions (Orange background)
Slide 9: Future Work (Purple background)
Slide 10: Thank You (Light Blue background)
```

每张幻灯片包含：
- 幻灯片标题
- 副标题
- 页码（右下角）
- 帧计数（左下角，用于调试）

## 测试步骤

### 1. 基本功能测试

```bash
# 创建测试视频
python examples/create_test_video.py -o test.mp4

# 使用默认设置处理
python main.py test.mp4

# 查看输出
ls -la OUTPUT/
```

### 2. 不同预设测试

```bash
# 使用高质量预设
python main.py test.mp4 --preset high_quality

# 使用快速预设
python main.py test.mp4 --preset fast
```

### 3. 页码识别测试

测试视频中的页码应该能够被成功识别，因为：
- 页码位于右下角
- 使用白色文字，对比度高
- 格式简单（纯数字）

### 4. 导出测试

```bash
# 导出为 PDF
python main.py test.mp4

# 导出为图片
python main.py test.mp4 --export-format png

# 导出为 JPEG
python main.py test.mp4 --export-format jpeg
```

## 高级测试

### 测试检测参数调整

编辑 `src/presets.py` 创建自定义预设：

```python
custom_preset = Preset(
    name="custom_test",
    detection=DetectionParams(
        frame_diff_threshold=20.0,  # 更敏感
        min_stable_frames=2,        # 更少确认帧
        min_duration_sec=1.0        # 更短最小时间
    )
)
```

### 测试不同分辨率

```bash
# 创建 4K 视频（需要更多资源）
python create_test_video.py --width 3840 --height 2160 -o 4k_test.mp4

# 创建小视频（快速处理）
python create_test_video.py --width 640 --height 480 -o small_test.mp4
```

### 测试长视频

```bash
# 创建 100 张幻灯片的视频
python create_test_video.py -n 100 -o long_test.mp4
```

## 性能基准

### 测试环境

- CPU: Intel i7-8700K
- RAM: 16GB
- SSD: 500GB
- OS: Ubuntu 20.04

### 基准结果

| 视频类型 | 分辨率 | 幻灯片数 | 处理时间 | 内存占用 |
|---------|--------|----------|----------|----------|
| 小测试  | 640x480 | 10 | ~30 秒 | ~200 MB |
| 标准测试| 1280x720 | 10 | ~1 分钟 | ~500 MB |
| HD 测试 | 1920x1080 | 10 | ~2 分钟 | ~1 GB |
| 4K 测试 | 3840x2160 | 10 | ~5 分钟 | ~2 GB |

### 优化建议

1. **预览时使用快速预设**
2. **最终导出时使用高质量预设**
3. **对于长视频，考虑分段处理**
4. **使用 SSD 存储临时文件**

## 故障排除示例

### 问题：检测不到幻灯片

**原因**: 幻灯片之间差异太小

**解决方案**:
```bash
# 使用更敏感的预设
python main.py test.mp4 --preset high_quality
```

### 问题：OCR 识别失败

**原因**: 页码不在默认 ROI 区域

**解决方案**: 调整 ROI 参数：
- ROI X: 70%（右边界）
- ROI Y: 70%（下边界）
- ROI Width: 25%
- ROI Height: 20%

### 问题：导出失败

**原因**: 磁盘空间不足

**解决方案**:
```bash
# 清理临时文件
rm -rf OUTPUT/tmp/* OUTPUT/thumbs/*

# 使用较小的缩略图
python create_test_video.py --width 640 --height 480
```

## 贡献示例

如果您创建了有用的示例或测试用例，欢迎提交！

1. Fork 项目
2. 在 `examples/` 目录中添加您的示例
3. 更新本 README.md
4. 创建 Pull Request

### 示例命名规范

- `demo_*.py`: 演示特定功能的脚本
- `benchmark_*.py`: 性能测试脚本
- `test_*.py`: 功能测试脚本
- `example_*.py`: 使用示例

## 更多资源

- [USAGE.md](../USAGE.md) - 详细使用指南
- [FAQ.md](../FAQ.md) - 常见问题解答
- [README.md](../README.md) - 项目概述

---

**注意**: 测试视频仅用于开发和测试目的，不应用于生产环境。
