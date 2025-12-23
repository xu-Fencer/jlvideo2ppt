# JPEG合并为PDF脚本使用说明

## 功能
`merge_jpeg_to_pdf.py` 是一个独立的脚本，用于按顺序合并当前目录下的所有JPEG文件为一个PDF文档。

## 特性
- ✅ 按文件名排序后添加（自然排序）
- ✅ 自动检测 `.jpg` 和 `.jpeg` 文件（不区分大小写）
- ✅ 保持原始图片尺寸（每页PDF大小等于对应图片尺寸）
- ✅ 支持自定义输出文件名和目录
- ✅ 详细的处理日志和进度显示
- ✅ 错误处理和清理机制

## 依赖安装
```bash
pip install reportlab Pillow
```

## 使用方法

### 1. 合并当前目录下的JPEG文件
```bash
python merge_jpeg_to_pdf.py
```
输出：`merged_20241222_143000.pdf`

### 2. 合并指定目录下的JPEG文件
```bash
python merge_jpeg_to_pdf.py -i /path/to/images
```

### 3. 指定输出文件名
```bash
python merge_jpeg_to_pdf.py -o my_presentation.pdf
```

### 4. 指定输出目录
```bash
python merge_jpeg_to_pdf.py -d /path/to/output
```

### 5. 组合使用
```bash
python merge_jpeg_to_pdf.py -i /path/to/images -o slides.pdf -d /path/to/output
```

## 处理流程
1. 扫描指定目录下的所有 `.jpg` 和 `.jpeg` 文件
2. 按文件名排序（自然排序，如 slide_1.jpg, slide_2.jpg, ...）
3. 逐个处理图片：
   - 获取图片尺寸
   - 设置PDF页面大小为图片尺寸
   - 将图片添加到PDF
   - 添加新页面（除了最后一页）
4. 保存PDF文件
5. 显示处理结果摘要

## 输出示例
```
找到 5 个JPEG文件
文件列表:
  1. slide_01.jpg
  2. slide_02.jpg
  3. slide_03.jpg
  4. slide_04.jpg
  5. slide_05.jpg
已添加: slide_01.jpg (1920x1080)
已添加: slide_02.jpg (1920x1080)
已添加: slide_03.jpg (1920x1080)
已添加: slide_04.jpg (1920x1080)
已添加: slide_05.jpg (1920x1080)

✅ PDF合并完成!
📄 输出文件: merged_20241222_143000.pdf
📊 包含 5 页
```

## 注意事项
- 脚本会按照文件名进行自然排序（如 slide_10.jpg 会在 slide_2.jpg 之后）
- 如果文件名中包含数字，建议使用前导零（如 slide_001.jpg, slide_002.jpg）
- PDF每页的尺寸会自动调整为对应图片的尺寸
- 如果处理过程中出错，会自动清理未完成的PDF文件

## 与项目集成
这个脚本可以独立使用，也可以与JLVideo2PPT项目配合使用：
1. 使用项目导出高清原图
2. 使用此脚本将原图合并为PDF
3. 获得高质量的PDF文档
