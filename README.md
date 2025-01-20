# RAW to YUV and BMP Converter

## 简介
该工具提供 RAW 图像格式转换功能，支持将 RAW 图像转换为 YUV、BMP 或其他 RAW 格式。工具提供命令行和图形界面两种使用方式，方便不同场景下的使用需求。

### 主要特性
- 支持多种 RAW 格式输入（RAW8/RAW10/RAW12）
- 支持 YUV420/BMP/RAW 格式输出
- 支持多种拜耳模式
- 提供预览图像功能
- 支持命令行和图形界面操作

## 安装

### 环境要求
- Python 3.x

### 依赖安装
```bash
pip install numpy opencv-python Pillow matplotlib
```

## 使用指南

### 1. 图形界面工具
运行以下命令启动图形界面：
```bash
python imagingtools_graphical.py
```

#### 操作步骤
1. **基本设置**
   - 输入图像尺寸（宽度和高度）
   - 选择输入文件和输出文件路径
   - 选择输入格式和输出格式
   - 选择拜耳模式
   - 设置是否需要预览图像

2. **转换操作**
   - 确认所有参数设置无误后，点击"转换"按钮
   - 查看状态栏获取处理进度和结果

### 2. 命令行工具
基本用法：
```bash
python imagingtools.py <input_file> <output_file> <width> <height> --input-format <input_format> --output-format <output_format> [--bayer-pattern <bayer_pattern>] [--save-preview]
```

#### 参数说明
- **必选参数**
  - `input_file`：输入 RAW 文件路径
  - `output_file`：输出文件路径
  - `width`：图像宽度
  - `height`：图像高度
  - `input-format`：输入格式
  - `output-format`：输出格式

- **可选参数**
  - `bayer-pattern`：拜耳模式（默认：RGGB）
  - `save-preview`：是否保存预览图像

#### 支持的格式
- **输入格式**
  - `raw8`
  - `raw10_plain`
  - `raw10_mipi`
  - `raw12_plain`
  - `raw12_mipi`

- **输出格式**
  - `raw10_plain`：RAW 格式输出
  - `yuv420`：YUV 格式输出
  - `bmp`：BMP 格式输出

- **拜耳模式**
  - `RGGB`
  - `BGGR`
  - `GRBG`
  - `GBRG`

### 使用示例

1. **YUV 格式转换**
```bash
python imagingtools.py input.raw output.yuv 800 448 --input-format raw12_plain --output-format yuv420 --bayer-pattern RGGB --save-preview
```

2. **RAW 格式转换**
```bash
python imagingtools.py input.raw output.raw10 800 448 --input-format raw10_plain --output-format raw10_plain --save-preview
```

3. **BMP 格式转换**
```bash
python imagingtools.py input.raw output.bmp 800 448 --input-format raw10_plain --output-format bmp --bayer-pattern RGGB --save-preview
```

## 注意事项
1. 确保输入文件格式与指定的输入格式匹配
2. 输出文件扩展名应与选择的输出格式相符
3. 图像尺寸参数必须与实际 RAW 图像尺寸一致
4. 使用预览功能时会额外生成 PNG 格式的预览图像

## 许可证
MIT License
```