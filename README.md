# RAW to YUV and BMP Converter

## 简介
该工具用于将 RAW 图像文件转换为 YUV 和 BMP 格式。支持多种输入格式，包括 RAW10 和 RAW12，并且可以将 RAW 格式直接转换为 RAW 格式。

## 安装依赖
确保安装以下 Python 库：

pip install numpy opencv-python Pillow matplotlib

## 使用方法

python imagingtools.py <input_file> <output_file> <width> <height> --input-format <input_format> --output-format <output_format> [--bayer-pattern <bayer_pattern>] [--save-preview]

### 参数说明
- `<input_file>`: 输入的 RAW 文件路径。
- `<output_file>`: 输出的 YUV、BMP 或 RAW 文件路径。
- `<width>`: 图像宽度。
- `<height>`: 图像高度。
- `--input-format`: 输入格式，支持以下选项：
  - `raw8`
  - `raw10_plain`
  - `raw10_mipi`
  - `raw12_plain`
  - `raw12_mipi`
- `--output-format`: 输出格式，支持以下选项：
  - `raw10_plain`（直接输出 RAW 格式）
  - `yuv420`（输出 YUV 格式）
  - `bmp`（输出 BMP 格式）
- `--bayer-pattern`: 拜耳模式，支持以下选项（默认值为 RGGB）：
  - `RGGB`
  - `BGGR`
  - `GRBG`
  - `GBRG`
- `--save-preview`: 可选参数，保存预览图像为 PNG 格式。

## 示例
1. 转换为 YUV 格式：
   ```bash
   python imagingtools.py input.raw output.yuv 800 448 --input-format raw12_plain --output-format yuv420 --bayer-pattern RGGB --save-preview
   ```

2. 转换为 RAW 格式：
   ```bash
   python imagingtools.py input.raw output.raw10 800 448 --input-format raw10_plain --output-format raw10_plain --save-preview
   ```

3. 转换为 BMP 格式：
   ```bash
   python imagingtools.py input.raw output.bmp 800 448 --input-format raw10_plain --output-format bmp --bayer-pattern RGGB --save-preview
   ```

## 注意事项
- 确保输入文件的格式与指定的输入格式匹配。
- 输出文件的格式应与指定的输出格式一致。

## 许可证
MIT License


