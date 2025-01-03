# RAW Image Format Conversion Tool

A Python tool for converting between different RAW image formats with preview support.

## Features

- Supports multiple RAW formats conversion to Plain RAW10
- Real-time preview of input and output images
- Optional comparison preview saving as PNG
- Detailed format verification and error checking
- Command-line interface with comprehensive help

## Supported Formats

### Input Formats
1. **RAW8** (raw8)
   - 8 bits per pixel
   - 1 byte per pixel
   - File size = width * height bytes

2. **Plain RAW10** (raw10_plain)
   - 10 bits per pixel
   - 2 bytes per pixel
   - File size = width * height * 2 bytes

3. **MIPI RAW10** (raw10_mipi)
   - 10 bits per pixel
   - 5 bytes per 4 pixels (packed)
   - File size = (width * height * 10 + 7) // 8 bytes

4. **Plain RAW12** (raw12_plain)
   - 12 bits per pixel
   - 2 bytes per pixel
   - File size = width * height * 2 bytes

5. **MIPI RAW12** (raw12_mipi)
   - 12 bits per pixel
   - 3 bytes per 2 pixels (packed)
   - File size = (width * height * 12 + 7) // 8 bytes

### Output Format
- **Plain RAW10**
  - 10 bits effective data
  - LSB aligned in 16-bit words
  - File size = width * height * 2 bytes

## Installation

### Prerequisites
- Python 3.6 or higher
- NumPy
- Matplotlib
pip install numpy matplotlib

## Usage

### Basic Command Format
python imagingtools.py <input_file> <output_file> <width> <height> --input-format <format> [--save-preview]

### Examples

1. Convert RAW8 sensor output:
python imagingtools.py input.raw output.raw10 2048 1528 --input-format raw8

2. Convert MIPI RAW10 with preview:
python imagingtools.py input.raw output.raw10 3840 2160 --input-format raw10_mipi --save-preview

3. Convert Plain RAW12:
python imagingtools.py input.raw output.raw10 4096 3072 --input-format raw12_plain

### Common Image Resolutions

| Resolution | Dimensions | Description |
|------------|------------|-------------|
| 8MP | 3840x2160 | 4K UHD |
| 12MP | 4096x3072 | Standard 4:3 |
| 16MP | 4624x3472 | Mobile Sensor |
| 20MP | 5184x3888 | DSLR |
| 32MP | 6528x4896 | High-res Mobile |
| 48MP | 8000x6000 | High-end Mobile |
| 50MP | 8192x6144 | Professional |

### File Size Reference

Example for 4K UHD (3840x2160):
- RAW8: 8.3 MB
- RAW10 (Plain): 16.6 MB
- RAW10 (MIPI): 10.4 MB
- RAW12 (Plain): 16.6 MB
- RAW12 (MIPI): 12.4 MB

## Preview Feature

The `--save-preview` option generates a PNG file showing:
- Input image with format and pixel value range
- Output image (Plain RAW10) with pixel value range
- Side-by-side comparison for quality verification

## Error Handling

The tool includes various error checks:
- File size verification
- Format compatibility
- Resolution validation
- Data range checking

## Tips

1. Always verify input format with file size:
   ```bash
   ls -l input.raw  # Check file size
   python imagingtools.py --help  # Check format sizes
   ```

2. Use preview for format verification:
   ```bash
   python imagingtools.py input.raw output.raw10 width height --input-format format --save-preview
   ```

3. Common issues:
   - File size mismatch: Check resolution and format
   - Corrupted image: Verify input format
   - Black/white image: Check bit depth conversion

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.


