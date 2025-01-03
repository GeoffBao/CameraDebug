import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

def read_raw12_plain(raw_data, width, height):
    """读取Plain RAW12格式"""
    data_16bit = np.frombuffer(raw_data, dtype=np.uint16)
    image = np.zeros((height, width), dtype=np.uint16)
    
    for y in range(height):
        for x in range(width):
            if (y * width + x) < len(data_16bit):
                image[y, x] = data_16bit[y * width + x] & 0x0FFF
    
    return image

def read_raw12_mipi(raw_data, width, height):
    """读取MIPI RAW12格式"""
    image = np.zeros((height, width), dtype=np.uint16)
    bytes_per_line = (width * 12 + 7) // 8
    
    for y in range(height):
        for x in range(0, width, 2):
            byte_offset = y * bytes_per_line + (x * 12) // 8
            if byte_offset + 2 >= len(raw_data):
                break
            
            # MIPI RAW12打包格式
            b0 = int(raw_data[byte_offset])
            b1 = int(raw_data[byte_offset + 1])
            b2 = int(raw_data[byte_offset + 2])
            
            # 解包两个12位像素
            pixel1 = ((b0 << 4) | (b1 >> 4)) & 0xFFF
            pixel2 = ((b1 & 0x0F) << 8 | b2) & 0xFFF
            
            image[y, x] = pixel1
            if x + 1 < width:
                image[y, x + 1] = pixel2
    
    return image

def read_raw8(raw_data, width, height):
    """Read RAW8 format image"""
    try:
        # 确保数据大小正确
        expected_size = width * height
        actual_size = len(raw_data)
        print(f"Expected RAW8 file size: {expected_size} bytes")
        print(f"Actual file size: {actual_size} bytes")
        
        if actual_size != expected_size:
            raise ValueError(f"File size mismatch. Expected {expected_size} bytes, got {actual_size} bytes")
        
        # 创建连续的内存数组
        image = np.frombuffer(raw_data, dtype=np.uint8).copy()  # 添加.copy()确保数据连续
        image = image.reshape((height, width))  # 重塑为2D数组
        
        return image.copy()  # 返回连续的数组副本
        
    except Exception as e:
        print(f"Error reading RAW8: {str(e)}")
        raise

def detect_raw_format(raw_data, width, height):
    """Auto detect RAW format"""
    # Calculate theoretical file sizes
    raw8_size = width * height          # RAW8: 1 byte per pixel
    plain_size = width * height * 2      # Plain RAW12: 2 bytes per pixel
    mipi_size = (width * height * 12 + 7) // 8  # MIPI RAW12: 12 bits per pixel packed
    
    file_size = len(raw_data)
    print(f"\nFile size analysis:")
    print(f"Actual file size: {file_size} bytes")
    print(f"RAW8 expected size: {raw8_size} bytes")
    print(f"Plain RAW12 expected size: {plain_size} bytes")
    print(f"MIPI RAW12 expected size: {mipi_size} bytes")
    
    try:
        # Try parsing as different formats
        raw8_image = read_raw8(raw_data, width, height)
        plain_image = read_raw12_plain(raw_data, width, height)
        mipi_image = read_raw12_mipi(raw_data, width, height)
        
        # Display comparison
        fig = plt.figure(figsize=(15, 5))
        fig.canvas.manager.set_window_title('RAW Format Auto Detection')
        
        plt.suptitle('RAW Format Detection - Parsing Results Comparison', fontsize=14, y=0.95)
        
        # RAW8
        plt.subplot(131)
        plt.imshow(raw8_image, cmap='gray')
        plt.title("Parsed as RAW8\n(1 byte per pixel)", pad=10)
        plt.colorbar(label='Pixel Value (8-bit: 0-255)')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        
        # Plain RAW12
        plt.subplot(132)
        plt.imshow(plain_image, cmap='gray')
        plt.title("Parsed as Plain RAW12\n(2 bytes per pixel)", pad=10)
        plt.colorbar(label='Pixel Value (12-bit: 0-4095)')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        
        # MIPI RAW12
        plt.subplot(133)
        plt.imshow(mipi_image, cmap='gray')
        plt.title("Parsed as MIPI RAW12\n(3 bytes per 2 pixels)", pad=10)
        plt.colorbar(label='Pixel Value (12-bit: 0-4095)')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        
        plt.tight_layout()
        plt.show()
        
        # Analyze image statistics for format detection
        raw8_stats = analyze_image(raw8_image, "RAW8")
        plain_stats = analyze_image(plain_image, "Plain RAW12")
        mipi_stats = analyze_image(mipi_image, "MIPI RAW12")
        
        # Score each format
        raw8_score = evaluate_image(raw8_stats, expected_max=255)
        plain_score = evaluate_image(plain_stats, expected_max=4095)
        mipi_score = evaluate_image(mipi_stats, expected_max=4095)
        
        print("\nFormat detection scores:")
        print(f"RAW8 score: {raw8_score:.2f}")
        print(f"Plain RAW12 score: {plain_score:.2f}")
        print(f"MIPI RAW12 score: {mipi_score:.2f}")
        
        # Determine the most likely format
        scores = {
            'raw8': raw8_score,
            'plain': plain_score,
            'mipi': mipi_score
        }
        detected_format = max(scores.items(), key=lambda x: x[1])[0]
        
        if detected_format == 'raw8':
            return 'raw8', raw8_image
        elif detected_format == 'plain':
            return 'plain', plain_image
        else:
            return 'mipi', mipi_image
            
    except Exception as e:
        print(f"Error during format detection: {str(e)}")
        raise

def analyze_image(image, format_name):
    """分析图像特征"""
    stats = {
        'min': np.min(image),
        'max': np.max(image),
        'mean': np.mean(image),
        'std': np.std(image),
        'entropy': calculate_entropy(image),
        'gradient': calculate_gradient(image)
    }
    
    print(f"\n{format_name} 统计信息:")
    print(f"范围: {stats['min']} - {stats['max']}")
    print(f"平均值: {stats['mean']:.2f}")
    print(f"标准差: {stats['std']:.2f}")
    print(f"熵: {stats['entropy']:.2f}")
    print(f"梯度: {stats['gradient']:.2f}")
    
    return stats

def evaluate_image(stats, expected_max):
    """Evaluate image quality score with expected maximum value"""
    score = 0
    
    # Check if values are in expected range
    if 0 <= stats['min'] <= expected_max and 0 <= stats['max'] <= expected_max:
        score += 1
    
    # Check dynamic range
    dynamic_range = stats['max'] - stats['min']
    score += min(dynamic_range / expected_max, 1)
    
    # Consider entropy
    score += min(stats['entropy'] / 8, 1)
    
    # Consider gradient information
    score += min(stats['gradient'] / 100, 1)
    
    return score

def calculate_entropy(image):
    """计算图像熵"""
    histogram = np.histogram(image, bins=256)[0]
    histogram = histogram / float(np.sum(histogram))
    histogram = histogram[histogram > 0]
    return -np.sum(histogram * np.log2(histogram))

def calculate_gradient(image):
    """计算图像梯度"""
    gradient_x = np.diff(image, axis=1)
    gradient_y = np.diff(image, axis=0)
    return np.mean(np.abs(gradient_x)) + np.mean(np.abs(gradient_y))

def display_raw_images(input_image, output_image, input_format, output_format):
    """显示输入和输出RAW图像的对比"""
    # 创建figure并设置窗口标题
    fig = plt.figure(figsize=(15, 6))
    fig.canvas.manager.set_window_title('RAW Image Comparison')  # 修改窗口标题
    
    # 显示输入图像
    plt.subplot(121)
    plt.imshow(input_image, cmap='gray')
    plt.title(f"Input RAW12 ({input_format})")
    plt.colorbar()
    
    # 显示输出图像
    plt.subplot(122)
    plt.imshow(output_image, cmap='gray')
    plt.title(f"Output RAW10 ({output_format})")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def convert_raw12_to_raw10(input_file, output_file, width, height, output_format='plain'):
    """转换RAW12到RAW10，自动检测输入格式"""
    try:
        # 读取输入数据
        with open(input_file, 'rb') as f:
            raw12_data = np.fromfile(f, dtype=np.uint8)
        
        # 自动检测输入格式
        input_format, input_image = detect_raw_format(raw12_data, width, height)
        
        # 转换为10位
        output_image = (input_image >> 2).astype(np.uint16)
        
        # 显示输入和输出图像对比
        display_raw_images(input_image, output_image, input_format, output_format)
        
        # 根据输出格式打包数据
        if output_format == 'plain':
            raw10_data = output_image.astype(np.uint16).tobytes()
        else:  # MIPI RAW10
            # 计算MIPI RAW10所需的字节数
            output_size = (width * height * 10 + 7) // 8
            raw10_data = bytearray(output_size)
            
            for y in range(height):
                for x in range(0, width, 4):
                    if x + 4 > width:
                        break
                    
                    # 获取4个像素
                    p0 = int(output_image[y, x]) & 0x3FF
                    p1 = int(output_image[y, x + 1]) & 0x3FF if x + 1 < width else 0
                    p2 = int(output_image[y, x + 2]) & 0x3FF if x + 2 < width else 0
                    p3 = int(output_image[y, x + 3]) & 0x3FF if x + 3 < width else 0
                    
                    # 计算目标偏移
                    dst_offset = (y * width + x) * 10 // 8
                    
                    if dst_offset + 4 < output_size:
                        # MIPI RAW10打包格式
                        raw10_data[dst_offset] = p0 & 0xFF
                        raw10_data[dst_offset + 1] = ((p0 >> 8) & 0x03) | ((p1 & 0x3F) << 2)
                        raw10_data[dst_offset + 2] = ((p1 >> 6) & 0x0F) | ((p2 & 0x0F) << 4)
                        raw10_data[dst_offset + 3] = ((p2 >> 4) & 0x3F) | ((p3 & 0x03) << 6)
                        raw10_data[dst_offset + 4] = (p3 >> 2) & 0xFF
        
        # 保存输出文件
        with open(output_file, 'wb') as f:
            f.write(raw10_data)
        
        print(f"\n转换完成: {output_file}")
        print(f"输入格式: {input_format}")
        print(f"输出格式: {output_format}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        raise

def read_raw10_plain(raw_data, width, height):
    """Read Plain RAW10 format (2 bytes per pixel)"""
    try:
        # 检查文件大小是否符合预期
        expected_size = width * height * 2  # 每个像素2字节
        actual_size = len(raw_data)
        print(f"Expected file size: {expected_size} bytes")
        print(f"Actual file size: {actual_size} bytes")
        
        # 确保数据大小正确
        if actual_size != expected_size:
            raise ValueError(f"File size mismatch. Expected {expected_size} bytes, got {actual_size} bytes")
        
        # 将字节数据转换为16位整数数组
        data = np.frombuffer(raw_data, dtype=np.uint16)
        image = data.reshape((height, width))
        
        # 只保留低10位
        image = image & 0x03FF
        
        return image
        
    except Exception as e:
        print(f"Error reading RAW10: {str(e)}")
        raise

def read_raw10_mipi(raw_data, width, height):
    """Read MIPI RAW10 format (5 bytes for 4 pixels)"""
    try:
        # 检查文件大小是否符合预期
        expected_size = (width * height * 10 + 7) // 8  # MIPI RAW10打包大小
        actual_size = len(raw_data)
        print(f"Expected MIPI RAW10 file size: {expected_size} bytes")
        print(f"Actual file size: {actual_size} bytes")
        
        if actual_size != expected_size:
            raise ValueError(f"File size mismatch. Expected {expected_size} bytes, got {actual_size} bytes")
        
        # 创建输出图像数组
        image = np.zeros((height, width), dtype=np.uint16)
        
        # 每5个字节包含4个10位像素
        for y in range(height):
            for x in range(0, width, 4):
                if x + 4 > width:
                    break
                    
                byte_offset = (y * width + x) * 10 // 8
                if byte_offset + 4 >= len(raw_data):
                    break
                
                # MIPI RAW10打包格式:
                # Byte0: P0[9:2]
                # Byte1: P1[9:2]
                # Byte2: P2[9:2]
                # Byte3: P3[9:2]
                # Byte4: P0[1:0],P1[1:0],P2[1:0],P3[1:0]
                
                b4 = raw_data[byte_offset + 4]  # 低位字节
                
                # 解包4个像素
                image[y, x] = (raw_data[byte_offset] << 2) | ((b4 >> 6) & 0x03)
                if x + 1 < width:
                    image[y, x + 1] = (raw_data[byte_offset + 1] << 2) | ((b4 >> 4) & 0x03)
                if x + 2 < width:
                    image[y, x + 2] = (raw_data[byte_offset + 2] << 2) | ((b4 >> 2) & 0x03)
                if x + 3 < width:
                    image[y, x + 3] = (raw_data[byte_offset + 3] << 2) | (b4 & 0x03)
        
        return image
        
    except Exception as e:
        print(f"Error reading MIPI RAW10: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description='''
RAW Image Format Conversion Tool
-------------------------------
This tool converts various RAW image formats to Plain RAW10.

Currently Supported Input Formats:
--------------------------------
1. RAW8 (raw8)
   - 8 bits per pixel
   - 1 byte per pixel
   - File size = width * height bytes
   
2. Plain RAW10 (raw10_plain)
   - 10 bits per pixel
   - 2 bytes per pixel
   - File size = width * height * 2 bytes
   
3. MIPI RAW10 (raw10_mipi)
   - 10 bits per pixel
   - 5 bytes per 4 pixels (packed)
   - File size = (width * height * 10 + 7) // 8 bytes
   
4. Plain RAW12 (raw12_plain)
   - 12 bits per pixel
   - 2 bytes per pixel
   - File size = width * height * 2 bytes
   
5. MIPI RAW12 (raw12_mipi)
   - 12 bits per pixel
   - 3 bytes per 2 pixels (packed)
   - File size = (width * height * 12 + 7) // 8 bytes

Output Format:
-------------
Plain RAW10 (2 bytes per pixel)
- 10 bits effective data
- LSB aligned in 16-bit words
- File size = width * height * 2 bytes
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('input', help='Input RAW file path')
    parser.add_argument('output', help='Output Plain RAW10 file path')
    parser.add_argument('width', type=int, help='Image width')
    parser.add_argument('height', type=int, help='Image height')
    parser.add_argument('--input-format', 
                        choices=['raw8', 'raw10_plain', 'raw10_mipi', 'raw12_plain', 'raw12_mipi'], 
                        required=True,
                        help='Input format')
    parser.add_argument('--save-preview', action='store_true',
                        help='Save preview image as PNG')
    
    parser.epilog = '''
Usage Examples:
--------------
1. Convert RAW8 sensor output (2048x1528):
   python imagingtools.py input.raw output.raw10 2048 1528 --input-format raw8

2. Convert MIPI packed RAW10 (3840x2160) and save comparison preview:
   python imagingtools.py input.raw output.raw10 3840 2160 --input-format raw10_mipi --save-preview

3. Convert Plain RAW12 (4096x3072):
   python imagingtools.py input.raw output.raw10 4096 3072 --input-format raw12_plain

4. Convert PDAF RAW8 image (2048x1528) and save preview:
   python imagingtools.py pdaf.raw pdaf_out.raw10 2048 1528 --input-format raw8 --save-preview

5. Convert high resolution MIPI RAW12 (8192x6144):
   python imagingtools.py sensor.raw output.raw10 8192 6144 --input-format raw12_mipi

6. Convert mobile camera RAW10 (4624x3472):
   python imagingtools.py camera.raw output.raw10 4624 3472 --input-format raw10_mipi

Common Image Resolutions:
------------------------
- 8MP (3840x2160): 4K UHD
- 12MP (4096x3072): Standard 4:3
- 16MP (4624x3472): Mobile Sensor
- 20MP (5184x3888): DSLR
- 32MP (6528x4896): High-res Mobile
- 48MP (8000x6000): High-end Mobile
- 50MP (8192x6144): Professional

File Size Examples:
------------------
For 4K UHD (3840x2160):
- RAW8: 8.3 MB
- RAW10 (Plain): 16.6 MB
- RAW10 (MIPI): 10.4 MB
- RAW12 (Plain): 16.6 MB
- RAW12 (MIPI): 12.4 MB

Note:
-----
Use --save-preview to generate a PNG file showing the input and output image comparison.
The preview helps verify correct format selection and conversion quality.
'''
    
    args = parser.parse_args()
    
    try:
        print(f"\nProcessing image:")
        print(f"Input file: {args.input}")
        print(f"Input format: {args.input_format}")
        print(f"Image size: {args.width}x{args.height}")
        
        # 读取输入文件
        with open(args.input, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint8)
            
        # 根据输入格式选择正确的读取函数
        if args.input_format == 'raw10_mipi':
            input_image = read_raw10_mipi(raw_data, args.width, args.height)
            output_image = input_image  # MIPI RAW10已经是10位数据，直接使用
            print("Reading as MIPI RAW10 format")
            
        elif args.input_format == 'raw10_plain':
            input_image = read_raw10_plain(raw_data, args.width, args.height)
            output_image = input_image  # Plain RAW10已经是10位数据，直接使用
            print("Reading as Plain RAW10 format")
            
        elif args.input_format == 'raw8':
            input_image = read_raw8(raw_data, args.width, args.height)
            output_image = (input_image.astype(np.uint16) << 2)  # 左移2位转换为10位
            print("Converting RAW8 to RAW10 (left shift 2 bits)")
            
        elif args.input_format == 'raw12_plain':
            input_image = read_raw12_plain(raw_data, args.width, args.height)
            output_image = (input_image >> 2)  # 右移2位转换为10位
            print("Converting Plain RAW12 to RAW10 (right shift 2 bits)")
            
        else:  # raw12_mipi
            input_image = read_raw12_mipi(raw_data, args.width, args.height)
            output_image = (input_image >> 2)  # 右移2位转换为10位
            print("Converting MIPI RAW12 to RAW10 (right shift 2 bits)")
        
        # 显示输入和输出图像
        fig = plt.figure(figsize=(12, 5))
        fig.canvas.manager.set_window_title('RAW Format Conversion Preview')
        
        plt.subplot(121)
        plt.imshow(input_image, cmap='gray')
        plt.title(f"Input Image ({args.input_format})")
        plt.colorbar(label='Input Pixel Value')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        
        plt.subplot(122)
        plt.imshow(output_image, cmap='gray')
        plt.title("Output Image (Plain RAW10)")
        plt.colorbar(label='Output Pixel Value (10-bit: 0-1023)')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        
        plt.tight_layout()
        
        # 保存预览图
        if args.save_preview:
            preview_path = os.path.splitext(args.output)[0] + '_preview.png'
            plt.savefig(preview_path, dpi=300, bbox_inches='tight')
            print(f"\nPreview image saved as: {preview_path}")
        
        plt.show()
        
        # 保存为Plain RAW10格式
        output_image = output_image.astype(np.uint16)
        with open(args.output, 'wb') as f:
            output_image.tofile(f)
            
        print(f"\nSuccessfully converted to Plain RAW10: {args.output}")
        print(f"Output range: {output_image.min()} - {output_image.max()}")
        
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()
