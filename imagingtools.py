import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

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

def raw10_to_rgb(raw_data, width, height, bayer_pattern='RGGB'):
    """Convert RAW10 bayer data to RGB image
    Args:
        raw_data: numpy array of raw10 data
        width: image width
        height: image height
        bayer_pattern: bayer pattern ('RGGB', 'BGGR', 'GRBG', 'GBRG')
    Returns:
        rgb_image: RGB image array (height, width, 3)
    """
    try:
        # Normalize RAW10 data to 8-bit
        raw_norm = (raw_data.astype(np.float32) / 1023.0 * 255.0).astype(np.uint8)
        
        # Reshape to 2D array
        bayer = raw_norm.reshape((height, width))
        
        # Convert bayer to RGB using OpenCV
        if bayer_pattern == 'RGGB':
            cv_bayer_pattern = cv2.COLOR_BAYER_RG2RGB
        elif bayer_pattern == 'BGGR':
            cv_bayer_pattern = cv2.COLOR_BAYER_BG2RGB
        elif bayer_pattern == 'GRBG':
            cv_bayer_pattern = cv2.COLOR_BAYER_GR2RGB
        elif bayer_pattern == 'GBRG':
            cv_bayer_pattern = cv2.COLOR_BAYER_GB2RGB
        else:
            raise ValueError(f"Unsupported bayer pattern: {bayer_pattern}")
            
        rgb = cv2.cvtColor(bayer, cv_bayer_pattern)
        return rgb
        
    except Exception as e:
        print(f"Error converting RAW10 to RGB: {str(e)}")
        raise

def rgb_to_yuv420(rgb_image):
    """Convert RGB image to YUV420 format
    Args:
        rgb_image: RGB image array (height, width, 3)
    Returns:
        yuv420_data: YUV420 data as bytes
    """
    try:
        # Convert RGB to YUV using OpenCV
        yuv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV_I420)
        return yuv.tobytes()
        
    except Exception as e:
        print(f"Error converting RGB to YUV420: {str(e)}")
        raise

def yuv420_to_bmp(yuv_data, width, height, output_path):
    """Convert YUV420 data to BMP format and save it."""
    try:
        # YUV420格式的Y通道
        y_channel = yuv_data[:width * height].reshape((height, width))
        
        # 创建RGB图像
        rgb_image = Image.new("RGB", (width, height))
        
        # 将YUV转换为RGB
        for y in range(height):
            for x in range(width):
                Y = y_channel[y, x]
                U = yuv_data[width * height + (y // 2) * (width // 2) + (x // 2)] - 128
                V = yuv_data[width * height + (width * height // 4) + (y // 2) * (width // 2) + (x // 2)] - 128
                
                R = Y + 1.402 * V
                G = Y - 0.344136 * U - 0.714136 * V
                B = Y + 1.772 * U
                
                # Clip values to [0, 255]
                R = max(0, min(255, int(R)))
                G = max(0, min(255, int(G)))
                B = max(0, min(255, int(B)))
                
                rgb_image.putpixel((x, y), (R, G, B))
        
        # 保存为BMP格式
        rgb_image.save(output_path, "BMP")
        print(f"BMP output saved as: {output_path}")
        
    except Exception as e:
        print(f"Error converting YUV to BMP: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert between RAW and YUV formats')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('output', help='Output file path')
    parser.add_argument('width', type=int, help='Image width')
    parser.add_argument('height', type=int, help='Image height')
    parser.add_argument('--input-format', 
                        choices=['raw8', 'raw10_plain', 'raw10_mipi', 'raw12_plain', 'raw12_mipi'],
                        required=True,
                        help='Input format')
    parser.add_argument('--output-format',
                        choices=['raw10_plain', 'yuv420', 'bmp'],
                        required=True,
                        help='Output format (raw10_plain, yuv420, bmp)')
    parser.add_argument('--bayer-pattern',
                        choices=['RGGB', 'BGGR', 'GRBG', 'GBRG'],
                        default='RGGB',
                        help='Bayer pattern for RAW conversion (default: RGGB)')
    parser.add_argument('--save-preview', action='store_true',
                        help='Save preview image as PNG')
    
    args = parser.parse_args()

    try:
        # Read input file
        with open(args.input, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint8)
        
        # Initialize input_data
        input_data = None
        
        # Process input based on format
        if args.input_format == 'raw10_plain':
            input_data = np.frombuffer(raw_data, dtype=np.uint16)
            input_data = input_data.reshape((args.height, args.width)) & 0x03FF
        elif args.input_format == 'raw12_plain':
            input_data = np.frombuffer(raw_data, dtype=np.uint16)
            input_data = input_data.reshape((args.height, args.width)) >> 2  # Convert to 10 bits
        elif args.input_format == 'raw10_mipi':
            input_data = read_raw10_mipi(raw_data, args.width, args.height)
        elif args.input_format == 'raw12_mipi':
            input_data = read_raw12_mipi(raw_data, args.width, args.height)
        elif args.input_format == 'raw8':
            input_data = read_raw8(raw_data, args.width, args.height)
            input_data = (input_data.astype(np.uint16) << 2)  # Convert to 10 bits

        # Check if input_data was set
        if input_data is None:
            raise ValueError("Input data could not be processed. Please check the input format.")

        # Convert to output format
        if args.output_format == 'yuv420':
            # Convert RAW10 to RGB
            rgb_image = raw10_to_rgb(input_data, args.width, args.height, args.bayer_pattern)
            
            # Convert RGB to YUV420
            output_data = rgb_to_yuv420(rgb_image)
            
            # Save YUV file
            if not args.output:
                args.output = 'yuv420.yuv'
            with open(args.output, 'wb') as f:
                f.write(output_data)
            print(f"YUV output saved as: {args.output}")

            # Save RAW output (as 10-bit)
            raw_output_path = os.path.splitext(args.output)[0] + '_output.raw10'
            input_data.tofile(raw_output_path)
            print(f"RAW output saved as: {raw_output_path}")

            # Show input and YUV preview
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(input_data, cmap='gray')
            plt.title(f"Input ({args.input_format})")
            plt.colorbar(label='Pixel Value')

            yuv_preview = np.frombuffer(output_data, dtype=np.uint8)
            y_channel = yuv_preview[:args.width * args.height].reshape((args.height, args.width))
            plt.subplot(1, 2, 2)
            plt.imshow(y_channel, cmap='gray')
            plt.title("YUV (Y-channel)")
            plt.colorbar(label='Pixel Value')

            plt.tight_layout()
            plt.show()
                
        elif args.output_format == 'raw10_plain':
            # Save RAW output (as 10-bit)
            raw_output_path = os.path.splitext(args.output)[0] + '_output.raw10'
            input_data.tofile(raw_output_path)
            print(f"RAW output saved as: {raw_output_path}")

            # Show input and RAW preview
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(input_data, cmap='gray')
            plt.title(f"Input ({args.input_format})")
            plt.colorbar(label='Pixel Value')

            plt.subplot(1, 2, 2)
            plt.imshow(input_data, cmap='gray')
            plt.title("RAW Output")
            plt.colorbar(label='Pixel Value')

            plt.tight_layout()
            plt.show()

        elif args.output_format == 'bmp':
            # Convert RAW to RGB and save as BMP
            rgb_image = raw10_to_rgb(input_data, args.width, args.height, args.bayer_pattern)
            bmp_output_path = os.path.splitext(args.output)[0] + '.bmp'
            Image.fromarray(rgb_image).save(bmp_output_path)
            print(f"BMP output saved as: {bmp_output_path}")

            # Save YUV output if needed
            yuv_output_path = os.path.splitext(args.output)[0] + '.yuv'
            yuv_data = rgb_to_yuv420(rgb_image)
            with open(yuv_output_path, 'wb') as f:
                f.write(yuv_data)
            print(f"YUV output saved as: {yuv_output_path}")

            # Show input and BMP preview
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(input_data, cmap='gray')
            plt.title(f"Input ({args.input_format})")
            plt.colorbar(label='Pixel Value')

            plt.subplot(1, 2, 2)
            plt.imshow(rgb_image)
            plt.title("BMP Output")
            plt.colorbar(label='Pixel Value')

            plt.tight_layout()
            plt.show()
            
        print(f"\nConversion completed: {args.output}")
        
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()
