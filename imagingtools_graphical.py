import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

class BayerImageAnalyzer:
    def __init__(self, rgb_image):
        if not isinstance(rgb_image, np.ndarray):
            raise ValueError("输入图像必须是 NumPy 数组。")
        self.image_data = rgb_image
        self.info = {}
        self.analyze_image_properties()  # 立即分析图像特征

    def analyze_image_properties(self):
        # 在这里填充 self.info 字典
        self.info['shape'] = self.image_data.shape
        self.info['channels'] = self.image_data.shape[2] if len(self.image_data.shape) == 3 else 1
        # 其他分析逻辑...

    def load_and_analyze(self):
        print("""加载并分析图像""")
        try:
            if not isinstance(self.image_path, str):
                raise ValueError("图像路径必须是字符串类型。")
            
            print(f"尝试打开图像: {self.image_path}")
            
            with Image.open(self.image_path) as img:
                self.info['format'] = img.format
                self.info['mode'] = img.mode
                self.info['size'] = img.size
                self.info['bits'] = img.bits
                self.info['channels'] = 1 if img.mode == 'L' else 3  # 灰度图像或 RGB
            
            self.image_data = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            if self.image_data is None:
                raise ValueError("图像未能正确加载，请检查文件路径和格式。")
            
            print("图像加载成功")
            print(f"分析信息: {self.info}")  # 打印信息以确认填充
            return True
        except OSError as e:
            print(f"文件打开错误: {str(e)}")
            return False
        except Exception as e:
            print(f"错误: {str(e)}")
            return False
    
    def check_bayer_pattern(self):
        print("""检查是否可能是Bayer图像""")
        if self.image_data is None:
            return False
        
        # 检查是否为单通道图像
        if len(self.image_data.shape) == 2:
            print("这是灰度图像，不包含Bayer模式")
            return False
            
        # 如果是3通道图像，检查是否已经被解码为RGB
        if len(self.image_data.shape) == 3:
            print("这是已经解码的RGB图像，不是原始Bayer数据")
            return False
    
    def save_analysis_report(self, output_path="./image_analysis_report.txt"):
        print("""保存分析报告""")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Image Analysis Report\n")
            f.write("=" * 50 + "\n\n")

            # 检查并写入信息
            if not self.info:
                f.write("No analysis information available.\n")
                return

            for key, value in self.info.items():
                f.write(f"{key}: {value}\n")

            if self.info.get('channels', 0) == 1:
                f.write("\nThis is a grayscale image. Possible reasons:\n")
                f.write("1. A monochrome sensor was used.\n")
                f.write("2. The image was converted to grayscale during processing.\n")
                f.write("3. BMP was saved in 8-bit grayscale format.\n")
            else:
                f.write("\nThis is a color image.\n")
                f.write(f"Number of channels: {self.info.get('channels', 'Unknown')}\n")
    
    def show_histogram(self):
        """显示图像直方图"""
        if self.image_data is None:
            print("No image data available to display histogram.")
            return
        
        plt.figure(figsize=(10, 5))
        
        if len(self.image_data.shape) == 2:
            plt.hist(self.image_data.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
            plt.title('Grayscale Histogram')
        else:
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                plt.hist(self.image_data[:, :, i].ravel(), bins=256, range=[0, 256], color=color, alpha=0.5)
            plt.title('RGB Channel Histogram')
        
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)  # 添加网格以提高可读性
        plt.xlim([0, 256])  # 设置 x 轴范围
        plt.legend(['Blue Channel', 'Green Channel', 'Red Channel'])  # 添加图例
        plt.title('Image Histogram')  # 添加图像标题
        plt.show()

# ////////////////

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
    """自动检测RAW格式"""
    # 计算理论文件大小
    raw8_size = width * height          # RAW8: 每个像素1字节
    plain_size = width * height * 2      # Plain RAW12: 每个像素2字节
    mipi_raw10_size = (width * height * 10 + 7) // 8  # MIPI RAW10: 每个像素10位打包
    mipi_raw12_size = (width * height * 12 + 7) // 8  # MIPI RAW12: 每个像素12位打包

    file_size = len(raw_data)
    print(f"\n文件大小分析:")
    print(f"实际文件大小: {file_size} 字节")
    print(f"RAW8 预期大小: {raw8_size} 字节")
    print(f"Plain RAW12 预期大小: {plain_size} 字节")
    print(f"MIPI RAW10 预期大小: {mipi_raw10_size} 字节")
    print(f"MIPI RAW12 预期大小: {mipi_raw12_size} 字节")

    # 检查文件大小以确定格式
    if file_size == raw8_size:
        return 'raw8', read_raw8(raw_data, width, height)
    elif file_size == plain_size:
        return 'plain', read_raw12_plain(raw_data, width, height)
    elif file_size == mipi_raw10_size:
        return 'mipi_raw10', read_raw10_mipi(raw_data, width, height)
    elif file_size == mipi_raw12_size:
        return 'mipi_raw12', read_raw12_mipi(raw_data, width, height)
    else:
        raise ValueError("无法识别的文件格式，文件大小不匹配。")

def analyze_image(image):
    """分析图像并返回统计信息"""
    mean_value = np.mean(image)
    min_value = np.min(image)
    max_value = np.max(image)
    std_dev = np.std(image)

    return {
        'mean': mean_value,
        'min': min_value,
        'max': max_value,
        'std_dev': std_dev
    }

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
    plt.title(f"Input RAW ({input_format})")
    plt.colorbar()

    # 显示输出图像
    plt.subplot(122)
    plt.imshow(output_image, cmap='gray')
    plt.title(f"Output RAW10 Plain ({output_format})")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def convert_raw12_to_raw10(input_file, output_file, width, height, output_format='plain'):
    """转换RAW12到RAW10, 自动检测输入格式"""
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

import numpy as np
import cv2
import time
from typing import Union

def rgb_to_yuv420(image: np.ndarray) -> bytes:
    """
    将 RGB 图像转换为 YUV420 格式
    
    参数:
        image (np.ndarray): 输入RGB图像，支持uint8或uint16格式
                           可以是2D(灰度图)或3D(RGB图)
                           
    返回:
        bytes: YUV420格式的字节数据
        
    异常:
        TypeError: 输入类型错误
        ValueError: 图像格式或尺寸错误
    """
    try:
        # 输入类型检查
        if not isinstance(image, np.ndarray):
            raise TypeError("输入必须是numpy数组")

        # 确保图像数据连续存储，提高性能
        image = np.ascontiguousarray(image)
        
        # 检查和处理图像维度
        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=-1)
        elif image.ndim != 3:
            raise ValueError("输入图像维度必须是2或3")
            
        # 检查通道数
        if image.shape[2] != 3:
            raise ValueError("输入图像必须是RGB格式(3通道)")
            
        # 检查图像尺寸（YUV420要求宽高为2的倍数）
        if image.shape[0] % 2 != 0 or image.shape[1] % 2 != 0:
            raise ValueError("图像尺寸必须是2的倍数")

        # 数据类型转换
        if image.dtype == np.uint16:
            # 使用位运算代替除法，提高性能
            image = (image >> 8).astype(np.uint8)
        elif image.dtype != np.uint8:
            # 处理其他数据类型（如float）
            image = (image * 255).clip(0, 255).astype(np.uint8)

        # 使用OpenCV进行颜色空间转换
        try:
            yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420)
        except cv2.error as e:
            raise RuntimeError(f"OpenCV转换失败: {str(e)}")

        # 直接返回字节数据，避免额外的内存复制
        return bytes(yuv_image)

    except Exception as e:
        # 添加更多上下文信息到错误消息
        raise type(e)(f"RGB到YUV420转换失败: {str(e)}") from e



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

# 创建主窗口
root = tk.Tk()
root.title("RAW格式转换器")

# 全局变量来存储完整路径
input_files_full_path = []
output_file_full_path = ""
output_format_var = tk.StringVar(value='yuv420')  # 默认输出格式

def select_input_files():
    """选择输入文件"""
    global input_files_full_path
    file_paths = filedialog.askopenfilenames(title="选择输入文件", filetypes=[("RAW Files", "*.raw"), ("All Files", "*.*")])
    if file_paths:  # 确保用户选择了文件
        input_files_full_path = list(file_paths)  # 存储完整路径
        input_file_entry.delete(0, tk.END)
        
        # 显示选择的文件名
        display_files = ', '.join([os.path.basename(path) for path in input_files_full_path])
        input_file_entry.insert(0, display_files)

def get_format_from_filename(filename):
    """从文件名获取格式"""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.bmp':
        return 'bmp'
    elif ext == '.yuv':
        return 'yuv420'
    elif ext == '.raw10':
        return 'raw10_plain'
    return 'bmp'  # 默认格式

def select_output_file():
    """选择输出文件"""
    global output_file_full_path
    file_path = filedialog.asksaveasfilename(
        defaultextension=".bmp",
        filetypes=[("BMP Files", "*.bmp"), 
                  ("YUV Files", "*.yuv"), 
                  ("RAW10 Plain Files", "*.raw10")]
    )
    if file_path:  # 确保用户选择了文件
        output_file_full_path = file_path
        output_file_entry.delete(0, tk.END)
        output_file_entry.insert(0, os.path.basename(output_file_full_path))
        
        # 根据选择的文件扩展名自动设置输出格式
        output_format = get_format_from_filename(file_path)
        output_format_var.set(output_format)

def show_progress_dialog(message):
    """显示进度对话框"""
    progress_window = tk.Toplevel(root)
    progress_window.title("转换进度")
    progress_label = tk.Label(progress_window, text=message)
    progress_label.pack(padx=20, pady=20)
    progress_window.geometry("300x100")
    return progress_window

def show_image(image, title):
    """在主线程中显示图像"""
    plt.figure(figsize=(15, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.colorbar(label='Pixel Value')
    plt.show()

def save_as_bmp(image_data, output_file):
    """将图像数据保存为 BMP 格式"""
    # 确保图像数据是 uint8 类型
    if image_data.dtype != np.uint8:
        image_data = (image_data / np.max(image_data) * 255).astype(np.uint8)

    # 使用 PIL 保存为 BMP
    img = Image.fromarray(image_data)
    img.save(output_file, format='BMP')

def save_yuv420(yuv_data, output_file):
    """保存 YUV420 数据到文件"""
    with open(output_file, 'wb') as f:
        f.write(yuv_data)

def generate_output_filename(output_file):
    """生成带时间戳的输出文件名，但保持原始的基本名称和扩展名"""
    # 分离路径、基本名称和扩展名
    dir_path = os.path.dirname(output_file)
    basename = os.path.basename(output_file)
    name, ext = os.path.splitext(basename)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 组合新文件名: 原始名称_时间戳.扩展名
    new_filename = f"{name}_{timestamp}{ext}"
    
    # 如果有目录路径，则组合完整路径
    if dir_path:
        return os.path.join(dir_path, new_filename)
    return new_filename

def bayer_to_rgb(bayer_image, width, height, bayer_pattern):
    """将 Bayer 模式图像转换为 RGB 格式"""
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 根据 Bayer 模式进行转换
    if bayer_pattern == 'RGGB':
        rgb_image[0::2, 0::2, 0] = bayer_image[0::2, 0::2]  # R
        rgb_image[0::2, 1::2, 1] = bayer_image[0::2, 1::2]  # G
        rgb_image[1::2, 0::2, 1] = bayer_image[1::2, 0::2]  # G
        rgb_image[1::2, 1::2, 2] = bayer_image[1::2, 1::2]  # B

        # 双线性插值填充 G 通道
        rgb_image[0::2, 1::2, 1] = (bayer_image[0::2, 0::2] + bayer_image[0::2, 2::2]) // 2  # 上方 G
        rgb_image[1::2, 0::2, 1] = (bayer_image[0::2, 0::2] + bayer_image[2::2, 0::2]) // 2  # 左侧 G

    elif bayer_pattern == 'GRBG':
        rgb_image[0::2, 0::2, 1] = bayer_image[0::2, 0::2]  # G
        rgb_image[0::2, 1::2, 0] = bayer_image[0::2, 1::2]  # R
        rgb_image[1::2, 0::2, 0] = bayer_image[1::2, 0::2]  # R
        rgb_image[1::2, 1::2, 1] = bayer_image[1::2, 1::2]  # G

        # 双线性插值填充 R 通道
        rgb_image[0::2, 1::2, 0] = (bayer_image[0::2, 0::2] + bayer_image[0::2, 2::2]) // 2  # 上方 R
        rgb_image[1::2, 0::2, 0] = (bayer_image[0::2, 0::2] + bayer_image[2::2, 0::2]) // 2  # 左侧 R

    elif bayer_pattern == 'GBRG':
        rgb_image[0::2, 0::2, 1] = bayer_image[0::2, 0::2]  # G
        rgb_image[0::2, 1::2, 2] = bayer_image[0::2, 1::2]  # B
        rgb_image[1::2, 0::2, 2] = bayer_image[1::2, 0::2]  # B
        rgb_image[1::2, 1::2, 1] = bayer_image[1::2, 1::2]  # G

        # 双线性插值填充 B 通道
        rgb_image[0::2, 1::2, 2] = (bayer_image[0::2, 0::2] + bayer_image[0::2, 2::2]) // 2  # 上方 B
        rgb_image[1::2, 0::2, 2] = (bayer_image[0::2, 0::2] + bayer_image[2::2, 0::2]) // 2  # 左侧 B

    elif bayer_pattern == 'BGGR':
        rgb_image[0::2, 0::2, 2] = bayer_image[0::2, 0::2]  # B
        rgb_image[0::2, 1::2, 1] = bayer_image[0::2, 1::2]  # G
        rgb_image[1::2, 0::2, 1] = bayer_image[1::2, 0::2]  # G
        rgb_image[1::2, 1::2, 0] = bayer_image[1::2, 1::2]  # R

        # 双线性插值填充 G 通道
        rgb_image[0::2, 1::2, 1] = (bayer_image[0::2, 0::2] + bayer_image[0::2, 2::2]) // 2  # 上方 G
        rgb_image[1::2, 0::2, 1] = (bayer_image[0::2, 0::2] + bayer_image[2::2, 0::2]) // 2  # 左侧 G

    else:
        raise ValueError("不支持的 Bayer 模式")

    # 确保 RGB 图像的值在 0 到 255 的范围内
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    return rgb_image

def run_conversion():
    """直接在主线程中运行转换操作"""
    # 获取用户选择的输入和输出文件路径
    input_file = input_file_entry.get()
    output_file = output_file_entry.get()

    # 从输出文件名获取格式
    output_format = get_format_from_filename(output_file)
    
    # 使用完整的输出文件路径生成带时间戳的文件名
    output_file_with_timestamp = generate_output_filename(output_file)
    print(f"将要保存的文件名: {output_file_with_timestamp}，格式: {output_format}")
    
    try:
        # 使用全局变量中的完整路径
        #input_files = input_files_full_path
        #output_file = output_file_full_path
        #output_format = output_format_var.get()  # 获取当前选择的输出格式

        # 打印实际路径以进行调试
        #print(f"输入文件路径: {input_files}")
        print(f"输出文件路径: {output_file}")

        # 路径验证
        if not os.path.exists(input_file):
            messagebox.showerror("错误", f"输入文件不存在：{input_file}")
            return

        # 读取输入文件
        with open(input_file, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint8)

        # 获取图像尺寸
        try:
            width = int(width_entry.get())
            height = int(height_entry.get())
        except ValueError:
            messagebox.showerror("错误", "宽度和高度必须是有效的数字")
            return

        # 根据输入格式处理数据
        if input_format_var.get() == 'bayer':
            bayer_image = raw_data.reshape((height, width))
            rgb_image = bayer_to_rgb(bayer_image, width, height, bayer_pattern_var.get())
        elif input_format_var.get() == 'raw8':
            input_data = read_raw8(raw_data, width, height)
            rgb_image = input_data
        elif input_format_var.get() == 'raw10_plain':
            input_data = read_raw10_plain(raw_data, width, height)
            rgb_image = input_data
        elif input_format_var.get() == 'raw10_mipi':
            input_data = read_raw10_mipi(raw_data, width, height)
            rgb_image = input_data
        elif input_format_var.get() == 'raw12_plain':
            input_data = read_raw12_plain(raw_data, width, height)
            rgb_image = input_data
        elif input_format_var.get() == 'raw12_mipi':
            input_data = read_raw12_mipi(raw_data, width, height)
            rgb_image = input_data
        else:
            raise ValueError("不支持的输入格式")

        # 转换为10位
        output_image = (input_data >> 2).astype(np.uint16)

        # 保存时使用带时间戳的文件名
        if output_format == 'raw10_plain':
            output_image.tofile(output_file_with_timestamp)
        elif output_format == 'bmp':
            save_as_bmp(rgb_image, output_file_with_timestamp)
        elif output_format == 'yuv420':
            yuv_data = rgb_to_yuv420(output_image)
            save_yuv420(yuv_data, output_file_with_timestamp)

        # BayerImageAnalyzer if output format is BMP
        if output_format == 'bmp':
            analyzer = BayerImageAnalyzer(rgb_image)  # 传递 rgb_image
            print("图像基本信息：")
            for key, value in analyzer.info.items():
                print(f"{key}: {value}")
            
            analyzer.check_bayer_pattern()
            analyzer.save_analysis_report()
            analyzer.show_histogram()

        # 调用 display_raw_images 显示输入和输出图像的对比
        display_raw_images(input_data, rgb_image, input_format_var.get(), output_format)

    except Exception as e:
        messagebox.showerror("错误", str(e))

def rgb_to_yuv420p(image):
    """将 RGB 图像转换为 YUV420P 格式"""
    if image.ndim == 2:  # 如果是单通道图像
        image = np.stack((image,) * 3, axis=-1)  # 复制到 3 个通道

    # 确保图像是 RGB 格式
    if image.shape[2] != 3:
        raise ValueError("输入图像必须是 RGB 格式")

    # 将 16 位图像转换为 8 位
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)  # 缩放到 8 位范围

    # 使用 OpenCV 进行转换
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420)

    # 返回 YUV420P 数据的字节格式
    return yuv_image.tobytes()

def save_yuv420p(yuv_data, width, height, output_file):
    """保存 YUV420P 数据到文件"""
    with open(output_file, 'wb') as f:
        f.write(yuv_data)

# 输入文件选择
tk.Label(root, text="输入文件:").grid(row=0, column=0)
input_file_entry = tk.Entry(root, width=50)
input_file_entry.grid(row=0, column=1)
tk.Button(root, text="浏览", command=select_input_files).grid(row=0, column=2)

# 输出文件选择
tk.Label(root, text="输出文件:").grid(row=1, column=0)
output_file_entry = tk.Entry(root, width=50)
output_file_entry.grid(row=1, column=1)
tk.Button(root, text="浏览", command=select_output_file).grid(row=1, column=2)

# 图像宽度和高度
tk.Label(root, text="宽度:").grid(row=2, column=0)
width_entry = tk.Entry(root)
width_entry.grid(row=2, column=1)

tk.Label(root, text="高度:").grid(row=3, column=0)
height_entry = tk.Entry(root)
height_entry.grid(row=3, column=1)

# 输入格式选择
tk.Label(root, text="输入格式:").grid(row=4, column=0)
input_format_var = tk.StringVar(value='raw12_plain')
input_format_options = ['raw8', 'raw10_plain', 'raw10_mipi', 'raw12_plain', 'raw12_mipi']
input_format_menu = tk.OptionMenu(root, input_format_var, *input_format_options)
input_format_menu.grid(row=4, column=1)

# 输出格式选择
#tk.Label(root, text="输出格式:").grid(row=5, column=0)
#output_format_var = tk.StringVar(value='yuv420')
#output_format_options = ['raw10_plain', 'yuv420', 'bmp']
#output_format_menu = tk.OptionMenu(root, output_format_var, *output_format_options)
#output_format_menu.grid(row=5, column=1)

# Bayer模式选择
tk.Label(root, text="Bayer模式:").grid(row=6, column=0)
bayer_pattern_var = tk.StringVar(value='RGGB')
bayer_pattern_options = ['RGGB', 'BGGR', 'GRBG', 'GBRG']
bayer_pattern_menu = tk.OptionMenu(root, bayer_pattern_var, *bayer_pattern_options)
bayer_pattern_menu.grid(row=6, column=1)

# 转换按钮
tk.Button(root, text="开始转换", command=run_conversion).grid(row=7, columnspan=3)

# 启动主循环
root.mainloop()