import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QComboBox, QFileDialog, QMessageBox
from PIL import Image
import matplotlib.pyplot as plt

class ConverterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('RAW Converter')

        layout = QVBoxLayout()

        self.input_file = QLineEdit(self)
        layout.addWidget(QLabel('输入文件:'))
        layout.addWidget(self.input_file)

        input_button = QPushButton('选择输入文件', self)
        input_button.clicked.connect(self.browse_input_file)
        layout.addWidget(input_button)

        self.output_file = QLineEdit(self)
        layout.addWidget(QLabel('输出文件:'))
        layout.addWidget(self.output_file)

        output_button = QPushButton('选择输出文件', self)
        output_button.clicked.connect(self.browse_output_file)
        layout.addWidget(output_button)

        self.width_input = QLineEdit(self)
        layout.addWidget(QLabel('宽度:'))
        layout.addWidget(self.width_input)

        self.height_input = QLineEdit(self)
        layout.addWidget(QLabel('高度:'))
        layout.addWidget(self.height_input)

        self.input_format = QComboBox(self)
        self.input_format.addItems(['raw8', 'raw10_plain', 'raw10_mipi', 'raw12_plain', 'raw12_mipi'])
        layout.addWidget(QLabel('输入格式:'))
        layout.addWidget(self.input_format)

        self.output_format = QComboBox(self)
        self.output_format.addItems(['raw10', 'yuv420', 'bmp'])
        layout.addWidget(QLabel('输出格式:'))
        layout.addWidget(self.output_format)

        convert_button = QPushButton('转换', self)
        convert_button.clicked.connect(self.convert)
        layout.addWidget(convert_button)

        self.setLayout(layout)

    def browse_input_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择输入文件", "", "RAW Files (*.raw);;All Files (*)")
        if filename:
            self.input_file.setText(filename)

    def browse_output_file(self):
        filename, _ = QFileDialog.getSaveFileName(self, "选择输出文件", "", "RAW Files (*.raw);;YUV Files (*.yuv);;BMP Files (*.bmp);;All Files (*)")
        if filename:
            self.output_file.setText(filename)

    def convert(self):
        input_file = self.input_file.text()
        output_file = self.output_file.text()
        input_format = self.input_format.currentText()
        output_format = self.output_format.currentText()

        try:
            width = int(self.width_input.text())
            height = int(self.height_input.text())
        except ValueError:
            QMessageBox.critical(self, "错误", "宽度和高度必须是整数！")
            return

        try:
            # 读取输入文件
            with open(input_file, 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.uint8)

            print(f"读取的原始数据长度: {len(raw_data)}")

            # 检查数据大小是否与宽度和高度匹配
            expected_size = self.calculate_expected_size(input_format, width, height)

            print(f"预期大小: {expected_size}, 实际大小: {len(raw_data)}")

            if len(raw_data) != expected_size:
                QMessageBox.critical(self, "错误", f"输入数据大小不匹配，预期大小: {expected_size}，实际大小: {len(raw_data)}")
                return

            # 处理输入图像
            rgb_image = self.process_raw_data(raw_data, input_format, width, height)

            # 打印 RGB 图像的形状
            print(f"生成的 RGB 图像形状: {rgb_image.shape}")

            # 保存输出图像
            if rgb_image is not None:
                Image.fromarray(rgb_image).save(output_file)
                QMessageBox.information(self, "成功", "转换完成！")

                # 显示对比图
                self.display_raw_images(raw_data, rgb_image, input_format, output_format, width)

        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def calculate_expected_size(self, input_format, width, height):
        """计算预期的输入数据大小"""
        if input_format == 'raw10_plain':
            return width * height * 2  # 每个像素 2 字节
        elif input_format == 'raw10_mipi':
            return int(width * height * 1.25)  # 每个像素 1.25 字节
        elif input_format == 'raw12_plain':
            return width * height * 2  # 每个像素 2 字节
        elif input_format == 'raw12_mipi':
            return int(width * height * 1.5)  # 每个像素 1.5 字节
        else:
            return width * height  # raw8 每个像素 1 字节

    def process_raw_data(self, raw_data, input_format, width, height):
        """处理原始数据并返回 RGB 图像"""
        if input_format == 'raw8':
            return self.raw8_to_rgb(raw_data, width, height)
        elif input_format == 'raw10_plain':
            return self.raw10_to_rgb(raw_data, width, height)
        elif input_format == 'raw10_mipi':
            return self.raw10_mipi_to_rgb(raw_data, width, height)
        elif input_format == 'raw12_plain':
            return self.raw12_to_rgb(raw_data, width, height)
        elif input_format == 'raw12_mipi':
            return self.raw12_mipi_to_rgb(raw_data, width, height)

    def display_raw_images(self, raw_data, output_image, input_format, output_format, width):
        """显示输入和输出RAW图像的对比"""
        input_image = self.raw_data_to_image(raw_data, input_format, width)

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
        plt.title(f"Output RGB ({output_format})")
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()

    def raw_data_to_image(self, raw_data, input_format, width):
        """将原始数据转换为图像"""
        if input_format in ['raw10_plain', 'raw10_mipi']:
            raw_data = np.frombuffer(raw_data, dtype=np.uint16).reshape((-1, width))  # 使用传入的宽度
            return (raw_data >> 2).astype(np.uint8)  # 转换为 8 位图像
        elif input_format in ['raw12_plain', 'raw12_mipi']:
            raw_data = np.frombuffer(raw_data, dtype=np.uint16).reshape((-1, width))  # 使用传入的宽度
            return (raw_data >> 4).astype(np.uint8)  # 转换为 8 位图像
        else:
            return raw_data.reshape((-1, width))  # 使用传入的宽度

    def raw8_to_rgb(self, raw_data, width, height):
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        raw_data = raw_data.reshape((height, width))

        for y in range(height):
            for x in range(width):
                pixel_value = raw_data[y, x]  # 直接使用 8 位数据
                rgb_image[y, x] = [pixel_value, pixel_value, pixel_value]  # 灰度图

        return rgb_image

    def raw10_to_rgb(self, raw_data, width, height):
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        raw_data = np.frombuffer(raw_data, dtype=np.uint16).reshape((height, width))

        for y in range(height):
            for x in range(width):
                pixel_value = raw_data[y, x] >> 2  # 假设是 10 位数据
                rgb_image[y, x] = [pixel_value, pixel_value, pixel_value]  # 灰度图

        return rgb_image

    def raw10_mipi_to_rgb(self, raw_data, width, height):
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        pixel_index = 0

        for i in range(0, len(raw_data), 5):
            if pixel_index >= width * height:
                break

            # 处理每5个字节
            byte0 = raw_data[i]
            byte1 = raw_data[i + 1]
            byte2 = raw_data[i + 2]
            byte3 = raw_data[i + 3]
            byte4 = raw_data[i + 4]

            # 提取像素值
            r = ((byte0 << 2) & 0xFF) | ((byte4 >> 6) & 0x03)
            g = ((byte1 << 2) & 0xFF) | ((byte4 >> 4) & 0x03)
            b = ((byte2 << 2) & 0xFF) | ((byte4 >> 2) & 0x03)

            # 确保像素值在 0-255 范围内
            r = min(max(r, 0), 255)
            g = min(max(g, 0), 255)
            b = min(max(b, 0), 255)

            rgb_image[pixel_index // width, pixel_index % width] = [r, g, b]

            # 打印调试信息
            #print(f"Pixel {pixel_index}: {rgb_image[pixel_index // width, pixel_index % width]}")

            pixel_index += 4  # 每5个字节表示4个像素

        return rgb_image

    def raw12_to_rgb(self, raw_data, width, height):
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        raw_data = np.frombuffer(raw_data, dtype=np.uint16).reshape((height, width))

        for y in range(height):
            for x in range(width):
                pixel_value = raw_data[y, x] >> 4  # 假设是 12 位数据
                rgb_image[y, x] = [pixel_value, pixel_value, pixel_value]  # 灰度图

        return rgb_image

    def raw12_mipi_to_rgb(self, raw_data, width, height):
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        pixel_index = 0

        for i in range(0, len(raw_data), 6):
            if pixel_index >= width * height:
                break

            # 处理每6个字节
            byte0 = raw_data[i]
            byte1 = raw_data[i + 1]
            byte2 = raw_data[i + 2]
            byte3 = raw_data[i + 3]
            byte4 = raw_data[i + 4]
            byte5 = raw_data[i + 5]

            # 提取像素值
            rgb_image[pixel_index // width, pixel_index % width] = [
                ((byte0 << 4) & 0xFF) | (byte1 >> 4),  # 第一个像素
                ((byte2 << 4) & 0xFF) | (byte1 & 0x0F),  # 第二个像素
                ((byte3 << 4) & 0xFF) | (byte4 >> 4)   # 第三个像素
            ]

            pixel_index += 4  # 每6个字节表示4个像素

        return rgb_image

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ConverterApp()
    ex.show()
    sys.exit(app.exec_()) 