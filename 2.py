import cv2
import numpy as np
import mss
import torch
from ultralytics import YOLO
import time
from collections import deque
import sys
import os
import psutil
import ctypes
import win32api
import win32con

class UltraLowLatencyMouse:
    def __init__(self):
        """初始化超低延迟鼠标控制器"""
        self._dpi_scale = self._get_dpi_scaling()
        self.screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        self.screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        
        # 预计算坐标转换参数
        self._65535_div_width = 65535 / self.screen_width
        self._65535_div_height = 65535 / self.screen_height
        
    def _get_dpi_scaling(self):
        """获取系统DPI缩放比例"""
        user32 = ctypes.windll.user32
        hdc = user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        user32.ReleaseDC(0, hdc)
        return dpi / 96.0
    
    def move_instant(self, x, y):
        """无延迟直接移动鼠标"""
        # 极速坐标转换（去除所有不必要的计算）
        nx = int(x * self._dpi_scale * self._65535_div_width)
        ny = int(y * self._dpi_scale * self._65535_div_height)
        
        # 使用最低延迟的鼠标移动API
        ctypes.windll.user32.mouse_event(
            win32con.MOUSEEVENTF_ABSOLUTE | win32con.MOUSEEVENTF_MOVE,
            nx, ny, 0, 0
        )

class InstantTracker:
    def __init__(self):
        """即时目标追踪器"""
        self.last_position = None
        self.last_update = time.time()
        
    def update(self, position):
        """更新目标位置"""
        self.last_position = position
        self.last_update = time.time()
        
    def get_position(self):
        """获取最新位置（无预测）"""
        return self.last_position

class ZeroLatencyDetector:
    def __init__(self):
        """初始化零延迟检测器"""
        # 硬件优化
        self._set_realtime_priority()
        
        # 检测参数
        self.model_path = "best.pt"
        self.input_size = 256  # 更小的输入尺寸
        self.conf_thres = 0.4  # 较低的置信度阈值
        self.iou_thres = 0.3   # 较低的IOU阈值
        self.roi = [640, 360, 1280, 720]  # 聚焦中心区域
        
        # 初始化组件
        self._init_model()
        self._init_capture()
        self.mouse = UltraLowLatencyMouse()
        self.tracker = InstantTracker()
        
    def _set_realtime_priority(self):
        """设置实时优先级"""
        try:
            pid = os.getpid()
            ps = psutil.Process(pid)
            ps.nice(psutil.REALTIME_PRIORITY_CLASS)
            ps.cpu_affinity([0])  # 绑定到单个CPU核心
        except Exception as e:
            print(f"⚠️ 无法设置实时优先级: {str(e)}")

    def _init_model(self):
        """初始化极简模型"""
        print("🚀 初始化零延迟模型...")
        try:
            # 最小化模型加载时间
            self.model = YOLO(self.model_path)
            self.model.fuse()
            
            # 极简预热
            dummy = torch.randn(1, 3, self.input_size, self.input_size)
            with torch.inference_mode():  # 比torch.no_grad()更快
                self.model(dummy)
                
            print(f"✅ 极速模型加载完成")
        except Exception as e:
            print(f"❌ 模型初始化失败: {str(e)}")
            sys.exit(1)
            
    def _init_capture(self):
        """初始化超低延迟截图"""
        self.sct = mss.mss()
        self.capture_area = {
            'left': self.roi[0],
            'top': self.roi[1],
            'width': self.roi[2]-self.roi[0],
            'height': self.roi[3]-self.roi[1]
        }
        
    def _ultra_fast_preprocess(self, frame):
        """极速预处理（仅缩放）"""
        return cv2.resize(frame, (self.input_size, self.input_size))
    
    def run(self):
        print("\n🎯 零延迟瞄准辅助已启动 | ESC退出")
        try:
            while True:
                # 极速截屏
                frame = np.array(self.sct.grab(self.capture_area), dtype=np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # 极简预处理
                processed = self._ultra_fast_preprocess(frame)
                
                # 即时推理
                with torch.inference_mode():
                    results = self.model(
                        processed,
                        imgsz=self.input_size,
                        conf=self.conf_thres,
                        iou=self.iou_thres,
                        verbose=False
                    )
                
                # 即时响应
                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0]  # 直接取第一个检测结果
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # 极速坐标计算（无缩放）
                    center_x = (x1 + x2) // 2 + self.roi[0]
                    center_y = (y1 + y2) // 2 + self.roi[1]
                    
                    # 无延迟移动
                    self.mouse.move_instant(center_x, center_y)
                
                # 退出检测（使用直接键盘检测）
                if win32api.GetAsyncKeyState(0x1B) & 0x8000:  # ESC键
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 用户终止")
        finally:
            cv2.destroyAllWindows()
            self.sct.close()

if __name__ == "__main__":
    if not ctypes.windll.shell32.IsUserAnAdmin():
        ctypes.windll.user32.MessageBoxW(0, "需要管理员权限!", "权限错误", 0x10)
        sys.exit(1)
        
    ZeroLatencyDetector().run()