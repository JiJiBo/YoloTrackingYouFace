# -*- coding: utf-8 -*-
"""
resource_tracking_tool.py

封装为可复用工具类：
  - ResourceMonitor：采样 CPU、内存、GPU 资源，计算统计数据，支持时间戳记录和绘制统计图
  - FaceTracker：基于 YOLOv8+ByteTrack 的人脸检测与追踪，同时驱动 ResourceMonitor（可后续扩展）

依赖：
    pip install psutil GPUtil matplotlib
"""

import os
import time

import psutil
import GPUtil
import statistics
import datetime
import matplotlib.pyplot as plt


class ResourceMonitor:
    """
    资源监控工具
    - cpu_samples: CPU 占用百分比列表
    - mem_samples: 内存占用 (MB) 列表
    - gpu_util_samples: GPU 利用率 (%) 列表
    - gpu_mem_samples: GPU 显存占用 (MB) 列表
    - timestamps: 每次采样对应的时间戳列表
    """

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        # 第一次调用，用于初始化内部计时（不记录）
        self.process.cpu_percent(interval=None)
        self.cpu_samples = []
        self.mem_samples = []
        self.gpu_util_samples = []
        self.gpu_mem_samples = []
        self.timestamps = []
        self.count = 0

    def __len__(self):
        return self.count

    def record(self):
        """
        采样一次资源使用情况，并记录当前时间
        """
        now = datetime.datetime.now()
        cpu = self.process.cpu_percent(interval=None)
        mem = self.process.memory_info().rss / (1024 ** 2)

        self.timestamps.append(now)
        self.cpu_samples.append(cpu)
        self.mem_samples.append(mem)

        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            self.gpu_util_samples.append(gpu.load * 100)
            self.gpu_mem_samples.append(gpu.memoryUsed)
        self.count += 1

    def summary(self):
        """
        计算并返回资源使用统计字典：最大、最小、平均、样本数
        """

        def _summarize(samples):
            return {
                'max': round(max(samples), 2),
                'min': round(min(samples), 2),
                'avg': round(statistics.mean(samples), 2),
                'count': len(samples)
            }

        data = {
            'CPU (%)': _summarize(self.cpu_samples),
            'Memory (MB)': _summarize(self.mem_samples)
        }
        if self.gpu_util_samples:
            data['GPU Util (%)'] = _summarize(self.gpu_util_samples)
            data['GPU Mem (MB)'] = _summarize(self.gpu_mem_samples)
        return data

    def print_summary(self):
        print("\n=== 资源使用统计 ===")
        for k, v in self.summary().items():
            print(f"{k} —— 最大: {v['max']}, 最小: {v['min']}, 平均: {v['avg']} (样本: {v['count']})")

    def plot(self, show: bool = True, save_path: str = None):
        """
        绘制资源使用随时间变化的折线图
        参数：
          - show: 是否直接展示图形
          - save_path: 如果提供路径，则保存到文件
        """
        if not self.timestamps:
            raise RuntimeError("没有采样数据，请先调用 record() 方法。")

        # 转换时间轴为 matplotlib 可识别格式
        times = self.timestamps

        # 绘制 CPU 和内存
        plt.figure()
        plt.plot(times, self.cpu_samples)
        plt.title('CPU Usage (%) Over Time')
        plt.xlabel('Time')
        plt.ylabel('CPU (%)')
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_cpu.png")
        if show:
            plt.show()
        plt.close()

        plt.figure()
        plt.plot(times, self.mem_samples)
        plt.title('Memory Usage (MB) Over Time')
        plt.xlabel('Time')
        plt.ylabel('Memory (MB)')
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_mem.png")
        if show:
            plt.show()
        plt.close()

        # 如果有 GPU 数据，则绘制 GPU 曲线
        if self.gpu_util_samples:
            plt.figure()
            plt.plot(times, self.gpu_util_samples)
            plt.title('GPU Utilization (%) Over Time')
            plt.xlabel('Time')
            plt.ylabel('GPU Util (%)')
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_gpu_util.png")
            if show:
                plt.show()
            plt.close()

            plt.figure()
            plt.plot(times, self.gpu_mem_samples)
            plt.title('GPU Memory Usage (MB) Over Time')
            plt.xlabel('Time')
            plt.ylabel('GPU Mem (MB)')
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_gpu_mem.png")
            if show:
                plt.show()
            plt.close()


if __name__ == '__main__':
    rm = ResourceMonitor()
    # 示例：每秒采样 5 次
    for _ in range(5):
        rm.record()
        time.sleep(1)
    rm.print_summary()
    rm.plot()
