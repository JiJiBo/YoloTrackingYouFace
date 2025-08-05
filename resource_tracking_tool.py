# -*- coding: utf-8 -*-
"""
resource_tracking_tool.py

封装为可复用工具类：
  - ResourceMonitor：采样 CPU、内存、GPU 资源，计算统计数据
  - FaceTracker：基于 YOLOv8+ByteTrack 的人脸检测与追踪，同时驱动 ResourceMonitor

依赖：
    pip install   psutil GPUtil
"""

import os
import psutil
import GPUtil
import statistics


class ResourceMonitor:
    """
    资源监控工具
    - cpu_samples: CPU 占用百分比列表
    - mem_samples: 内存占用 (MB) 列表
    - gpu_util_samples: GPU 利用率 (%) 列表
    - gpu_mem_samples: GPU 显存占用 (MB) 列表
    """

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(interval=None)
        self.cpu_samples = []
        self.mem_samples = []
        self.gpu_util_samples = []
        self.gpu_mem_samples = []
        self.count = 0

    def __len__(self):
        return self.count

    def record(self):
        """
        采样一次资源使用情况
        """
        cpu = self.process.cpu_percent(interval=None)
        mem = self.process.memory_info().rss / (1024 ** 2)
        self.cpu_samples.append(cpu)
        self.mem_samples.append(mem)
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            self.gpu_util_samples.append(gpu.load * 100)
            self.gpu_mem_samples.append(gpu.memoryUsed)
        self.count = self.count + 1

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


if __name__ == '__main__':
    rm = ResourceMonitor()
    rm.record()
    rm.print_summary()
