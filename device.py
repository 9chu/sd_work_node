#!python3
# -*- coding: utf-8 -*-
import torch
import logging
import contextlib
from options import DeviceOptions
from typing import List, Optional


CPU_TORCH_DEVICE = torch.device("cpu")


class Device:
    def __init__(self, options: DeviceOptions):
        self._logger = logging.getLogger("Device")
        self._options = options

        # M1 芯片支持
        # 使用 getattr 保证兼容性
        has_mps = getattr(torch, "has_mps", False)

        # 选择最合适的设备
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif has_mps:
            self._device = torch.device("mps")
        else:
            self._device = CPU_TORCH_DEVICE

        # 精度选项
        self._data_type = torch.float16 if options.half_precision else torch.float32

        # 设备上的模型
        self._on_device_model = None

        # tf32 支持
        try:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            self._logger.exception("Enable TF32 support fail")

    def get_options(self):
        return self._options

    def get_optimal_device(self):
        """
        返回优选设备
        """
        return self._device

    def get_data_type(self):
        """
        返回数据类型
        """
        return self._data_type

    def swap_on_device_model(self, m):
        """
        交换设备上的模型
        手动追踪哪些模型被置入显存，并在其他模型需要的时候交换到 CPU 上。
        用于低内存优化。
        :param m: 需要被提交的模型
        """
        # 如果已经在设备上了，则直接返回
        if self._on_device_model == m:
            return

        # 否则，如果已经有模型在 GPU 上，进行换出
        if self._on_device_model is not None:
            self._on_device_model.to(CPU_TORCH_DEVICE)

        if m is not None:
            m.to(self._device)
        self._on_device_model = m

    def send_everything_to_cpu(self):
        if self._on_device_model is not None:
            self._on_device_model.to(CPU_TORCH_DEVICE)
        self._on_device_model = None

    def collect_garbage(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def auto_cast(self, disable=False):
        if disable:
            return contextlib.nullcontext()

        if self._data_type == torch.float32:
            return contextlib.nullcontext()

        return torch.autocast("cuda")

    def make_random_tensor(self, shape: List[int], seed: Optional[int] = None):
        # PyTorch 当前在 Metal 上尚未实现 randn 方法
        if self.get_optimal_device().type == "mps":
            generator = torch.Generator(device=CPU_TORCH_DEVICE)
            if seed is not None:
                generator.manual_seed(seed)
            noise = torch.randn(shape, generator=generator, device=CPU_TORCH_DEVICE).to(self.get_optimal_device())
            return noise

        if seed is not None:
            torch.manual_seed(seed)
        return torch.randn(shape, device=self.get_optimal_device())
