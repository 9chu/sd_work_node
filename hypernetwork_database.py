#!python3
# -*- coding: utf-8 -*-
import os
import torch
import logging
from pathlib import Path
from options import DEVICE_HIGH_MEMORY
from device import Device, CPU_TORCH_DEVICE
from sd_model import set_global_hypernetwork


class HyperLogic(torch.nn.Module):
    logic_multiplier = 1.0

    def __init__(self, dim, heads=0):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim*2)
        self.linear2 = torch.nn.Linear(dim*2, dim)

    def forward(self, _x):
        return _x + (self.linear2(self.linear1(_x)) * HyperLogic.logic_multiplier)


class HypernetworkDatabase:
    def __init__(self, device: Device, hypernetwork_dir: str):
        self._logger = logging.getLogger("HypernetworkDatabase")
        self._device = device
        self._hypernetwork_dir = hypernetwork_dir
        self._dict = {}

        # 针对存在内存优化的场合，先加载到 CPU 侧，需要时再读入
        if self._device.get_options().memory_level == DEVICE_HIGH_MEMORY:
            self._default_device = self._device.get_optimal_device()
        else:
            self._default_device = CPU_TORCH_DEVICE
        self._last_loaded_hypernetwork = None

    def _load_module(self, path):
        path = Path(path)
        if not path.is_file():
            self._logger.error(f"Path '{path}' is not a file")
            return False

        network = {
            768: (HyperLogic(768).to(self._default_device), HyperLogic(768).to(self._default_device)),
            1280: (HyperLogic(1280).to(self._default_device), HyperLogic(1280).to(self._default_device)),
            640: (HyperLogic(640).to(self._default_device), HyperLogic(640).to(self._default_device)),
            320: (HyperLogic(320).to(self._default_device), HyperLogic(320).to(self._default_device)),
        }

        state_dict = torch.load(path, map_location=CPU_TORCH_DEVICE)
        for key in state_dict.keys():
            if isinstance(key, str):  # StableDiffusionWebUI 训练结果兼容处理
                continue
            network[key][0].load_state_dict(state_dict[key][0])
            network[key][1].load_state_dict(state_dict[key][1])

        # 注册
        name = os.path.splitext(os.path.basename(path))[0]
        self._dict[name] = network

    def load_hypernetworks(self):
        self._dict.clear()

        for fn in os.listdir(self._hypernetwork_dir):
            try:
                fullfn = os.path.join(self._hypernetwork_dir, fn)

                if os.stat(fullfn).st_size == 0:
                    continue

                self._load_module(fullfn)
            except Exception:
                self._logger.exception(f"Error loading hypernetwork {fn}:")

        self._logger.info(f"Loaded a total of {len(self._dict)} hypernetworks.")

    def contains_hypernetwork(self, name: str):
        return name in self._dict

    def load_hypernetwork(self, name: str):
        # 卸载上一个 hypernetwork
        if self._last_loaded_hypernetwork is not None:
            self.unload_hypernetwork()

        hn = self._dict[name]

        # 如果当前设备不是期望的设备，则进行一个加载操作
        if self._default_device != self._device.get_optimal_device():
            for k in hn:
                layer = hn[k]
                for sub_layer in layer:
                    sub_layer.to(self._device.get_optimal_device())

        # 设置到 CrossAttension
        set_global_hypernetwork(hn)
        self._last_loaded_hypernetwork = name

    def unload_hypernetwork(self):
        set_global_hypernetwork(None)

        if self._last_loaded_hypernetwork is None:
            return

        # 如果开启了内存优化，则进行一个卸载操作
        if self._default_device != self._device.get_optimal_device():
            hn = self._dict[self._last_loaded_hypernetwork]
            for k in hn:
                layer = hn[k]
                for sub_layer in layer:
                    sub_layer.to(self._default_device)
        self._last_loaded_hypernetwork = None
