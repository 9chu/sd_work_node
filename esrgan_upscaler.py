#!python3
# -*- coding: utf-8 -*-
import torch
import esrgan_model as arch
import numpy as np
from PIL import Image
from upscaler_model import UpscalerModel
from basicsr.utils.download_util import load_file_from_url
from device import Device, CPU_TORCH_DEVICE


def fix_model_layers(crt_model, pretrained_net):
    # this code is adapted from https://github.com/xinntao/ESRGAN
    if "conv_first.weight" in pretrained_net:
        return pretrained_net

    if "model.0.weight" not in pretrained_net:
        is_realesrgan = "params_ema" in pretrained_net and "body.0.rdb1.conv1.weight" in pretrained_net["params_ema"]
        if is_realesrgan:
            raise Exception("The file is a RealESRGAN model, it can't be used as a ESRGAN model.")
        else:
            raise Exception("The file is not a ESRGAN model.")

    crt_net = crt_model.state_dict()
    load_net_clean = {}
    for k, v in pretrained_net.items():
        if k.startswith("module."):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    pretrained_net = load_net_clean

    tbd = []
    for k, v in crt_net.items():
        tbd.append(k)

    # directly copy
    for k, v in crt_net.items():
        if k in pretrained_net and pretrained_net[k].size() == v.size():
            crt_net[k] = pretrained_net[k]
            tbd.remove(k)

    crt_net["conv_first.weight"] = pretrained_net["model.0.weight"]
    crt_net["conv_first.bias"] = pretrained_net["model.0.bias"]

    for k in tbd.copy():
        if "RDB" in k:
            ori_k = k.replace("RRDB_trunk.", "model.1.sub.")
            if ".weight" in k:
                ori_k = ori_k.replace(".weight", ".0.weight")
            elif ".bias" in k:
                ori_k = ori_k.replace(".bias", ".0.bias")
            crt_net[k] = pretrained_net[ori_k]
            tbd.remove(k)

    crt_net["trunk_conv.weight"] = pretrained_net["model.1.sub.23.weight"]
    crt_net["trunk_conv.bias"] = pretrained_net["model.1.sub.23.bias"]
    crt_net["upconv1.weight"] = pretrained_net["model.3.weight"]
    crt_net["upconv1.bias"] = pretrained_net["model.3.bias"]
    crt_net["upconv2.weight"] = pretrained_net["model.6.weight"]
    crt_net["upconv2.bias"] = pretrained_net["model.6.bias"]
    crt_net["HRconv.weight"] = pretrained_net["model.8.weight"]
    crt_net["HRconv.bias"] = pretrained_net["model.8.bias"]
    crt_net["conv_last.weight"] = pretrained_net["model.10.weight"]
    crt_net["conv_last.bias"] = pretrained_net["model.10.bias"]
    return crt_net


class ESRGanUpscaler(UpscalerModel):
    def __init__(self, device: Device):
        super().__init__(device)
        self._real_device = self._device.get_optimal_device()
        if self._real_device.type == "mps":
            self._real_device = CPU_TORCH_DEVICE

        self._model_path = load_file_from_url("https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth",
                                              file_name="ESRGAN_4x.pth")

        pretrained_net = torch.load(self._model_path, map_location=self._real_device)
        self._model = arch.RRDBNet(3, 3, 64, 23, gc=32)

        pretrained_net = fix_model_layers(self._model, pretrained_net)
        self._model.load_state_dict(pretrained_net)
        self._model.to(self._real_device)
        self._model.eval()

    def _upscale(self, img: Image.Image):
        img = np.array(img)
        img = img[:, :, ::-1]
        img = np.moveaxis(img, 2, 0) / 255
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0).to(self._real_device)
        with torch.no_grad():
            output = self._model(img)
        output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = 255. * np.moveaxis(output, 0, 2)
        output = output.astype(np.uint8)
        output = output[:, :, ::-1]
        return Image.fromarray(output, "RGB")
