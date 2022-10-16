#!python3
# -*- coding: utf-8 -*-
import io
import base64
from abc import abstractmethod
from PIL import Image
from device import Device
from pydantic import BaseModel


class UpscaleResult(BaseModel):
    image: str
    width: int
    height: int
    scale: float


class UpscalerModel:
    def __init__(self, device: Device):
        self._device = device
        self._scale = 1

    @abstractmethod
    def _upscale(self, img: Image.Image):
        raise NotImplementedError()

    def upscale(self, img: Image.Image, scale: float) -> UpscaleResult:
        self._scale = scale
        dest_w = img.width * scale
        dest_h = img.height * scale
        for i in range(0, 3):  # 最多缩放 3 次
            if img.width >= dest_w and img.height >= dest_h:
                break
            img = self._upscale(img)
        if img.width != dest_w or img.height != dest_h:
            sampler = (Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
            img = img.resize((int(dest_w), int(dest_h)), resample=sampler)

        # 编码到 base64
        with io.BytesIO() as image_bytes:
            img.save(image_bytes, format="PNG")
            image_base64 = base64.b64encode(image_bytes.getvalue()).decode("ascii")

        ret = UpscaleResult.construct(image=image_base64, width=img.width, height=img.height, scale=scale)
        del img
        return ret
