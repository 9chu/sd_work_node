#!python3
# -*- coding: utf-8 -*-
from typing import Optional
from pydantic import BaseModel


DEVICE_HIGH_MEMORY = 0
DEVICE_MEDIUM_MEMORY = 1  # <= 8GB
DEVICE_LOW_MEMORY = 2  # <= 4GB


class DeviceOptions(BaseModel):
    half_precision: bool = False
    memory_level: int = DEVICE_HIGH_MEMORY


class StableDiffusionModelOptions(BaseModel):
    model_config_path: str
    model_check_point_path: str
    model_vae_path: Optional[str] = None
    using_penultimate_layer: bool = False
    embedding_dir_path: Optional[str] = None
