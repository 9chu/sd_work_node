#!python3
# -*- coding: utf-8 -*-
from typing import Optional, List, Union
from pydantic import BaseModel
from sd_processing import StableDiffusionProcessingResult
from upscaler_model import UpscaleResult


class RequestMessage(BaseModel):
    pass


class ResponseMessage(BaseModel):
    code: int
    msg: str
    data: BaseModel


class Tex2ImgParameters(BaseModel):
    width: int
    height: int
    prompts: str
    negative_prompts: str
    count: int = 1
    steps: int = 20
    scale: float = 7
    seed: Optional[int] = None
    module: Optional[str] = None


class Img2ImgParameters(BaseModel):
    width: int
    height: int
    initial_images: List[str]
    prompts: str
    negative_prompts: str
    steps: int = 20
    scale: float = 7
    denoise: float = 0.7
    resize_mode: int = 0
    seed: Optional[int] = None
    module: Optional[str] = None


class UpscaleParameters(BaseModel):
    image: str
    scale: float


# method: 'pull_task'
class TaskPullRequest(RequestMessage):
    pass


class TaskPullResponse(BaseModel):
    type: str  # 'none', 'tex2img', 'img2img', 'upscale'
    task_id: int
    parameters: Optional[Union[Tex2ImgParameters, Img2ImgParameters, UpscaleParameters]]


TASK_STATUS_RUNNING = 0
TASK_STATUS_FINISHED = 1
TASK_STATUS_ERROR = 2


# method: 'update_task'
class TaskStateUpdateRequest(RequestMessage):
    task_id: int
    status: int
    error_msg: Optional[str]  # when TASK_STATUS_ERROR
    progress: Optional[float]  # when TASK_STATUS_RUNNING
    result: Optional[Union[StableDiffusionProcessingResult, UpscaleResult]]  # when TASK_STATUS_FINISHED


class TaskStateUpdateResponse(BaseModel):
    pass
