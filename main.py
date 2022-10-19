#!python3
# -*- coding: utf-8 -*-
import io
import zlib
import base64
import argparse
import asyncio
import aiohttp
import logging
import requests
import worker_messages
from PIL import Image
from typing import Optional, List
from pydantic import BaseModel
from device import Device
from options import DeviceOptions, StableDiffusionModelOptions
from sd_model import StableDiffusionModel
from sd_sampler import SAMPLER_TYPE_DDIM
from upscaler_model import UpscaleResult
from esrgan_upscaler import ESRGanUpscaler
from hypernetwork_database import HypernetworkDatabase
from sd_processing import StableDiffusionTex2ImgProcessing, StableDiffusionImg2ImgProcessing, \
    StableDiffusionSamplerParameters, StableDiffusionProcessingResult


class Configure(BaseModel):
    device: DeviceOptions
    model: StableDiffusionModelOptions
    manager_url: str
    url_path_prefix: str = "/api"
    hypernetwork_dir: Optional[str] = None
    secret: str


class Worker:
    def __init__(self, cfg: Configure):
        self._cfg = cfg
        self._device = Device(cfg.device)
        self._model = StableDiffusionModel(self._device, cfg.model)
        self._upscaler = ESRGanUpscaler(self._device)
        self._hypernetwork_db = HypernetworkDatabase(self._device,
                                                     '' if cfg.hypernetwork_dir is None else cfg.hypernetwork_dir)
        if cfg.hypernetwork_dir is not None:
            self._hypernetwork_db.load_hypernetworks()
        self._last_progress = None

    def tex2img(self, task_id: int, width: int, height: int, prompts: str, negative_prompts: str, steps: int = 20,
                scale: float = 7, seed: Optional[int] = None, count: int = 1, module: Optional[str] = None) \
            -> StableDiffusionProcessingResult:
        try:
            parameters = StableDiffusionSamplerParameters.construct(steps=steps, scale=scale)
            processor = StableDiffusionTex2ImgProcessing(self._device, self._model, parameters)

            # 施加 Hypernetwork
            if module is not None:
                self._hypernetwork_db.load_hypernetwork(module)

            # 设置 callback
            processor.set_on_progress_callback(lambda p: self._update_task_progress_block(task_id, p))

            # 执行计算过程
            self._last_progress = None
            return processor.run(width, height, SAMPLER_TYPE_DDIM, count, prompts, negative_prompts, seed)
        finally:
            self._hypernetwork_db.unload_hypernetwork()

    def img2img(self, task_id: int, width: int, height: int, prompts: str, negative_prompts: str,
                initial_images: List[Image.Image], steps: int = 20, scale: float = 7, denoise: float = 0.7,
                resize_mode: int = 0, seed: Optional[int] = None, module: Optional[str] = None) \
            -> StableDiffusionProcessingResult:
        try:
            parameters = StableDiffusionSamplerParameters.construct(steps=steps, scale=scale,
                                                                    denoising_strength=denoise)
            processor = StableDiffusionImg2ImgProcessing(self._device, self._model, parameters, resize_mode)

            # 施加 Hypernetwork
            if module is not None:
                self._hypernetwork_db.load_hypernetwork(module)

            # 设置 callback
            processor.set_on_progress_callback(lambda p: self._update_task_progress_block(task_id, p))

            # 执行计算过程
            self._last_progress = None
            return processor.run(width, height, SAMPLER_TYPE_DDIM, len(initial_images), prompts, negative_prompts, seed,
                                 initial_images)
        finally:
            self._hypernetwork_db.unload_hypernetwork()

    def upscale(self, image: Image.Image, scale: float) -> UpscaleResult:
        return self._upscaler.upscale(image, scale)

    async def run(self):
        logging.info("Begin to service")
        try:
            while True:
                # 拉取任务
                try:
                    # pull 超时 300 秒
                    resp = await self._request("pullTask", worker_messages.TaskPullRequest.construct(), 300)
                except asyncio.TimeoutError:
                    continue
                except Exception as ex:
                    logging.exception("Pulling task error")
                    await asyncio.sleep(1)
                    continue

                # 检查任务类型并分发
                resp_body = worker_messages.TaskPullResponse.construct(**resp)
                if resp_body.type == "none":
                    continue
                else:
                    task_id = resp_body.taskId
                    parameters = resp_body.parameters
                    try:
                        if resp_body.type == "tex2img":
                            parameters = worker_messages.Tex2ImgParameters.construct(**parameters)

                            # 执行 tex2img
                            result = self.tex2img(task_id, parameters.width, parameters.height, parameters.prompts,
                                                  parameters.negativePrompts, parameters.steps, parameters.scale,
                                                  parameters.seed, parameters.count, parameters.module)
                            result = worker_messages.TaskProcessingResult.construct(
                                images=result.images,
                                width=result.width,
                                height=result.height,
                                seed=result.seed,
                                prompt=result.prompt,
                                negativePrompt=result.negative_prompt,
                                samplerType=result.sampler_type
                            )
                        elif resp_body.type == "img2img":
                            parameters = worker_messages.Img2ImgParameters.construct(**parameters)

                            initial_images = []
                            for i in parameters.initialImages:
                                with io.BytesIO() as fp:
                                    fp.write(base64.b64decode(i))
                                    fp.seek(0, io.SEEK_SET)
                                    img = Image.open(fp)
                                    initial_images.append(img.convert("RGB"))
                                    del img

                            # 执行 img2img
                            result = self.img2img(task_id, parameters.width, parameters.height, parameters.prompts,
                                                  parameters.negativePrompts, initial_images, parameters.steps,
                                                  parameters.steps, parameters.denoise, parameters.resizeMode,
                                                  parameters.seed, parameters.module)
                            result = worker_messages.TaskProcessingResult.construct(
                                images=result.images,
                                width=result.width,
                                height=result.height,
                                seed=result.seed,
                                prompt=result.prompt,
                                negativePrompt=result.negative_prompt,
                                samplerType=result.sampler_type
                            )
                        elif resp_body.type == "upscale":
                            parameters = worker_messages.UpscaleParameters.construct(**parameters)

                            with io.BytesIO() as fp:
                                fp.write(base64.b64decode(parameters.image))
                                fp.seek(0, io.SEEK_SET)
                                img = Image.open(fp)
                                # 执行 upsacle
                                result = self.upscale(img, parameters.scale)
                                result = worker_messages.TaskUpscaleResult.construct(
                                    image=result.image,
                                    width=result.width,
                                    height=result.height,
                                    scale=result.scale
                                )
                        else:
                            logging.error(f"Unexpected task {resp_body.type}, id={task_id}")
                            continue
                    except Exception as ex:
                        await self._update_task_status(task_id, worker_messages.TASK_STATUS_ERROR, error_msg=str(ex))
                        continue

                    # 回包
                    await self._update_task_status(task_id, worker_messages.TASK_STATUS_FINISHED, result=result)
        except KeyboardInterrupt:
            logging.info("Receive ctrl+c, exiting main loop")

    async def _request(self, method: str, args: worker_messages.RequestMessage, timeout=300):
        headers = {"X-API-SECRET": self._cfg.secret, "Content-Type": "application/json"}
        payload = args.json()
        # 压缩数据
        if len(payload) > 4096:
            payload = zlib.compress(payload.encode('utf-8'))
            headers['Content-Encoding'] = 'deflate'
        async with aiohttp.ClientSession(self._cfg.manager_url, headers=headers) as session:
            async with session.post(f"{self._cfg.url_path_prefix}/TaskDispatch/{method}", data=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Send request fail, method={method}, status={resp.status}")
                resp_body = await resp.json()
                if resp_body["code"] != 0:
                    raise RuntimeError(f"API error, method={method}, code={resp_body['code']}, msg={resp_body['msg']}")
                return resp_body["data"]

    async def _update_task_status(self, task_id: int, status: int, error_msg: Optional[str] = None,
                                  progress: Optional[float] = None, result=None):
        try:
            req = worker_messages.TaskStateUpdateRequest.construct(taskId=task_id, status=status, errorMsg=error_msg,
                                                                   progress=progress, result=result)
            await self._request("updateTask", req, 60)  # 对于超大图片，超时不能过短
        except Exception:
            logging.exception("Update task status error")

    def _update_task_progress_block(self, task_id: int, progress: float):
        # 由于网络I/O比较费时，我们在这里限制只有增长 10% 才汇报一次
        if (self._last_progress is not None) and progress < 0.95 and abs(progress - self._last_progress) < 0.1:
            return
        self._last_progress = progress

        # 发送 HTTP 请求，此处必须阻塞发
        headers = {"X-API-SECRET": self._cfg.secret, "Content-Type": "application/json"}
        payload = {
            "taskId": task_id,
            "status": worker_messages.TASK_STATUS_RUNNING,
            "progress": progress,
        }
        try:
            requests.post(f"{self._cfg.manager_url}{self._cfg.url_path_prefix}/TaskDispatch/updateTask", json=payload, headers=headers,
                          timeout=3)
        except Exception:
            logging.exception("Update task status error")


def main():
    parser = argparse.ArgumentParser(description="Worker node for stable diffusion model")
    parser.add_argument("--config", dest="config", required=True, type=str, help="Configure file path")
    parser.add_argument("--verbose", dest="verbose", required=False, action="store_true", default=False,
                        help="Show debug logs")
    args = parser.parse_args()

    # 初始化日志
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger_format = logging.Formatter("[%(asctime)s][%(levelname)s][%(module)s:%(funcName)s:%(lineno)d] %(message)s")
    logger_output = logging.StreamHandler()
    logger_output.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger_output.setFormatter(logger_format)
    logger.addHandler(logger_output)

    # 加载配置文件
    cfg = Configure.parse_file(args.config)

    # 初始化
    worker = Worker(cfg)
    asyncio.get_event_loop().run_until_complete(worker.run())


if __name__ == "__main__":
    main()
