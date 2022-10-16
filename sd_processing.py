#!python3
# -*- coding: utf-8 -*-
import io
import torch
import base64
import random
import logging
import prompt_parser
import numpy as np
from abc import abstractmethod
from PIL import Image
from pydantic import BaseModel
from typing import Optional, List
from device import Device
from sd_model import StableDiffusionModel
from sd_sampler import StableDiffusionSamplerParameters, StableDiffusionSamplerBase, create_sampler
from options import DeviceOptions, StableDiffusionModelOptions, DEVICE_HIGH_MEMORY


class StableDiffusionProcessingResult(BaseModel):
    images: List[str]
    width: int
    height: int
    seed: int
    prompt: str
    negative_prompt: str
    sampler_type: int
    sampler_parameters: StableDiffusionSamplerParameters


class StableDiffusionProcessingBase:
    def __init__(self, device: Device, model: StableDiffusionModel, parameters: StableDiffusionSamplerParameters):
        self._device = device
        self._model = model
        self._parameters = parameters
        self._sampler: Optional[StableDiffusionSamplerBase] = None
        self._width = 0
        self._height = 0

    def _create_random_tensors(self, shape: List[int], seeds: List[int]):
        xs = []

        # 保证在 batch 情况下使用的种子在单独使用时也是一致的
        if len(seeds) > 1:
            sampler_noises = [[] for _ in range(self._sampler.calc_number_of_needed_noises(self._parameters))]
        else:
            sampler_noises = None

        for i, seed in enumerate(seeds):
            # 使用 GPU 侧的种子
            noise = self._device.make_random_tensor(shape, seed)

            if sampler_noises is not None:
                cnt = self._sampler.calc_number_of_needed_noises(self._parameters)
                for j in range(cnt):
                    sampler_noises[j].append(self._device.make_random_tensor(shape))

            xs.append(noise)

        if sampler_noises is not None:
            self._sampler.update_sampler_noises([torch.stack(n).to(self._device.get_optimal_device())
                                                 for n in sampler_noises])

        x = torch.stack(xs).to(self._device.get_optimal_device())
        return x

    @abstractmethod
    def _init(self, sampler_type: int, all_prompts: List[str], all_seeds: List[int]):
        raise NotImplementedError()

    @abstractmethod
    def _sample(self, conditioning: prompt_parser.MulticondLearnedConditioning,
                unconditional_conditioning: List[List[prompt_parser.ScheduledPromptConditioning]], seeds: List[int],
                initial_images: Optional[List[Image.Image]]):
        raise NotImplementedError()

    def run(self, width: int, height: int, sampler_type: int, count: int, prompt: str, negative_prompt: str,
            seed: Optional[int] = None, initial_images: Optional[List[Image.Image]] = None):
        assert initial_images is None or len(initial_images) == count
        self._width = width
        self._height = height

        self._device.collect_garbage()

        if seed is None or seed == -1:
            seed = int(random.randrange(4294967294))

        all_prompts = count * [prompt]
        all_seeds = [int(seed) + x for x in range(len(all_prompts))]  # 给一组图片很小的种子偏移量

        output_images = []
        with torch.no_grad(), self._model.get_sd_model().ema_scope():
            with self._device.auto_cast():
                self._init(sampler_type, all_prompts, all_seeds)

            for n in range(count):
                prompts = all_prompts[n: (n + 1)]
                seeds = all_seeds[n: (n + 1)]
                images = None if initial_images is None else initial_images[n: (n + 1)]
                assert len(prompts) != 0
                assert images is None or len(images) == len(seeds)

                # 准备正/负向词汇
                with self._device.auto_cast():
                    uc = prompt_parser.get_learned_conditioning(self._model.get_sd_model(),
                                                                len(prompts) * [negative_prompt],
                                                                self._parameters.steps)
                    c = prompt_parser.get_multicond_learned_conditioning(self._model.get_sd_model(), prompts,
                                                                         self._parameters.steps)

                # 采样
                with self._device.auto_cast():
                    samples_ddim = self._sample(c, uc, seeds, images)
                samples_ddim = samples_ddim.to(self._device.get_data_type())
                with self._device.auto_cast(disable=(samples_ddim.dtype == self._device.get_data_type())):
                    x_samples_ddim = self._model.get_sd_model().decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                # 清理
                del samples_ddim
                if self._device.get_options().memory_level != DEVICE_HIGH_MEMORY:
                    self._device.send_everything_to_cpu()
                self._device.collect_garbage()

                # 保存图片
                # 这里理论上应该只能解出一个结果？
                for i, x_sample in enumerate(x_samples_ddim):
                    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                    x_sample = x_sample.astype(np.uint8)
                    image = Image.fromarray(x_sample)
                    # 编码到 base64
                    with io.BytesIO() as image_bytes:
                        image.save(image_bytes, format="PNG")
                        image_base64 = base64.b64encode(image_bytes.getvalue()).decode("ascii")
                        output_images.append(image_base64)
                    del image

                # 清理
                del x_samples_ddim
                self._device.collect_garbage()

        return StableDiffusionProcessingResult(images=output_images, width=width, height=height, seed=seed,
                                               prompt=prompt, negative_prompt=negative_prompt,
                                               sampler_type=sampler_type, sampler_parameters=self._parameters)


class StableDiffusionTex2ImgProcessing(StableDiffusionProcessingBase):
    def __init__(self, device: Device, model: StableDiffusionModel, parameters: StableDiffusionSamplerParameters):
        super().__init__(device, model, parameters)

    def _init(self, sampler_type: int, all_prompts: List[str], all_seeds: List[int]):
        self._sampler = create_sampler(self._model, sampler_type)

    def _sample(self, conditioning: prompt_parser.MulticondLearnedConditioning,
                unconditional_conditioning: List[List[prompt_parser.ScheduledPromptConditioning]],
                seeds: List[int], initial_images: Optional[List[Image.Image]]):
        const_c = 4
        const_f = 8
        x = self._create_random_tensors([const_c, self._height // const_f, self._width // const_f], seeds)
        samples = self._sampler.sample(self._parameters, x, conditioning, unconditional_conditioning)
        return samples


def _resize_image(resize_mode, im, width, height):
    def resize(img, w, h):
        resampler = (Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
        return img.resize((w, h), resample=resampler)

    if resize_mode == 0:  # 直接缩放
        res = resize(im, width, height)
    elif resize_mode == 1:  # Crop and resize
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:  # Resize and fill
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                      box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                      box=(fill_width + src_w, 0))
    return res


class StableDiffusionImg2ImgProcessing(StableDiffusionProcessingBase):
    def __init__(self, device: Device, model: StableDiffusionModel, parameters: StableDiffusionSamplerParameters,
                 resize_mode: int = 0):
        super().__init__(device, model, parameters)
        self._resize_mode = resize_mode

    def _init(self, sampler_type: int, all_prompts: List[str], all_seeds: List[int]):
        self._sampler = create_sampler(self._model, sampler_type)

    def _sample(self, conditioning: prompt_parser.MulticondLearnedConditioning,
                unconditional_conditioning: List[List[prompt_parser.ScheduledPromptConditioning]],
                seeds: List[int], initial_images: Optional[List[Image.Image]]):
        assert initial_images is not None and len(seeds) == len(initial_images)

        # 图像预处理
        processed_images = []
        for img in initial_images:
            image = img.convert("RGB")
            image = _resize_image(self._resize_mode, image, self._width, self._height)
            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)
            processed_images.append(image)
        image = torch.from_numpy(np.array(processed_images))
        image = 2. * image - 1.
        image = image.to(self._device.get_optimal_device())

        init_latent = self._model.get_sd_model().get_first_stage_encoding(
            self._model.get_sd_model().encode_first_stage(image))

        const_c = 4
        const_f = 8
        x = self._create_random_tensors([const_c, self._height // const_f, self._width // const_f], seeds)
        samples = self._sampler.sample_img2img(self._parameters, init_latent, x, conditioning,
                                               unconditional_conditioning)

        del x
        self._device.collect_garbage()
        return samples


def _test():
    from hypernetwork_database import HypernetworkDatabase
    from esrgan_upscaler import ESRGanUpscaler

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger_format = logging.Formatter("[%(asctime)s][%(levelname)s][%(module)s:%(funcName)s:%(lineno)d] %(message)s")
    logger_output = logging.StreamHandler()
    logger_output.setLevel(logging.DEBUG)
    logger_output.setFormatter(logger_format)
    logger.addHandler(logger_output)

    dev_opt = DeviceOptions.construct()
    sd_opt = StableDiffusionModelOptions.construct(
        model_config_path="./data/config.yaml",
        model_check_point_path="./data/model.ckpt",
        model_vae_path="./data/model.vae.pt")
    sampler_opt = StableDiffusionSamplerParameters.construct()
    dev = Device(dev_opt)
    hnd = HypernetworkDatabase(dev, "./data/modules")
    sd = StableDiffusionModel(dev, sd_opt)
    hnd.load_hypernetworks()
    p = StableDiffusionTex2ImgProcessing(dev, sd, sampler_opt)
    hnd.load_hypernetwork("furry_kemono")
    ret = p.run(512, 512, 1, 1, "loli, masterpiece", "ugly, deformed, lowres")
    print(ret)
    with io.BytesIO() as fp:
        fp.write(base64.b64decode(ret.images[0]))
        fp.seek(0, io.SEEK_SET)
        img = Image.open(fp)
        #img.save('0.png')
        p2 = StableDiffusionImg2ImgProcessing(dev, sd, sampler_opt, 0)
        ret2 = p2.run(512, 512, 1, 1, "fox_girl, fox_ears, loli, masterpiece", "ugly", None, [img])
        print(ret2)
    hnd.unload_hypernetwork()
    with io.BytesIO() as fp:
        fp.write(base64.b64decode(ret2.images[0]))
        fp.seek(0, io.SEEK_SET)
        img = Image.open(fp)
        #img.save('1.png')
        fp.seek(0, io.SEEK_SET)
        upscaler = ESRGanUpscaler(dev)
        img_scaled = upscaler.upscale(img, 4)
        #img_scaled.save('2.png')


if __name__ == "__main__":
    _test()
