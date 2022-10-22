#!python3
# -*- coding: utf-8 -*-
import torch
import prompt_parser
import ldm.models.diffusion.ddpm
import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms
import numpy as np
from abc import abstractmethod
from PIL import Image
from typing import Optional, List
from pydantic import BaseModel
from device import Device
from options import StableDiffusionModelOptions
from sd_model import StableDiffusionModel
from prompt_parser import MulticondLearnedConditioning, ScheduledPromptConditioning


SAMPLER_TYPE_DDIM = 1
SAMPLER_TYPE_PLMS = 2


class StableDiffusionSamplerParameters(BaseModel):
    ddim_eta: float = 0
    ddim_discretize: str = "uniform"
    steps: int = 50
    scale: float = 7
    denoising_strength: float = 0.7


class StableDiffusionSamplingProgress:
    def __init__(self, sampler, percent: float, latent):
        self._sampler = sampler
        self._percent = percent
        self._latent = latent

    def get_percent(self) -> float:
        return self._percent

    def to_image(self) -> Image:
        sd_model = self._sampler.get_sd_model()
        x_sample = sd_model.decode_first_stage(self._latent[0:1].type(sd_model.dtype))[0]
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
        x_sample = x_sample.astype(np.uint8)
        return Image.fromarray(x_sample)


class StableDiffusionSamplerBase:
    def __init__(self, device: Device, options: StableDiffusionModelOptions,
                 sd_model: ldm.models.diffusion.ddpm.LatentDiffusion):
        self._device = device
        self._options = options
        self._model = sd_model
        self._on_progress_callback = None

    def get_sd_model(self):
        return self._model

    def set_on_progress_callback(self, callback):
        self._on_progress_callback = callback

    def calc_number_of_needed_noises(self, p: StableDiffusionSamplerParameters):
        return 0

    def update_sampler_noises(self, noises: List[torch.Tensor]):
        pass

    @abstractmethod
    def sample(self, p: StableDiffusionSamplerParameters, x: torch.Tensor, conditioning: MulticondLearnedConditioning,
               unconditional_conditioning: List[List[ScheduledPromptConditioning]],
               mask: Optional[torch.Tensor] = None, nmask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def sample_img2img(self, p: StableDiffusionSamplerParameters, x: torch.Tensor, noise: torch.Tensor,
                       conditioning: MulticondLearnedConditioning,
                       unconditional_conditioning: List[List[ScheduledPromptConditioning]], steps: Optional[int] = None,
                       mask: Optional[torch.Tensor] = None, nmask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()

    def _setup_img2img_steps(self, p, steps=None):
        if steps is not None:
            steps = int((steps or p.steps) / min(p.denoising_strength, 0.999)) if p.denoising_strength > 0 else 0
            t_enc = p.steps - 1
        else:
            steps = p.steps
            t_enc = int(min(p.denoising_strength, 0.999) * steps)
        return steps, t_enc

    def _update_progress(self, progress: float, current_latent: torch.Tensor):
        if self._on_progress_callback:
            self._on_progress_callback(StableDiffusionSamplingProgress(self, float(np.clip(progress, 0, 1)),
                                                                       current_latent))


class VanillaStableDiffusionSampler(StableDiffusionSamplerBase):
    def __init__(self, device: Device, options: StableDiffusionModelOptions,
                 sd_model: ldm.models.diffusion.ddpm.LatentDiffusion, sampler_type: int):
        super().__init__(device, options, sd_model)
        if sampler_type == SAMPLER_TYPE_DDIM:
            self._sampler = ldm.models.diffusion.ddim.DDIMSampler(sd_model)
            self._org_p_sample_ddim = self._sampler.p_sample_ddim
            self._sampler.p_sample_ddim = self._sample_hook
        else:
            assert sampler_type == SAMPLER_TYPE_PLMS
            self._sampler = ldm.models.diffusion.plms.PLMSSampler(sd_model)
            self._org_p_sample_ddim = self._sampler.p_sample_plms
            self._sampler.p_sample_plms = self._sample_hook
        self._mask = None
        self._nmask = None
        self._eta = None
        self._init_latent = None
        self._step = 0
        self._total_step = 0

    def _sample_hook(self, x_dec: torch.Tensor, cond: MulticondLearnedConditioning, ts,
                     unconditional_conditioning: List[List[ScheduledPromptConditioning]], *args, **kwargs):
        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self._step)
        unconditional_conditioning = prompt_parser.reconstruct_cond_batch(unconditional_conditioning, self._step)

        assert all([len(conds) == 1 for conds in conds_list]), \
            "composition via AND is not supported for DDIM/PLMS samplers"
        cond = tensor

        # for DDIM, shapes must match, we can't just process cond and uncond independently;
        # filling unconditional_conditioning with repeats of the last vector to match length is
        # not 100% correct but should work well enough
        if unconditional_conditioning.shape[1] < cond.shape[1]:
            last_vector = unconditional_conditioning[:, -1:]
            last_vector_repeated = last_vector.repeat([1, cond.shape[1] - unconditional_conditioning.shape[1], 1])
            unconditional_conditioning = torch.hstack([unconditional_conditioning, last_vector_repeated])
        elif unconditional_conditioning.shape[1] > cond.shape[1]:
            unconditional_conditioning = unconditional_conditioning[:, :cond.shape[1]]

        if self._mask is not None:
            img_orig = self._sampler.model.q_sample(self._init_latent, ts)
            x_dec = img_orig * self._mask + self._nmask * x_dec

        res = self._org_p_sample_ddim(x_dec, cond, ts, unconditional_conditioning=unconditional_conditioning, *args,
                                      **kwargs)

        # 保存中间结果
        if self._mask is not None:
            current_latent = self._init_latent * self._mask + self._nmask * res[1]
        else:
            current_latent = res[1]

        # 更新进度
        self._step += 1
        self._update_progress(self._step / self._total_step, current_latent)
        return res

    def _initialize(self, p: StableDiffusionSamplerParameters, mask: Optional[torch.Tensor] = None,
                    nmask: Optional[torch.Tensor] = None):
        self._eta = p.ddim_eta
        self._mask = mask
        self._nmask = nmask
        self._step = 0

    def sample(self, p: StableDiffusionSamplerParameters, x: torch.Tensor, conditioning: MulticondLearnedConditioning,
               unconditional_conditioning: List[List[ScheduledPromptConditioning]], steps: Optional[int] = None,
               mask: Optional[torch.Tensor] = None, nmask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._initialize(p, mask, nmask)
        self._init_latent = None

        # 有些特殊的 Step 会失败，需要 +1 重试
        steps = steps or p.steps
        try:
            self._total_step = steps
            ret, _ = self._sampler.sample(S=steps, conditioning=conditioning, batch_size=int(x.shape[0]),
                                          shape=x[0].shape, verbose=False,
                                          unconditional_guidance_scale=p.scale,
                                          unconditional_conditioning=unconditional_conditioning, x_T=x,
                                          eta=self._eta)
        except Exception:
            self._total_step = steps + 1
            ret, _ = self._sampler.sample(S=steps+1, conditioning=conditioning, batch_size=int(x.shape[0]),
                                          shape=x[0].shape, verbose=False,
                                          unconditional_guidance_scale=p.scale,
                                          unconditional_conditioning=unconditional_conditioning, x_T=x,
                                          eta=self._eta)
        return ret

    def sample_img2img(self, p: StableDiffusionSamplerParameters, x: torch.Tensor, noise: torch.Tensor,
                       conditioning: MulticondLearnedConditioning,
                       unconditional_conditioning: List[List[ScheduledPromptConditioning]], steps: Optional[int] = None,
                       mask: Optional[torch.Tensor] = None, nmask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert hasattr(self._sampler, "p_sample_ddim"), "img2img only supports ddim sampler"

        steps, t_enc = self._setup_img2img_steps(p, steps)

        self._initialize(p, mask, nmask)

        # 有些特殊的 Step 会失败，需要 +1 重试
        try:
            self._total_step = steps
            self._sampler.make_schedule(ddim_num_steps=steps,  ddim_eta=self._eta, ddim_discretize=p.ddim_discretize,
                                        verbose=False)
        except Exception:
            self._total_step = steps + 1
            self._sampler.make_schedule(ddim_num_steps=steps+1, ddim_eta=self._eta, ddim_discretize=p.ddim_discretize,
                                        verbose=False)

        x1 = self._sampler.stochastic_encode(
            x, torch.tensor([t_enc] * int(x.shape[0])).to(self._device.get_optimal_device()), noise=noise)

        self._init_latent = x

        samples = self._sampler.decode(x1, conditioning, t_enc, unconditional_guidance_scale=p.scale,
                                       unconditional_conditioning=unconditional_conditioning)
        return samples


def create_sampler(model: StableDiffusionModel, sampler_type: int):
    if sampler_type == SAMPLER_TYPE_DDIM or sampler_type == SAMPLER_TYPE_PLMS:
        return VanillaStableDiffusionSampler(model.get_device(), model.get_options(), model.get_sd_model(),
                                             sampler_type)
    else:
        raise RuntimeError(f"Unexpected sampler_type: {sampler_type}")
