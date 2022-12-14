#!python3
# -*- coding: utf-8 -*-
import logging
import math
import torch
import ldm.modules.attention
import ldm.modules.diffusionmodules.model
from torch import einsum
from torch.nn.functional import silu
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config, default
from einops import rearrange
from options import DeviceOptions, StableDiffusionModelOptions, DEVICE_HIGH_MEMORY, DEVICE_MEDIUM_MEMORY
from device import Device, CPU_TORCH_DEVICE
from embedding_database import EmbeddingDatabase
from prompt_parser import parse_prompt_attention
from sd_optimization import split_cross_attention_forward_invokeAI, cross_attention_attnblock_forward, \
    split_cross_attention_forward

# for type hints
import transformers.models.clip.modeling_clip
import ldm.models.diffusion.ddpm
import ldm.modules.encoders.modules
from typing import Optional


def _get_target_prompt_token_count(token_count):
    return math.ceil(max(token_count, 1) / 75) * 75


class StableDiffusionModelHijack:
    def __init__(self, device: Device, options: StableDiffusionModelOptions, embedding_db: EmbeddingDatabase):
        self.device = device
        self.options = options
        self.embedding_db = embedding_db
        self.fixes = None
        self.comments = []
        self.layers = None
        self.circular_enabled = False
        self.clip = None  # type: Optional[ldm.modules.encoders.modules.FrozenCLIPEmbedder]

    def hijack(self, m: ldm.models.diffusion.ddpm.LatentDiffusion):
        cond_stage_model = m.cond_stage_model  # type: ldm.modules.encoders.modules.FrozenCLIPEmbedder

        model_embeddings = cond_stage_model.transformer.text_model.embeddings  # type: transformers.models.clip.modeling_clip.CLIPTextTransformer
        model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
        self.clip = m.cond_stage_model = FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

        # 优化
        if True:
            ldm.modules.diffusionmodules.model.nonlinearity = silu

            if self.device.get_optimal_device() != CPU_TORCH_DEVICE:
                # ldm.modules.attention.CrossAttention.forward = xformers_attention_forward
                # ldm.modules.diffusionmodules.model.AttnBlock.forward = xformers_attnblock_forward

                ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_invokeAI

                # ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward
                # ldm.modules.diffusionmodules.model.AttnBlock.forward = cross_attention_attnblock_forward

        def flatten(el):
            flattened = [flatten(children) for children in el.children()]
            res = [el]
            for c in flattened:
                res += c
            return res

        self.layers = flatten(m)

    def undo_hijack(self, m):
        if type(m.cond_stage_model) == FrozenCLIPEmbedderWithCustomWords:
            m.cond_stage_model = m.cond_stage_model.wrapped

        model_embeddings = m.cond_stage_model.transformer.embeddings
        if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
            model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped

    # def apply_circular(self, enable):
    #     if self.circular_enabled == enable:
    #         return
    #
    #     self.circular_enabled = enable
    #
    #     for layer in [layer for layer in self.layers if type(layer) == torch.nn.Conv2d]:
    #         layer.padding_mode = 'circular' if enable else 'zeros'

    def tokenize(self, text):
        _, remade_batch_tokens, _, _, _, token_count = self.clip.process_text([text])
        return remade_batch_tokens[0], token_count, _get_target_prompt_token_count(token_count)
        # max_length = self.clip.max_length - 2
        # _, remade_batch_tokens, _, _, _, token_count = self.clip.process_text([text])
        # return remade_batch_tokens[0], token_count, max_length


class EmbeddingsWithFixes(torch.nn.Module):
    def __init__(self, wrapped: torch.nn.Module, embeddings: StableDiffusionModelHijack):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                emb = embedding.vec
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[0: offset + 1], emb[0: emb_len], tensor[offset + 1 + emb_len:]])
            vecs.append(tensor)

        return torch.stack(vecs)


class FrozenCLIPEmbedderWithCustomWords(torch.nn.Module):
    def __init__(self, wrapped: ldm.modules.encoders.modules.FrozenCLIPEmbedder, hijack: StableDiffusionModelHijack):
        super().__init__()
        self.wrapped = wrapped
        self.hijack = hijack
        self.tokenizer = wrapped.tokenizer
        self.token_mults = {}

        self.comma_token = [v for k, v in self.tokenizer.get_vocab().items() if k == ',</w>'][0]

        tokens_with_parens = [(k, v) for k, v in self.tokenizer.get_vocab().items() if
                              '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

    def tokenize_line(self, line, used_custom_terms, hijack_comments):
        id_end = self.wrapped.tokenizer.eos_token_id

        parsed = parse_prompt_attention(line)

        tokenized = self.wrapped.tokenizer([text for text, _ in parsed], truncation=False, add_special_tokens=False)[
            "input_ids"]

        fixes = []
        remade_tokens = []
        multipliers = []
        last_comma = -1

        comma_padding_backtrack = 20

        for tokens, (text, weight) in zip(tokenized, parsed):
            i = 0
            while i < len(tokens):
                token = tokens[i]

                embedding, embedding_length_in_tokens = self.hijack.embedding_db.find_embedding_at_position(tokens, i)

                if token == self.comma_token:
                    last_comma = len(remade_tokens)
                elif comma_padding_backtrack != 0 and max(len(remade_tokens), 1) % 75 == 0 and last_comma != -1 and \
                        len(remade_tokens) - last_comma <= comma_padding_backtrack:
                    last_comma += 1
                    reloc_tokens = remade_tokens[last_comma:]
                    reloc_mults = multipliers[last_comma:]

                    remade_tokens = remade_tokens[:last_comma]
                    length = len(remade_tokens)

                    rem = int(math.ceil(length / 75)) * 75 - length
                    remade_tokens += [id_end] * rem + reloc_tokens
                    multipliers = multipliers[:last_comma] + [1.0] * rem + reloc_mults

                if embedding is None:
                    remade_tokens.append(token)
                    multipliers.append(weight)
                    i += 1
                else:
                    emb_len = int(embedding.vec.shape[0])
                    iteration = len(remade_tokens) // 75
                    if (len(remade_tokens) + emb_len) // 75 != iteration:
                        rem = (75 * (iteration + 1) - len(remade_tokens))
                        remade_tokens += [id_end] * rem
                        multipliers += [1.0] * rem
                        iteration += 1
                    fixes.append((iteration, (len(remade_tokens) % 75, embedding)))
                    remade_tokens += [0] * emb_len
                    multipliers += [weight] * emb_len
                    used_custom_terms.append((embedding.name, embedding.checksum()))
                    i += embedding_length_in_tokens

        token_count = len(remade_tokens)
        prompt_target_length = _get_target_prompt_token_count(token_count)
        tokens_to_add = prompt_target_length - len(remade_tokens)

        remade_tokens = remade_tokens + [id_end] * tokens_to_add
        multipliers = multipliers + [1.0] * tokens_to_add

        return remade_tokens, fixes, multipliers, token_count

    def process_text(self, texts):
        used_custom_terms = []
        remade_batch_tokens = []
        hijack_comments = []
        hijack_fixes = []
        token_count = 0

        cache = {}
        batch_multipliers = []
        for line in texts:
            if line in cache:
                remade_tokens, fixes, multipliers = cache[line]
            else:
                remade_tokens, fixes, multipliers, current_token_count = self.tokenize_line(line, used_custom_terms,
                                                                                            hijack_comments)
                token_count = max(current_token_count, token_count)

                cache[line] = (remade_tokens, fixes, multipliers)

            remade_batch_tokens.append(remade_tokens)
            hijack_fixes.append(fixes)
            batch_multipliers.append(multipliers)

        return batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count

    def forward(self, text):
        batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count = \
            self.process_text(text)

        # self.hijack.comments += hijack_comments

        # if len(used_custom_terms) > 0:
        #     self.hijack.comments.append(
        #         "Used embeddings: " + ", ".join([f'{word} [{checksum}]' for word, checksum in used_custom_terms]))

        z = None
        i = 0
        while max(map(len, remade_batch_tokens)) != 0:
            rem_tokens = [x[75:] for x in remade_batch_tokens]
            rem_multipliers = [x[75:] for x in batch_multipliers]

            self.hijack.fixes = []
            for unfiltered in hijack_fixes:
                fixes = []
                for fix in unfiltered:
                    if fix[0] == i:
                        fixes.append(fix[1])
                self.hijack.fixes.append(fixes)

            tokens = []
            multipliers = []
            for j in range(len(remade_batch_tokens)):
                if len(remade_batch_tokens[j]) > 0:
                    tokens.append(remade_batch_tokens[j][:75])
                    multipliers.append(batch_multipliers[j][:75])
                else:
                    tokens.append([self.wrapped.tokenizer.eos_token_id] * 75)
                    multipliers.append([1.0] * 75)

            z1 = self.process_tokens(tokens, multipliers)
            z = z1 if z is None else torch.cat((z, z1), axis=-2)

            remade_batch_tokens = rem_tokens
            batch_multipliers = rem_multipliers
            i += 1
        return z

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        #if not opts.use_old_emphasis_implementation:
        remade_batch_tokens = [
            [self.wrapped.tokenizer.bos_token_id] + x[:75] + [self.wrapped.tokenizer.eos_token_id] for x in
            remade_batch_tokens]
        batch_multipliers = [[1.0] + x[:75] + [1.0] for x in batch_multipliers]

        CLIP_stop_at_last_layers = 2 if self.hijack.options.using_penultimate_layer else 1

        tokens = torch.asarray(remade_batch_tokens).to(self.hijack.device.get_optimal_device())
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=-CLIP_stop_at_last_layers)

        if CLIP_stop_at_last_layers > 1:
            z = outputs.hidden_states[-CLIP_stop_at_last_layers]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state

        # restoring original mean is likely not correct,
        # but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers_of_same_length = [x + [1.0] * (75 - len(x)) for x in batch_multipliers]
        batch_multipliers = torch.asarray(batch_multipliers_of_same_length).to(self.hijack.device.get_optimal_device())
        original_mean = z.mean()
        z *= batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z *= original_mean / new_mean

        return z


class StableDiffusionModel:
    """
    Stable Diffusion 模型
    """
    def __init__(self, device: Device, options: StableDiffusionModelOptions):
        self._logger = logging.getLogger("SDModel")
        self._device = device
        self._options = options

        # 从配置文件初始化模型
        self._logger.debug(f"Reading model config from {options.model_config_path}")
        model_config = OmegaConf.load(options.model_config_path)
        self._model = instantiate_from_config(model_config.model)  # type: ldm.models.diffusion.ddpm.LatentDiffusion

        # 加载权重信息
        self._logger.debug(f"Loading checkpoint from {options.model_check_point_path}")
        pl_sd = torch.load(options.model_check_point_path, map_location="cpu")
        if "state_dict" not in pl_sd:
            sd = pl_sd
        else:
            sd = pl_sd["state_dict"]
        self._model.load_state_dict(sd, strict=False)

        # 精度切换
        if self._device.get_data_type() == torch.float16:
            self._model.half()

        # 加载 VAE 信息
        if options.model_vae_path is not None:
            self._logger.debug(f"Loading VAE weights from {options.model_vae_path}")
            vae_ckpt = torch.load(options.model_vae_path, map_location="cpu")
            vae_dict = {k: v for k, v in vae_ckpt["state_dict"].items() if k[0:4] != "loss"}
            self._model.first_stage_model.load_state_dict(vae_dict)

        # 低内存优化
        if device.get_optimal_device() == CPU_TORCH_DEVICE or device.get_options().memory_level == DEVICE_HIGH_MEMORY:
            # 无内存优化，直接提交到设备
            self._logger.debug("No memory optimal applied")
            self._model.to(device.get_optimal_device())
        else:
            # 跟踪父、子模型，有的模型按照父模型整体提交，不能在子模型级别拆解
            parent_models = {}

            def send_me_to_device(m, _):
                # 获取实际要提交的模型
                model = parent_models.get(m, m)

                # 提交到 GPU
                device.swap_on_device_model(model)

            def first_stage_model_encode_wrap(self, encoder, x):
                send_me_to_device(self, None)
                return encoder(x)

            def first_stage_model_decode_wrap(self, decoder, z):
                send_me_to_device(self, None)
                return decoder(z)

            self._logger.debug("Applying memory optimization")

            # 将三个大模型从 GPU 中移除
            stored = self._model.cond_stage_model.transformer, self._model.first_stage_model, self._model.model
            self._model.cond_stage_model.transformer, self._model.first_stage_model, self._model.model =\
                None, None, None
            self._model.to(device.get_optimal_device())  # 先把移除的变动提交
            self._model.cond_stage_model.transformer, self._model.first_stage_model, self._model.model = stored

            # 增加 Hook，当使用的时候载入 GPU
            self._model.cond_stage_model.transformer.register_forward_pre_hook(send_me_to_device)
            self._model.first_stage_model.register_forward_pre_hook(send_me_to_device)
            self._model.first_stage_model.encode = lambda x, en=self._model.first_stage_model.encode:\
                first_stage_model_encode_wrap(self._model.first_stage_model, en, x)
            self._model.first_stage_model.decode = lambda z, de=self._model.first_stage_model.decode:\
                first_stage_model_decode_wrap(self._model.first_stage_model, de, z)
            parent_models[self._model.cond_stage_model.transformer] = self._model.cond_stage_model

            if device.get_options().memory_level == DEVICE_MEDIUM_MEMORY:
                self._model.model.register_forward_pre_hook(send_me_to_device)
            else:
                # 如果启用了低内存优化，则对 Diffusion Model 也进行类似处理
                diff_model = self._model.model.diffusion_model

                stored = diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks,\
                         diff_model.time_embed
                diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed =\
                    None, None, None, None
                self._model.model.to(device.get_optimal_device())
                diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed =\
                    stored

                # 增加钩子
                diff_model.time_embed.register_forward_pre_hook(send_me_to_device)
                for block in diff_model.input_blocks:
                    block.register_forward_pre_hook(send_me_to_device)
                diff_model.middle_block.register_forward_pre_hook(send_me_to_device)
                for block in diff_model.output_blocks:
                    block.register_forward_pre_hook(send_me_to_device)

        # 构造 Embedding 数据库
        self._embedding_db = EmbeddingDatabase(device, options.embedding_dir_path, self._model)

        # Hook 模型，对网络进行调优
        self._logger.debug("Hooking model")
        if options.using_penultimate_layer:
            self._logger.debug(f"Using penultimate layer")
            # self._model.cond_stage_model.return_layer = -2
            # self._model.cond_stage_model.do_final_ln = True
        hijack = StableDiffusionModelHijack(self._device, self._options, self._embedding_db)
        hijack.hijack(self._model)

        # 构造模型
        self._logger.debug("Constructing model")
        self._model.eval()

        # 加载 Embedding
        if options.embedding_dir_path is not None:
            self._logger.debug("Loading embeddings")
            self._embedding_db.load_textual_inversion_embeddings()

    def get_device(self) -> Device:
        return self._device

    def get_options(self) -> StableDiffusionModelOptions:
        return self._options

    def get_sd_model(self) -> ldm.models.diffusion.ddpm.LatentDiffusion:
        return self._model


def _test():
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
    dev = Device(dev_opt)
    sd = StableDiffusionModel(dev, sd_opt)


if __name__ == "__main__":
    _test()
