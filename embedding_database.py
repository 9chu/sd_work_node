#!python3
# -*- coding: utf-8 -*-
import os
import torch
import logging
from device import Device


class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        torch.save(embedding_data, filename)

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f"{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}"
        return self.cached_checksum


class EmbeddingDatabase:
    def __init__(self, device: Device, embeddings_dir: str, sd_model):
        self.logger = logging.getLogger("EmbeddingDatabase")
        self.device = device
        self.embeddings_dir = embeddings_dir
        self.sd_model = sd_model
        self.ids_lookup = {}
        self.word_embeddings = {}

    def _process_file(self, path, filename):
        name = os.path.splitext(filename)[0]

        data = torch.load(path, map_location="cpu")

        # textual inversion embeddings
        if "string_to_param" in data:
            param_dict = data["string_to_param"]
            # fix for torch 1.12.1 loading saved file from torch 1.11
            if hasattr(param_dict, "_parameters"):
                param_dict = getattr(param_dict, "_parameters")
            assert len(param_dict) == 1, "embedding file has multiple terms in it"
            emb = next(iter(param_dict.items()))[1]
        # diffuser concepts
        elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
            assert len(data.keys()) == 1, "embedding file has multiple terms in it"

            emb = next(iter(data.values()))
            if len(emb.shape) == 1:
                emb = emb.unsqueeze(0)
        else:
            raise Exception(f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept")

        vec = emb.detach().to(self.device.get_optimal_device(), dtype=torch.float32)
        embedding = Embedding(vec, name)
        embedding.step = data.get("step", None)
        embedding.sd_checkpoint = data.get("hash", None)
        embedding.sd_checkpoint_name = data.get("sd_checkpoint_name", None)
        self.register_embedding(embedding, self.sd_model)

    def register_embedding(self, embedding, model):
        self.word_embeddings[embedding.name] = embedding

        ids = model.cond_stage_model.tokenizer([embedding.name], add_special_tokens=False)["input_ids"][0]

        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []

        self.ids_lookup[first_id] = sorted(self.ids_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]),
                                           reverse=True)
        return embedding

    def load_textual_inversion_embeddings(self):
        self.ids_lookup.clear()
        self.word_embeddings.clear()

        for fn in os.listdir(self.embeddings_dir):
            try:
                fullfn = os.path.join(self.embeddings_dir, fn)

                if os.stat(fullfn).st_size == 0:
                    continue

                self._process_file(fullfn, fn)
            except Exception:
                self.logger.exception(f"Error loading embedding {fn}:")

        self.logger.info(f"Loaded a total of {len(self.word_embeddings)} textual inversion embeddings.")

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None
