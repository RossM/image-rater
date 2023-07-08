import json

import torch
import open_clip
import numpy

from PIL import Image
from pathlib import Path
from torch import Tensor

class EmbeddingCache:
    def __init__(self, cache_path: Path, config: str = 'ViT-H-14'):
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        self.model = model
        self.preprocess = preprocess
        self.config = config
        self.cache_file = cache_path / (config + ".json")
        self.cache = {}
        
        try:
            with open(self.cache_file, "r") as f:
                for line in f:
                    try:
                        cache_entry = json.loads(line)
                        key = cache_entry['key']
                        value = self.decode(cache_entry['value'])
                        self.cache[key] = value
                    except Exception as e:
                        print(e)
        except Exception as e:
            print(e)
            
    def encode(self, tensor: Tensor):
        return self.config + ":" + tensor.numpy().tobytes().hex()
        
    def decode(self, encoded: str):
        prefix, hex = encoded.split(':')
        if prefix != self.config:
            raise ValueError(f"Expected OpenCLIP config '{self.config}' but got '{prefix}'")
        return Tensor(numpy.frombuffer(bytearray.fromhex(hex)))
        
    def get_embedding(self, filename: str, image: Image):
        embedding = self.cache.get(str(filename), None)
        if embedding != None:
            print(f"Found cached embedding for {filename}")
            return embedding
    
        image = self.preprocess(image).unsqueeze(0)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        embedding = image_features.flatten().cpu()
        self.cache[filename] = embedding
        print(f"Calculated embedding for {filename}")
        try:
            with open(self.cache_file, 'a') as f:
                f.write(json.dumps({
                    "key": filename,
                    "value": self.encode(embedding)
                }) + "\n")
        except Exception as e:
            print(e)
        return embedding