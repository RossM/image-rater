import json

import torch
import open_clip
import numpy

import modules.sd_models as sd
from modules import devices

from PIL import Image
from pathlib import Path
from torch import Tensor
from torch.utils.data import DataLoader

class EmbeddingCache:
    def __init__(self, cache_path: Path, config: str = 'ViT-H-14'):
        if config == 'ViT-L-14':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
            batch_size = 16
            embed_length = 768
        elif config == 'DataComp-ViT-L-14':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')
            batch_size = 16
            embed_length = 768
        elif config == 'ViT-H-14':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
            batch_size = 4
            embed_length = 768
        elif config == 'convnext_large_d_320':
            model, _, preprocess = open_clip.create_model_and_transforms('convnext_large_d_320', pretrained='laion2b_s29b_b131k_ft_soup')
            batch_size = 16
            embed_length = 768
        else:
            raise ValueError(f"Unknown config {config}")

        self.device = devices.get_device_for('image_rater')
        self.model = model.to(device=self.device)
        self.preprocess = preprocess
        self.config = config
        self.cache_file = cache_path / (config + ".json")
        self.cache = {}
        self.batch_size = batch_size
        self.embed_length = embed_length

        try:
            with open(self.cache_file, "r") as f:
                for line in f:
                    try:
                        cache_entry = json.loads(line)
                        key = cache_entry['key']
                        value = self.decode(cache_entry['value'])
                        self.cache[key] = value
                    except Exception as e:
                        print(f"Error reading cache entry: {e}")
                        print(e.__traceback__)
        except Exception as e:
            print(f"Error opening {self.cache_file}: {e}")
        print(f"Loaded {len(self.cache)} cache entries")

    def encode(self, tensor: Tensor):
        assert(tensor.dtype == torch.float32)
        assert(tensor.shape[0] == self.embed_length)
        return self.config + ":" + tensor.float().numpy().tobytes().hex()

    def decode(self, encoded: str):
        prefix, hex = encoded.split(':')
        if prefix != self.config:
            raise ValueError(f"Expected OpenCLIP config '{self.config}' but got '{prefix}'")
        value = Tensor(numpy.frombuffer(bytearray.fromhex(hex), dtype=numpy.float32))
        assert(value.shape[0] == self.embed_length)
        return value

    def save_to_cache(self, filename, embedding):
        embedding = embedding.float()
        assert(embedding.dtype == torch.float32)
        self.cache[filename] = embedding
        try:
            with open(self.cache_file, 'a') as f:
                f.write(json.dumps({
                    "key": filename,
                    "value": self.encode(embedding)
                }) + "\n")
        except Exception as e:
            print(e)

    def precalc_embedding_batch(self, filenames, progress):
        filenames = [filename for filename in filenames if not str(filename) in self.cache]
        
        if len(filenames) == 0:
            return

        data = []
        for filename in progress.tqdm(filenames, desc="Loading", unit="files"):
            try:
                image = Image.open(filename)
                data.append({"filename": filename, "image": image})
            except Exception as e:
                print(e)
        
        if len(data) == 0:
            return

        for item in progress.tqdm(data, desc="Preprocessing", unit="files"):
            item['image'] = self.preprocess(item['image'])
        
        sd.unload_model_weights()

        dataloader = DataLoader(data, batch_size=self.batch_size)
        with torch.no_grad(), torch.cuda.amp.autocast():
            for batch in progress.tqdm(dataloader, desc="Calculating embeddings", unit="batches"):
                image_features = self.model.encode_image(batch['image'].to(device=self.device)).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)
                for (filename, embedding) in zip(batch['filename'], image_features.cpu()):
                    self.save_to_cache(filename, embedding)
                
        sd.load_model()

    def get_embedding(self, filename: str, image: Image = None):
        embedding = self.cache.get(str(filename), None)
        if embedding != None:
            assert(embedding.dtype == torch.float32)
            return embedding
            
        if image == None:
            image = Image.open(filename)

        image = self.preprocess(image).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image.to(device=self.device)).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

        embedding = image_features.flatten().cpu().float()
        assert(embedding.dtype == torch.float32)
        self.save_to_cache(filename, embedding)
        return embedding