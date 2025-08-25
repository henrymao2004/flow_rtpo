# Based on https://github.com/RE-N-Y/imscore/blob/main/src/imscore/preference/model.py

from importlib import resources
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from transformers import AutoImageProcessor,CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image
import os

def get_size(size):
    if isinstance(size, int):
        return (size, size)
    elif "height" in size and "width" in size:
        return (size["height"], size["width"])
    elif "shortest_edge" in size:
        return size["shortest_edge"]
    else:
        raise ValueError(f"Invalid size: {size}")
    
def get_image_transform(processor:AutoImageProcessor):
    config = processor.to_dict()
    resize = T.Resize(get_size(config.get("size"))) if config.get("do_resize") else nn.Identity()
    crop = T.CenterCrop(get_size(config.get("crop_size"))) if config.get("do_center_crop") else nn.Identity()
    normalise = T.Normalize(mean=processor.image_mean, std=processor.image_std) if config.get("do_normalize") else nn.Identity()

    return T.Compose([resize, crop, normalise])

class ClipScorer(torch.nn.Module):
    def __init__(self, device, use_local=False, local_base_path="/mnt/data/group/zhaoliangjie/ICLR-work/", 
                 local_model_name="clip-vit-large-patch14", hf_model_name="openai/clip-vit-large-patch14"):
        super().__init__()
        self.device = device
        self.use_local = use_local
        self.local_base_path = local_base_path
        self.local_model_name = local_model_name
        self.hf_model_name = hf_model_name
        
        # Load CLIP model and processor
        self.model, self.processor = self._load_clip_model()
        self.tform = get_image_transform(self.processor.image_processor)
        self.eval()
    
    def _load_clip_model(self):
        """Load CLIP model from either local path or HuggingFace."""
        if self.use_local:
            return self._load_clip_from_local()
        else:
            return self._load_clip_from_huggingface()
    
    def _load_clip_from_local(self):
        """Load CLIP model from local path."""
        try:
            local_model_path = os.path.join(self.local_base_path, self.local_model_name)
            print(f"[CLIP LOCAL] Loading CLIP model from: {local_model_path}")
            
            if not os.path.exists(local_model_path):
                print(f"[CLIP LOCAL] Local model path does not exist: {local_model_path}")
                print("[CLIP LOCAL] Falling back to HuggingFace loading...")
                return self._load_clip_from_huggingface()
            
            # Load model and processor from local path
            model = CLIPModel.from_pretrained(local_model_path).to(self.device)
            processor = CLIPProcessor.from_pretrained(local_model_path)
            
            print(f"[CLIP LOCAL] Successfully loaded CLIP model from local path")
            return model, processor
            
        except Exception as e:
            print(f"[CLIP LOCAL] Failed to load from local path: {e}")
            print("[CLIP LOCAL] Falling back to HuggingFace loading...")
            return self._load_clip_from_huggingface()
    
    def _load_clip_from_huggingface(self):
        """Load CLIP model from HuggingFace."""
        try:
            print(f"[CLIP HF] Loading CLIP model from HuggingFace: {self.hf_model_name}")
            model = CLIPModel.from_pretrained(self.hf_model_name).to(self.device)
            processor = CLIPProcessor.from_pretrained(self.hf_model_name)
            print(f"[CLIP HF] Successfully loaded CLIP model from HuggingFace")
            return model, processor
        except Exception as e:
            print(f"[CLIP HF] Failed to load from HuggingFace: {e}")
            raise e
    
    def _process(self, pixels):
        dtype = pixels.dtype
        pixels = self.tform(pixels)
        pixels = pixels.to(dtype=dtype)

        return pixels

    @torch.no_grad()
    def __call__(self, pixels, prompts, return_img_embedding=False):
        texts = self.processor(text=prompts, padding='max_length', truncation=True, return_tensors="pt").to(self.device)
        pixels = self._process(pixels).to(self.device)
        outputs = self.model(pixel_values=pixels, **texts)
        if return_img_embedding:
            return outputs.logits_per_image.diagonal()/30, outputs.image_embeds
        return outputs.logits_per_image.diagonal()/30

    @torch.no_grad()
    def image_similarity(self, pixels, ref_pixels):
        pixels = self._process(pixels).to(self.device)
        ref_pixels = self._process(ref_pixels).to(self.device)

        pixel_embeds = self.model.get_image_features(pixel_values=pixels)
        ref_embeds = self.model.get_image_features(pixel_values=ref_pixels)

        pixel_embeds = pixel_embeds / pixel_embeds.norm(p=2, dim=-1, keepdim=True)
        ref_embeds = ref_embeds / ref_embeds.norm(p=2, dim=-1, keepdim=True)

        sim = pixel_embeds @ ref_embeds.T
        sim = torch.diagonal(sim, 0)
        return sim


def main():
    scorer = ClipScorer(
        device='cuda'
    )

    images=[
    "assets/test.jpg",
    "assets/test.jpg"
    ]
    pil_images = [Image.open(img) for img in images]
    prompts=[
        'an image of cat',
        'not an image of cat'
    ]
    images = [np.array(img) for img in pil_images]
    images = np.array(images)
    images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    images = torch.tensor(images, dtype=torch.uint8)/255.0
    print(scorer(images, prompts))

if __name__ == "__main__":
    main()