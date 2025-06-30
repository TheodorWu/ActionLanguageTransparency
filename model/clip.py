from transformers import CLIPModel, CLIPProcessor, BitsAndBytesConfig
import torch
from einops import rearrange, reduce
from os.path import dirname, abspath

class ClipSimilarityScore():
    def __init__(self, clip_model, clip_processor, dtype) -> None:
        self.model = clip_model
        self.model.train(False)
        self.processor = clip_processor
        self.dtype = dtype

    def __call__(self, text, images):
        if images.ndim == 5:
            b, n, _, _, _ = images.shape
            images = rearrange(images, "b n c w h -> (b n) c w h")
            inputs = self.processor(images=images, text=text, return_tensors="pt", padding=True).to(self.dtype)
            inputs["pixel_values"] = rearrange(inputs["pixel_values"], "(b n) c w h -> b n c w h", b=b, n=n)
            inputs["pixel_values"] = reduce(inputs["pixel_values"], "b n c w h -> b c w h", "mean") # could actually concatenate along c which would result in b c (wn) (hn)
        else:
            inputs =  self.processor(text=text, images=images, return_tensors="pt", padding=True).to(self.dtype)     
    
        outputs = self.model(**inputs).logits_per_image # scaled to range 0-100
        outputs = outputs*torch.eye(outputs.shape[0])
        return torch.sum(outputs)/outputs.shape[0]

def get_clip(cp=None):
    dtype = torch.float32
    bnb_conf = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=dtype)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",quantization_config=bnb_conf, torch_dtype=dtype)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if not cp is None:
        parent_dir = dirname(dirname(abspath(__file__)))
        checkpoint = torch.load(f"{parent_dir}/checkpoints/{cp}")
        model.load_state_dict(checkpoint)

    return model, processor, dtype

def get_clip_similarity_score(cp=None) -> ClipSimilarityScore:
    model, processor, dtype = get_clip(cp=cp)
    return ClipSimilarityScore(clip_model=model, clip_processor=processor, dtype=dtype)
