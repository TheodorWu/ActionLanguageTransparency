import inspect
import torch
import torch.nn as nn

from transformers.feature_extraction_utils import BatchFeature
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration, LlavaForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration, VideoLlavaProcessor, VideoLlavaForConditionalGeneration, AutoProcessor, PaliGemmaForConditionalGeneration
from einops import rearrange, reduce

def load_blip_processor(og_task="vqa"):
    if og_task == "vqa":
        processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    elif og_task == "image_captioning":
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    else:
        raise NameError(f"Unknown BLIP processor specified: {og_task}")
    dtype = torch.float32
    processor.image_processor.do_rescale = False
    return processor, dtype
    
def load_blip_model(og_task="vqa"):
    # bnb_conf = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float32)
    if og_task == "vqa":
        return BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")# , quantization_config=bnb_conf, torch_dtype=torch.float32)
    elif og_task == "image_captioning":
        return BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base") #, quantization_config=bnb_conf, torch_dtype=torch.float16)
    else:
        raise NameError(f"Unknown BLIP model specified: {og_task}")

def load_generic_processor(name):
    dtype = torch.float32
    if name=="llava":
        processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.5-7b-hf")
    if name=="blip2":
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        processor.image_processor.do_rescale = False # pylint: disable=no-member
        # dtype = torch.float16
    if name=="git":
        processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
        processor.image_processor.do_rescale = False
        processor.tokenizer.padding_side = "left"
    if name=="instructblip":
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        processor.image_processor.do_rescale = False # pylint: disable=no-member
        dtype = torch.float16
    if name=="instructblip_flan":
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        processor.image_processor.do_rescale = False # pylint: disable=no-member
        dtype = torch.float16
    if name=="video_llava":
        processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
        processor.image_processor.do_rescale = False # pylint: disable=no-member
        processor.tokenizer.padding_side = "left"
        dtype = torch.bfloat16
    if name=="pali_gemma":
        processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
        processor.image_processor.do_rescale = False
    if name=="pali_gemma_bfloat16":
        processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
        processor.image_processor.do_rescale = False
        dtype = torch.bfloat16
    return processor, dtype
    
def load_generic_model(name):
    if name=="llava":
        model =  LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    if name=="blip2":
        bnb_conf = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float32)
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", quantization_config=bnb_conf, torch_dtype=torch.float32)
    if name=="git":
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")
    if name=="instructblip":
        bnb_conf = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b" , quantization_config=bnb_conf, torch_dtype=torch.float16)
    if name=="instructblip_flan":
        bnb_conf = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl" , quantization_config=bnb_conf, torch_dtype=torch.float16)
    if name=="video_llava":
        bnb_conf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", quantization_config=bnb_conf, torch_dtype=torch.bfloat16)
    if name=="pali_gemma":
        # bnb_config = None
        model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-224") #, quantization_config=bnb_config)
    if name=="pali_gemma_bfloat16":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-224", quantization_config=quantization_config)
    return model

def process_batch(vqa_processor, images, text, answers, training, device, dtype=torch.float32):

    if isinstance(vqa_processor, VideoLlavaProcessor):
        b = images.shape[0]
        # images = rearrange(images, "b n c h w -> b n h w c")
        inputs = {
            "pixel_values_videos": []
        }
        for i in range(b):
            inputs_tmp = vqa_processor(videos=images[i], text=text[i], return_tensors="pt", padding=True).to(dtype)
            inputs["pixel_values_videos"].append(inputs_tmp["pixel_values_videos"])

        inputs["pixel_values_videos"] = torch.cat(inputs["pixel_values_videos"])
        inputs_tmp = vqa_processor(text=text, return_tensors="pt", padding=True)
        inputs["attention_mask"], inputs["input_ids"] = inputs_tmp["attention_mask"], inputs_tmp["input_ids"]
        inputs = BatchFeature(inputs)

    elif images.ndim == 5:
        b, n, _, _, _ = images.shape
        images = rearrange(images, "b n c w h -> (b n) c w h")
        if "suffix" in inspect.signature(vqa_processor).parameters.keys():
            inputs = vqa_processor(images=images, text=text,  suffix=answers, return_tensors="pt", padding=True).to(dtype)   
            return inputs.to(device)                    
        else:
            inputs = vqa_processor(images=images, text=text, return_tensors="pt", padding=True).to(dtype)
        inputs["pixel_values"] = rearrange(inputs["pixel_values"], "(b n) c w h -> b n c w h", b=b, n=n)
        inputs["pixel_values"] = reduce(inputs["pixel_values"], "b n c w h -> b c w h", "mean") # could actually concatenate along c which would result in b c (wn) (hn)

    else:
        if "suffix" in inspect.signature(vqa_processor).parameters.keys():
            inputs = vqa_processor(images=images, text=text,  suffix=answers, return_tensors="pt", padding=True).to(dtype)
            return inputs.to(device)           
        else:
            inputs = vqa_processor(images=images, text=text, return_tensors="pt", padding=True).to(dtype)
    
    if training:
        text_and_answers = text + answers
        text_and_answers_inputs = vqa_processor(text=text_and_answers, images=images, return_tensors="pt", padding=True).to(dtype)
        inputs["input_ids"], inputs["labels"] = torch.split(text_and_answers_inputs["input_ids"], split_size_or_sections =len(text_and_answers_inputs["input_ids"])//2, dim=0)
        inputs["attention_mask"], inputs["decoder_attention_mask"] = torch.split(text_and_answers_inputs["attention_mask"], split_size_or_sections=len(text_and_answers_inputs["attention_mask"])//2)

    return inputs.to(device)

def decode_output(vqa_processor, raw_outputs):
    decoded_output = [vqa_processor.decode(raw_output, skip_special_tokens=True) for raw_output in raw_outputs]
    return decoded_output

class VQA(nn.Module):
    def __init__(self, backbone="blip", blip_task="vqa", freeze_llm=False, freeze_vision_encoder=False, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone

        self.load_backbone(backbone, blip_task=blip_task)
        self.config = self.vqa_model.config
        
        if freeze_llm:
            for param in self.vqa_model.language_model.parameters():
                param.requires_grad = False

        if freeze_vision_encoder:
            if self.backbone == "video_llava":
                for param in self.vqa_model.multi_modal_projector.parameters():
                    param.requires_grad = False
                for param in self.vqa_model.image_tower.parameters():
                    param.requires_grad = False
                for param in self.vqa_model.video_tower.parameters():
                    param.requires_grad = False
            elif self.backbone in ["pali_gemma", "pali_gemma_bfloat16"]:
                for param in self.vqa_model.vision_tower.parameters():
                    param.requires_grad = False
            else:
                for param in self.vqa_model.vision_model.parameters():
                    param.requires_grad = False

    def forward(self, **inputs):
        if self.backbone in ["video_llava"]:
            inputs.pop("decoder_attention_mask")

        outputs = self.vqa_model(**inputs)
        # if self.backbone == "git":
        #     outputs = self.vqa_model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], attention_mask=inputs.get("attention_mask", None), labels=inputs.get("labels", None), **kwargs)
        # elif self.backbone == "video_llava":
        #     # outputs = self.vqa_model(**inputs, **kwargs)   
        # else:
        #     # outputs = self.vqa_model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels, decoder_attention_mask=decoder_attention_mask, **kwargs)
        #     outputs = self.vqa_model(**inputs, **kwargs)   
        return outputs
    
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable: {trainable:,} || all params: {total:,} || trainable%: {trainable/total*100:.4f}")
    
    def generate(self, **inputs):
        
        outputs = self.vqa_model.generate(**inputs)    
        return outputs
    
    def load_backbone(self, backbone, **kwargs):
        if backbone == "blip":
            self.vqa_model = load_blip_model(og_task=kwargs.get("blip_task", "vqa"))
        else:
            self.vqa_model = load_generic_model(name=backbone)

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype