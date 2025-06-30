import torch
from model.vqa import VQA

class COT(VQA):
    def __init__(self, thought_pattern, thought_attention_mask, backbone="blip", blip_task="vqa", **kwargs):
        super().__init__(backbone, blip_task, **kwargs)

        self.thought_pattern = thought_pattern # provide the already embedded thought pattern to the model upon initialization. Appended to first CoT step.
        self.thought_attention_mask = thought_attention_mask

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, pixel_values=None, attention_mask=None, labels=None, decoder_attention_mask=None):
        b = input_ids.shape[0]
        pattern_tensor = self.thought_pattern.repeat(b, 1)
        think = torch.cat([pattern_tensor, input_ids], 1)

        if not attention_mask is None:
            thought_attention_mask = self.thought_attention_mask.repeat(b, 1)
            thought_attention_mask = torch.cat([thought_attention_mask, attention_mask], 1)

        thought = self.vqa_model.generate(input_ids=think, pixel_values=pixel_values, attention_mask=thought_attention_mask, max_new_tokens=30)

        # Extend input by previous thought as context
        input_ids = torch.cat([thought, input_ids], 1)
        attention_mask = torch.cat([torch.ones(thought.shape, dtype=attention_mask.dtype, device=self.thought_pattern.device), attention_mask], 1)
        labels = torch.cat([labels, torch.zeros(thought.shape, dtype=labels.dtype, device=self.thought_pattern.device)], 1)

        outputs = self.vqa_model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels, decoder_attention_mask=decoder_attention_mask)
        return outputs

    def generate(self, input_ids, pixel_values=None, attention_mask=None, **kwargs):
        b = input_ids.shape[0]
        pattern_tensor = torch.repeat(self.thought_pattern, b, axis=0)
        think = torch.cat([pattern_tensor, input_ids], 1)

        if not attention_mask is None:
            thought_attention_mask = torch.repeat(self.thought_attention_mask, b, axis=0)
            thought_attention_mask = torch.cat([thought_attention_mask, attention_mask], 1)

        thought = self.vqa_model.generate(input_ids=think, pixel_values=pixel_values, attention_mask=thought_attention_mask, **kwargs)

        input_ids = torch.cat([input_ids, thought], 1)
        attention_mask = torch.cat([torch.ones(thought.shape, dtype=attention_mask.dtype, device=self.thought_pattern.device), attention_mask], 1)
        outputs = self.vqa_model.generate(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **kwargs)

        return { "outputs": outputs, "thought": thought }
