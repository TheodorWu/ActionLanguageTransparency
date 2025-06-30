from model.vqa import VQA

class PureVLMActor(VQA):
    def __init__(self, num_action_tokens=0, backbone="blip", blip_task="vqa", freeze_llm=False, freeze_vision_encoder=False, img_source="image", **kwargs):
        super().__init__(backbone, blip_task, freeze_llm, freeze_vision_encoder, **kwargs)
        self.img_source = img_source
        self.num_action_tokens = num_action_tokens

    def resize_embeddings(self):
        if self.num_action_tokens > 0:
            self.vqa_model.resize_token_embeddings(self.config.vocab_size + 1 + self.num_action_tokens) # 1 for the state special token
            # Another option could be to also discretize the state and map it to tokens 

    # def forward(self, inputs):
    #     # input should look like this: prompt image <state-special-token> state 
    #     # target then is: description actionTokens 

    #     return super().forward(inputs)
    