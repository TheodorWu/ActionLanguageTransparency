import torch.nn as nn
import torch
from transformers import AutoTokenizer, T5EncoderModel, AutoImageProcessor, EfficientNetModel
from model.vqa import process_batch, decode_output

class VLMActor(nn.Module):
    def __init__(self, thought_generator, thought_processor, out_features, policy_dim, output_layer, policy_layers, img_source, prompt_template="Question: current goal is: <question>. immediate next step? Answer:", answer_template="Answer: <answer>" ,freeze_thought_generator=True, dtype=torch.float32, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.prompt_template = prompt_template
        self.answer_template = answer_template

        self.img_source = img_source

        self.thought_generator = thought_generator
        self.freeze_thought_generator = freeze_thought_generator
        if self.freeze_thought_generator:
            self.thought_generator.requires_grad_(False)
        self.thought_processor = thought_processor

        self.backbone = self.thought_generator.backbone

        layers = []
        for layer_key in policy_layers:
            if layer_key == "linear_projection":
                layer = nn.LazyLinear(policy_dim, dtype=dtype)
            elif layer_key == "transformer":
                layer = nn.TransformerEncoderLayer(policy_dim, 8, 2048, dtype=dtype)
            else:
                raise ValueError(f"Unkown output layer specified {layer_key}")
            layers.append(layer)
        self.policy_layers = nn.ModuleList(layers)

        if output_layer == "linear":
            self.single_token = True
            self.last_layer = nn.LazyLinear(out_features, dtype=dtype)
        elif output_layer == "lstm":
            self.single_token = False
            input_size = (policy_dim + 14)/16 # has to change or be computed when pooling kernel changes
            self.last_layer = nn.LSTMCell(input_size=input_size, hidden_size=out_features, dtype=dtype)
        else:
            raise ValueError(f"Unkown output layer specified {output_layer}")

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(16)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def print_trainable_parameters(self):
        print("If LazyModules are not yet initialized the total number of parameters changes after initialization.")
        trainable = sum(p.numel() for p in self.parameters() if (not isinstance(p, torch.nn.parameter.UninitializedParameter)) and p.requires_grad)
        total = sum(p.numel() for p in self.parameters() if  (not isinstance(p, torch.nn.parameter.UninitializedParameter)))
        print(f"trainable: {trainable:,} || all params: {total:,} || trainable%: {trainable/(total*100+1):.4f}")


    def format_text(self, instructions, answers):
        text = [ self.prompt_template.replace("<question>", t) for t in instructions ]
        if not answers is None:
            for i, _ in enumerate(answers):
                answers[i] = self.answer_template.replace("<question>", instructions[i]).replace("<answer>", answers[i])

        return text, answers

    def process_text_and_image(self, text, images, answers=None, generate=False):
        if answers is None:
            answers = text
        inputs = process_batch(self.thought_processor, images=images, text=text, answers=answers, training=not generate, device=self.device, dtype=self.dtype)
        return inputs

    def forward(self, **inputs):
        text, answers = self.format_text(inputs.get("instruction", [""]), inputs.get("answer", None))

        img = inputs.get(self.img_source)

        thought = self.process_text_and_image(text=text, images=img, answers=answers)

        if self.backbone == "blip":
            thought = self.thought_generator(**thought).last_hidden_state
            vlm_logits = None
        else:
            thought = self.thought_generator(output_hidden_states=True, **thought)
            vlm_logits = thought.logits
            thought = thought.language_model_outputs.encoder_last_hidden_state

        vlm_embedding = thought.clone().detach()

        for layer in self.policy_layers:
            thought = layer(thought)
            if not isinstance(layer, nn.TransformerEncoderLayer):
                thought = self.relu(thought)

        if self.single_token:
            thought = torch.flatten(thought[:,0], start_dim=1)

        thought = self.pool(thought)

        output = self.tanh(self.last_layer(thought))
        outputs = {
            "vlm_logits": vlm_logits,
            "vlm_embedding": vlm_embedding,
            "single_action": output
        }
        return outputs

    def generate(self, **inputs):
        text, _ = self.format_text(inputs.get("instruction", [""]), None)
        img = inputs.get(self.img_source)

        thought = self.process_text_and_image(text=text, images=img, generate=True)

        thought = self.thought_generator.generate(**thought)
        if isinstance(thought, dict):
            thought = thought["output"]
        thought = decode_output(self.thought_processor, thought)
        return thought



class EfficientNetThoughtActor(nn.Module):
    def __init__(self, d_model, thought_generator, thought_processor, out_features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.thought_generator = thought_generator
        self.thought_processor = thought_processor

        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.text_encoder = T5EncoderModel.from_pretrained("t5-small")
        self.text_encoder.requires_grad_(False)
        self.text_to_model_dim = nn.LazyLinear(d_model)

        self.state_to_model_dim = nn.LazyLinear(d_model)

        self.image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")
        self.image_encoder = EfficientNetModel.from_pretrained("google/efficientnet-b7")
        self.image_encoder.requires_grad_(False)
        self.img_to_model_dim = nn.LazyLinear(d_model)

        self.relu = nn.ReLU()
        self.last_layer = nn.LazyLinear(out_features)
        self.tanh = nn.Tanh()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def process_text_and_image(self, text, images):
        inputs = process_batch(self.thought_processor, images=images, text=text, answers=None, training=False, device=self.device, dtype=self.dtype)
        return inputs

    def forward(self, **inputs):
        text = inputs.get("instruction", "")
        img = inputs.get("image")

        thought = self.process_text_and_image(text=text, images=img)
        thought = self.thought_generator.generate(**thought, temperature=1, max_length=50)
        thought = self.relu(self.text_to_model_dim(thought))

        text = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=50).input_ids.to(self.device)
        text = self.text_encoder(text).last_hidden_state
        text = self.relu(self.text_to_model_dim(text))

        img = self.image_processor(img, return_tensors="pt", do_rescale=False, input_data_format="channels_first").pixel_values.to(self.device)
        img = self.image_encoder(img).pooler_output
        img = self.relu(self.img_to_model_dim(img))

        effectors = inputs.get("robot_state")
        effectors = self.relu(self.state_to_model_dim(effectors))

        combined = torch.cat([effectors, img, text, thought], dim=0)

        output = self.tanh(self.last_layer(combined))
        outputs = {
            "single_action": output
        }
        return outputs
