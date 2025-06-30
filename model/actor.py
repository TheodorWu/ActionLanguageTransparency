import torch.nn as nn
import torch
from transformers import AutoTokenizer, T5EncoderModel, AutoImageProcessor, ViTModel, EfficientNetModel
from model.positional_encoding import PositionalEncoding
from einops import rearrange

class TestActor(nn.Module):
    def __init__(self, num_cells, out_features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.linear1 = nn.LazyLinear(num_cells)
        self.linear2 = nn.LazyLinear(num_cells)
        self.linear3 = nn.LazyLinear(num_cells)
        self.activation = nn.Tanh()
        self.last_layer = nn.LazyLinear(out_features)

    def forward(self, **inputs):
        x = self.flatten_dict_observation(**inputs)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.last_layer(x)
        return x
    
    def flatten_dict_observation(self, **inputs):
        flattened = [torch.flatten(v, start_dim=1) for v in inputs.values() if torch.is_tensor(v)]
        return torch.cat(flattened, 1)
    
class TestActorMultimodal(TestActor):
    def __init__(self, num_cells, out_features, *args, **kwargs) -> None:
        super().__init__(num_cells, out_features, *args, **kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.text_encoder = T5EncoderModel.from_pretrained("t5-small")

    @property
    def device(self):
        return next(self.parameters()).device

    def flatten_dict_observation(self, **inputs):
        text = inputs.get("instruction", None)
        if not text is None:
            text = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=50).input_ids.to(self.device)
            text = self.text_encoder(text).last_hidden_state
            text = torch.flatten(text, start_dim=1)

        flattened = [torch.flatten(v, start_dim=1) for v in inputs.values() if torch.is_tensor(v)]
        flattened = torch.cat(flattened, 1)
        flattened = torch.cat([text, flattened], 1)

        return flattened

class EfficientNetActor(nn.Module):
    def __init__(self, d_text, d_img, d_state, out_features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.text_encoder = T5EncoderModel.from_pretrained("t5-small")
        self.text_encoder.requires_grad_(False)
        self.text_to_model_dim = nn.LazyLinear(d_text)

        self.image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")
        self.image_encoder = EfficientNetModel.from_pretrained("google/efficientnet-b7")
        self.image_encoder.requires_grad_(False)
        self.img_to_model_dim = nn.LazyLinear(d_img)
        
        self.state_to_model_dim = nn.LazyLinear(d_state)

        self.relu = nn.ReLU()
        self.last_layer = nn.LazyLinear(out_features)
        self.tanh = nn.Tanh()
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, **inputs):
        text = inputs.get("instruction", "")
        text = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=50).input_ids.to(self.device)
        text = self.text_encoder(text).last_hidden_state
        text = torch.flatten(text)
        text = self.relu(self.text_to_model_dim(text))

        img = inputs.get("secondary_image")
        img = self.image_processor(img, return_tensors="pt", do_rescale=False).pixel_values.to(self.device)
        img = self.image_encoder(img).last_hidden_state
        img = self.relu(self.img_to_model_dim(img))

        effectors = inputs.get("robot_state")
        effectors = self.relu(self.state_to_model_dim(effectors))
        effectors = rearrange(effectors, "b d -> b 1 d")

        combined = torch.cat([effectors, img, text], dim=1)
        combined = torch.flatten(combined)

        output = self.tanh(self.last_layer(combined))
        outputs = {
            "single_action": output
        }
        return outputs


class TransformerActor(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, transformer_layers, out_features, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.text_encoder = T5EncoderModel.from_pretrained("t5-small")
        self.text_encoder.requires_grad_(False)
        self.text_to_model_dim = nn.LazyLinear(d_model)

        self.state_to_model_dim = nn.LazyLinear(d_model)

        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.image_encoder.requires_grad_(False)
        self.img_to_model_dim = nn.LazyLinear(d_model)

        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=5000)

        self.relu = nn.ReLU()
        self.last_layer = nn.LazyLinear(out_features)

        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                               nhead=nhead, 
                                                               dim_feedforward=dim_feedforward,
                                                               dropout=dropout, 
                                                               batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_layer,
                                                         num_layers=transformer_layers)
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, **inputs):
        text = inputs.get("instruction", "")
        text = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=50).input_ids.to(self.device)
        text = self.text_encoder(text).last_hidden_state
        text = self.relu(self.text_to_model_dim(text))

        img = inputs.get("secondary_image")
        img = self.image_processor(img, return_tensors="pt", do_rescale=False).pixel_values.to(self.device)
        img = self.image_encoder(img).last_hidden_state
        img = self.relu(self.img_to_model_dim(img))

        effectors = inputs.get("robot_state")
        effectors = self.relu(self.state_to_model_dim(effectors))
        effectors = rearrange(effectors, "b d -> b 1 d")

        combined = torch.cat([effectors, img, text], dim=1)
        combined = self.pe(combined)
        combined = self.transformer(combined)

        out_sequence = self.last_layer(combined)
        outputs = {
            "sequence": out_sequence,
            "single_action": out_sequence[:, 0]
        }
        return outputs
    
class EfficientNetTransformerActor(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, transformer_layers, out_features, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.text_encoder = T5EncoderModel.from_pretrained("t5-small")
        self.text_encoder.requires_grad_(False)
        self.text_to_model_dim = nn.LazyLinear(d_model)

        self.state_to_model_dim = nn.LazyLinear(d_model)

        self.image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")
        self.image_encoder = EfficientNetModel.from_pretrained("google/efficientnet-b7")
        self.image_encoder.requires_grad_(False)
        self.img_to_model_dim = nn.LazyLinear(d_model)

        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=5000)

        self.relu = nn.ReLU()
        self.last_layer = nn.LazyLinear(out_features)

        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                               nhead=nhead, 
                                                               dim_feedforward=dim_feedforward,
                                                               dropout=dropout, 
                                                               batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_layer,
                                                         num_layers=transformer_layers)
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, **inputs):
        text = inputs.get("instruction", "")
        text = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=50).input_ids.to(self.device)
        text = self.text_encoder(text).last_hidden_state
        text = self.relu(self.text_to_model_dim(text))

        img = inputs.get("image")
        img = self.image_processor(img, return_tensors="pt", do_rescale=False, input_data_format="channels_first").pixel_values.to(self.device)
        img = self.image_encoder(img).pooler_output
        img = rearrange(img, "b (n d) -> b n d", n=32) # slice pooled output features
        img = self.relu(self.img_to_model_dim(img))

        effectors = inputs.get("robot_state")
        effectors = self.relu(self.state_to_model_dim(effectors))
        effectors = rearrange(effectors, "b d -> b 1 d")

        combined = torch.cat([effectors, img, text], dim=1)
        combined = self.pe(combined)
        combined = self.transformer(combined)

        out_sequence = self.last_layer(combined)
        outputs = {
            "sequence": out_sequence,
            "single_action": out_sequence[:, 0]
        }
        return outputs