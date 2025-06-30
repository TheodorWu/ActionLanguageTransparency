import torch
import bitsandbytes as bnb

def get_optimizer(id, model_params, optimizer_params):
    if id.lower()=="adam":
        return torch.optim.Adam(model_params, **optimizer_params)
    if id.lower()=="adam_8bit":
        return bnb.optim.Adam8bit(model_params, **optimizer_params)
    elif id.lower()=="adamw":
        return torch.optim.AdamW(model_params, **optimizer_params)
    else:
        raise NameError(f"Unknown optimizer specified: {id}")