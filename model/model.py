from model.vqa import VQA
from model.vqa import load_blip_processor, load_generic_processor
from model.cot import COT

import torch.nn as nn
import torch

from model.actor import TestActor, TestActorMultimodal, TransformerActor, EfficientNetTransformerActor, EfficientNetActor
from model.think_actor import VLMActor, EfficientNetThoughtActor
from model.pure_vlm_actor import PureVLMActor

from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType

from os.path import dirname, abspath


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.Tensor(0)

def load_checkpoint(checkpoint, model):

    parent_dir = dirname(dirname(abspath(__file__)))
    checkpoint = torch.load(f"{parent_dir}/checkpoints/{checkpoint}")
    model.load_state_dict(checkpoint)

    return model

def init_model(id, peft_args=None, **kwargs):
    checkpoint = kwargs.pop("checkpoint", None)
    
    if id == "dummy":
        model = DummyModel()
    elif id == "vqa":
        model = VQA(**kwargs)
    elif id == "cot":
        model = COT(**kwargs)
    else:
        raise NameError(f"Unknown model specified: {id}")

    if checkpoint:
        model = load_checkpoint(checkpoint, model)

    if peft_args and peft_args.get("active", False):
        peft_config = get_peft_config(peft_args.args)
        module_names = [n for n, _ in model.named_modules()]
        peft_config.target_modules = get_module_names(module_names, peft_config.target_modules)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        
    model.print_trainable_parameters()

    return model


def init_agent(id, peft_args=None, **kwargs):
    if id == "test":
        agent = TestActor(**kwargs)
    elif id == "transformer":
        agent = TransformerActor(**kwargs)
    elif id == "efficient_net":
        agent = EfficientNetActor(**kwargs)
    elif id == "purevlm":
        checkpoint = kwargs.pop("checkpoint", None)
        agent = PureVLMActor(**kwargs)
        if checkpoint and "agent.pt" in checkpoint:
            agent.resize_embeddings()
            agent = load_checkpoint(checkpoint, agent)
        elif checkpoint:
            agent = load_checkpoint(checkpoint, agent)
            agent.resize_embeddings()
        else:
            agent.resize_embeddings()
    elif id == "vlm":
        generator_id = kwargs.pop("thought_generator")
        generator_args = kwargs.pop("generator_args")
        generator = init_model(generator_id, **generator_args.kwargs)
        processor, dtype = load_processor(generator_args.kwargs.get("backbone"), **generator_args.kwargs)
        agent = VLMActor(thought_generator=generator, thought_processor=processor, dtype=dtype, **kwargs)
    elif id == "efficient_thought":
        agent = EfficientNetThoughtActor(**kwargs)
    elif id == "efficient_transformer":
        agent = EfficientNetTransformerActor(**kwargs)
    elif id == "test_multimodal":
        agent = TestActorMultimodal(**kwargs)

    if peft_args and peft_args.get("active", False):
        peft_config = get_peft_config(peft_args.args)
        module_names = [n for n, _ in agent.named_modules()]
        peft_config.target_modules = get_module_names(module_names, peft_config.target_modules)
        agent = prepare_model_for_kbit_training(agent)
        agent = get_peft_model(agent, peft_config)

    if hasattr(agent, "print_trainable_parameters") and callable(getattr(agent, "print_trainable_parameters")):
        agent.print_trainable_parameters()

    return agent


def load_processor(id, **kwargs):
    if id == "blip":
        return load_blip_processor(og_task=kwargs.get("blip_task", "vqa"))
    else:
        return load_generic_processor(name=id)

def get_peft_config(peft_args):
    return LoraConfig(**peft_args, task_type=TaskType.QUESTION_ANS)

def get_module_names(named_modules, patterns):
    names = []
    for n in named_modules:
        for pattern in patterns:
            if pattern in n:
                names.append(n)
                break
    return names