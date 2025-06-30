import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch
import hydra
import wandb

from model.model import init_model, load_processor, init_agent
from model.clip import get_clip
from clip_train import finetune_clip
from data import load_roboVQA
from optimizer import get_optimizer
from vqa import finetune_vlm
from environment import init_env_for_imitation
from imitation_learning import behavior_cloning
from imitation_learning_vlm import behavior_cloning_vlm_only
from loss import get_loss_fn, LanguagePreservationLoss
from utils import Discretizer, seed_all
DIR_PATH = Path(__file__).parent.resolve()

def train_clip(cfg, checkpoint_dir, device):
    vqa_dataset_train = load_roboVQA(mode="train", folder = cfg.training.clip.data_dir)
    vqa_dataset_train = vqa_dataset_train.batch(batch_size=cfg.training.clip.batch_size).shuffle(
        buffer_size=cfg.training.clip.shuffle_buffer_size)
    
    vqa_dataset_val = load_roboVQA(mode="val", folder = cfg.training.clip.data_dir)
    vqa_dataset_val = vqa_dataset_val.batch(batch_size=cfg.training.clip.batch_size)

    clip_model, clip_processor, dtype = get_clip()  # not using "to" since bitsandbytes handles the correct device for me
    if hasattr(clip_processor, "image_processor"):
        clip_processor.image_processor.do_rescale = False
    clip_optimizer = get_optimizer(id=cfg.training.optimizer.name,
                                    model_params=clip_model.parameters(),
                                optimizer_params=cfg.training.optimizer.optimizer_params)
    
    clip_model = finetune_clip(model=clip_model,
                    processor=clip_processor,
                    train_dataset=vqa_dataset_train,
                    val_dataset=vqa_dataset_val,
                    epochs=cfg.training.clip.epochs,
                    optimizer=clip_optimizer,
                    log_frequency=cfg.wandb.log_frequency,
                    single_image=cfg.model.single_image,
                    use_context=cfg.training.vqa.use_context,
                    device=device,
                    dtype=dtype
                    )
    torch.save(clip_model.state_dict(), f"{checkpoint_dir}/{wandb.run.name}-clip.pt")
    return clip_model

def train_vqa(cfg, checkpoint_dir, device):
    vqa_dataset_train = load_roboVQA(mode="train", folder = cfg.training.vqa.data_dir)
    vqa_dataset_train = vqa_dataset_train.batch(batch_size=cfg.training.vqa.batch_size).shuffle(
        buffer_size=cfg.training.vqa.shuffle_buffer_size)
    
    vqa_dataset_val = load_roboVQA(mode="val", folder = cfg.training.vqa.data_dir)
    vqa_dataset_val = vqa_dataset_val.batch(batch_size=cfg.training.vqa.batch_size)
    
    processor, dtype = load_processor(id=cfg.model.kwargs.get("backbone"), **cfg.model.kwargs)

    model_args = dict(cfg.model.kwargs)
    thought_pattern = model_args.get("thought_pattern", None)
    if thought_pattern:
        processed_thought = processor(text=thought_pattern, return_tensors="pt").to(device)
        model_args["thought_pattern"] = processed_thought["input_ids"]
        model_args["thought_attention_mask"] = processed_thought["attention_mask"]

    model = init_model(id=cfg.model.id, peft_args=cfg.training.get("peft", None), **model_args).to(device)
    torch.save(model.state_dict(), f"{checkpoint_dir}/test-vqa.pt")
    vqa_optimizer = get_optimizer(id=cfg.training.optimizer.name,
                                    model_params=model.parameters(),
                                optimizer_params=cfg.training.optimizer.optimizer_params)
                
    model = finetune_vlm(model=model,
                    processor=processor,
                    train_dataset=vqa_dataset_train,
                    val_dataset=vqa_dataset_val,
                    optimizer=vqa_optimizer,
                    device=device, 
                    dtype=dtype,
                    cfg=cfg)
    torch.save(model.state_dict(), f"{checkpoint_dir}/{wandb.run.name}-vqa.pt")
    return model

def train_bc(cfg, checkpoint_dir, device):
    loss_fn = get_loss_fn(cfg.training.bc.loss_function, **cfg.training.bc.get("loss_function_kwargs", {}))

    agent_kwargs = dict(cfg.model.agent.kwargs)
    agent_kwargs["img_source"] = cfg.environment.get("img_source", "img")
    agent_model = init_agent(cfg.model.agent.id, peft_args=cfg.training.get("peft", None), **agent_kwargs).to(device)

    agent_model_copy = None
    if isinstance(loss_fn, LanguagePreservationLoss):
        agent_model_copy = init_agent(cfg.model.agent.id, **agent_kwargs).to(device)

    optimizer = get_optimizer(id=cfg.training.optimizer.name,
                                model_params=agent_model.parameters(),
                                optimizer_params=cfg.training.optimizer.optimizer_params)

    train_env, episode_iterator_func = init_env_for_imitation(dataset_id=cfg.environment.id, mode="train", **cfg.environment.kwargs)
    val_env, episode_iterator_func = init_env_for_imitation(dataset_id=cfg.environment.id, mode="validation", **cfg.environment.kwargs)
    test_env, episode_iterator_func = init_env_for_imitation(dataset_id=cfg.environment.id, mode="test", **cfg.environment.kwargs)

    action_discretizer = None
    if cfg.environment.get("action_discretizer", False):
        action_discretizer = Discretizer(**cfg.environment.action_discretizer)

    state_discretizer = None
    if cfg.environment.get("state_discretizer", False):
        state_discretizer = Discretizer(**cfg.environment.state_discretizer)
    
    if not (cfg.environment.id in ["lhmanip", "language_table"]):
        train_env = train_env.batch(batch_size=cfg.training.bc.batch_size, drop_last=True).shuffle(buffer_size=cfg.training.bc.shuffle_buffer_size)
        val_env = val_env.batch(batch_size=cfg.training.bc.batch_size, drop_last=True)

    if cfg.model.agent.id == "purevlm":
        if action_discretizer is None:
            raise TypeError("Make sure you initialize and pass an action_discretizer to start BC VLM only.")
        
        processor, _ = load_processor(id=cfg.model.agent.kwargs.get("backbone"), **cfg.model.kwargs)
        new_tokens = [f"a{i}" for i in range(action_discretizer.num_bins)]
        if state_discretizer:
            new_tokens.extend([f"s{i}" for i in range(state_discretizer.num_bins)])
        new_tokens.append("[state]")
        new_tokens.append("[action]")
        processor.tokenizer.add_tokens(new_tokens)
        
        agent_model = behavior_cloning_vlm_only(model=agent_model,
                                processor=processor,
                                optimizer=optimizer,
                                train_dataset=train_env,
                                val_dataset=val_env,
                                test_dataset=test_env,
                                episode_iterator_func=episode_iterator_func,
                                device=device,
                                cfg=cfg,
                                action_discretizer=action_discretizer,
                                state_discretizer=state_discretizer)
    else:
        agent_model = behavior_cloning(model=agent_model,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                train_dataset=train_env,
                                val_dataset=val_env,
                                test_dataset=test_env,
                                episode_iterator_func=episode_iterator_func,
                                action_discretizer=action_discretizer,
                                state_discretizer=state_discretizer,
                                device=device,
                                cfg=cfg,
                                model_copy=agent_model_copy
                                )
    
    torch.save(agent_model.state_dict(), f"{checkpoint_dir}/{wandb.run.name}-agent.pt")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed_all(42)
    print(f"Using torch {torch.__version__}", file=sys.stdout)

    print(f"Cuda available: {torch.cuda.is_available()}", file=sys.stdout)
    print(f"GPU Config Active: {cfg.training.device.gpu}")
    if cfg.training.device.gpu and torch.cuda.is_available():
        print('__CUDNN VERSION:', torch.backends.cudnn.version(), file=sys.stdout)
        print('Available devices ', torch.cuda.device_count(), file=sys.stdout)
        print('Current cuda device ', torch.cuda.current_device(), file=sys.stdout)
    
        device = torch.device("cuda")
    else:
        print("Using CPU", file=sys.stdout)
        device = torch.device("cpu")

    print(OmegaConf.to_yaml(cfg), file=sys.stdout)

    # setup wandb
    wandb.init(entity=cfg.wandb.entity,
               project=cfg.wandb.project, config=OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    ))
    print(f"Starting run: {wandb.run.name}")

    checkpoint_dir = f"{DIR_PATH}/checkpoints"
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)

    ### Stage 0.5:
    ### Finetune CLIP
    if cfg.training.get("clip", False) and cfg.training.clip.active:
        train_clip(cfg=cfg, checkpoint_dir=checkpoint_dir, device=device)

    ### Stage 1:
    ### Finetune VLM
    if cfg.training.get("vqa", False) and cfg.training.vqa.active:
        train_vqa(cfg=cfg, checkpoint_dir=checkpoint_dir, device=device)

    ### Stage 2:
    ### Imitation Learning
    if cfg.training.get("bc", False) and cfg.training.bc.active:
        train_bc(cfg=cfg, checkpoint_dir=checkpoint_dir, device=device)
    # evaluation
    wandb.finish()


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
