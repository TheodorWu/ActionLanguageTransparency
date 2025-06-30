import copy
import torch
from log_metrics import log_metrics, log_action_obs_pair, log_vqa_pair
from utils import batch_of_dicts_to_outer_dict, DotDict
from loss import LanguagePreservationLoss
from model.vqa import process_batch, decode_output

from model.clip import get_clip_similarity_score

def behavior_cloning(model,
                     optimizer,
                     loss_fn,
                     train_dataset,
                     val_dataset,
                     test_dataset,
                     episode_iterator_func, 
                     cfg,
                     device,
                     action_discretizer=None,
                     state_discretizer=None,
                     model_copy=None
                     ):
    
    epochs=cfg.training.bc.epochs
    log_frequency=cfg.wandb.log_frequency
    batch_size=cfg.training.bc.batch_size
    img_source = cfg.environment.get("img_source", "img")
    clip_eval=cfg.training.bc.clip_eval
    optimizer_step_freq=cfg.training.bc.get("optimizer_step_freq", 1)
    clip_checkpoint=cfg.training.bc.get("clip_checkpoint", None)
    copy_each_epoch=cfg.training.bc.get("copy_each_epoch", False)
    
    if clip_eval:
        clip_sim = get_clip_similarity_score(cp=clip_checkpoint)
    
    def train_step(obs_action_pair, global_train_step, epoch, model=model, model_copy=None):
        inputs = obs_action_pair["observation"]
        target_action = obs_action_pair["action"] 

        outputs = model(**inputs)

        target_action = target_action.to(outputs["single_action"].dtype)

        if isinstance(loss_fn, LanguagePreservationLoss):
            copy_outputs = model_copy(**inputs)
            loss = loss_fn(outputs["single_action"], outputs["vlm_logits"], target_action, copy_outputs["vlm_logits"])
            outputs = outputs.get("single_action")
        else:
            outputs = outputs.get("single_action")
            loss = loss_fn(outputs, target_action)

        if torch.isnan(loss):
            print(outputs)
            print(target_action)
            raise RuntimeError("NaN loss")

        loss.backward()
        
        ## Accumulate gradients
        if global_train_step%optimizer_step_freq==0:
            optimizer.step()
            optimizer.zero_grad()

        log_metrics(loss, "loss", global_train_step, epoch, "train", log_frequency)
        return model


    def inference_step(obs_action_pair, global_inference_step, epoch, model=model, model_copy=None, stage="val"):
        inputs = obs_action_pair["observation"]
        target_action = obs_action_pair["action"] 

        outputs = model(**inputs)

        target_action = target_action.to(outputs["single_action"].dtype)

        if isinstance(loss_fn, LanguagePreservationLoss):
            copy_outputs = model_copy(**inputs)
            loss = loss_fn(outputs["single_action"], outputs["vlm_logits"], target_action, copy_outputs["vlm_logits"])
            outputs = outputs.get("single_action")
        else:
            outputs = outputs.get("single_action")
            loss = loss_fn(outputs, target_action)

        log_metrics(loss, "loss", global_inference_step, epoch, stage, log_frequency)

        if hasattr(model, "generate") and callable(model.generate):
            thought = model.generate(**inputs)
            if clip_eval:
                clip_score = clip_sim(text=thought, images=inputs[img_source])
                log_metrics(clip_score, "CLIP",
                            global_inference_step, epoch, stage, log_frequency)
        
        if i == 0 and j ==0:
            observation = obs_action_pair["observation"]
            log_action_obs_pair(output=outputs, target=obs_action_pair["action_string"], loss_fn=torch.nn.functional.mse_loss, obs_img=observation[img_source], obs_instruction=observation["instruction"], obs_effectors=observation["robot_state"], step=global_inference_step, stage=stage, epoch=epoch, thought=thought, max_rows=20)

        ### Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(outputs, target_action) # pylint: disable=not-callable
        cos_sim = torch.mean(cos_sim)
        log_metrics(cos_sim, "CosSim", global_inference_step, epoch, stage, log_frequency)

        ### Euclidean distance
        eucdist = torch.cdist(outputs, target_action)
        eucdist = torch.mean(eucdist)
        log_metrics(eucdist, "EuclideanDist", global_inference_step, epoch, stage, log_frequency)
        return model
    
    
    ### Sanity Check
    print("\nPerforming Sanity Check.")
    model.eval()
    episode_batch = batch_of_dicts_to_outer_dict(next(iter(val_dataset)))
    obs_action_pair = next(episode_iterator_func(batched_sample=episode_batch, device=device, batch_size=batch_size, shuffle=True, action_discretizer=action_discretizer))
    i, j = 0,0
    model = inference_step(obs_action_pair=obs_action_pair, global_inference_step=-1,
                                   epoch=-1, model=model, model_copy=model_copy)
    print("Sanity Check Complete.")
    
    
    global_train_step = 0
    global_val_step = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        ### Training Loop ###
        if isinstance(loss_fn, LanguagePreservationLoss) and copy_each_epoch:
            if next(model.parameters()).is_cuda:
                model.cuda()
                model_copy.cuda()
            model_copy.load_state_dict(model.state_dict())
            model_copy.thought_generator.requires_grad_(False)

        model.train(True)

        for i, episode_batch in enumerate(train_dataset):
            episode_batch = batch_of_dicts_to_outer_dict(episode_batch)

            for j, obs_action_pair in enumerate(episode_iterator_func(batch=episode_batch, device=device, batch_size=batch_size, shuffle=True, action_discretizer=action_discretizer, state_discretizer=state_discretizer)):
                model = train_step(obs_action_pair=obs_action_pair, global_train_step=global_train_step, epoch=epoch, model=model, model_copy=model_copy)
                global_train_step += 1
            #     break
            # break
        
        ### Validation Loop ###
        model.eval()
        with torch.no_grad():
            for i, episode_batch in enumerate(val_dataset):
                episode_batch = batch_of_dicts_to_outer_dict(episode_batch)

                for j, obs_action_pair in enumerate(episode_iterator_func(batch=episode_batch, device=device, batch_size=batch_size, shuffle=True, action_discretizer=action_discretizer, state_discretizer=state_discretizer)):
                    model = inference_step(obs_action_pair=obs_action_pair, global_inference_step=global_val_step,
                                   epoch=epoch, model=model, model_copy=model_copy, stage="val")
                    global_val_step += 1
                #     break
                # break
    ### Test ###
    model.eval()
    global_test_step = 0
    with torch.no_grad():
        for i, episode_batch in enumerate(test_dataset):
            episode_batch = batch_of_dicts_to_outer_dict(episode_batch)
            for j, obs_action_pair in enumerate(episode_iterator_func(batch=episode_batch, device=device, batch_size=batch_size, shuffle=True)):
                model = inference_step(obs_action_pair=obs_action_pair, global_inference_step=global_test_step,
                                   epoch=epoch, model=model, model_copy=model_copy, stage="test")
                global_test_step += 1
    
    return model
