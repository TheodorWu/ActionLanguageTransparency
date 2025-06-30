import torch
import random
import re
from log_metrics import log_language_metrics, log_metrics, log_vqa_pair, evaluate_action_tokens, log_binary_accuracy
from utils import batch_of_dicts_to_outer_dict, DotDict
from model.vqa import process_batch, decode_output

from model.clip import get_clip_similarity_score

def behavior_cloning_vlm_only(model,
                     processor,
                     optimizer,
                     train_dataset,
                     val_dataset,
                     test_dataset,
                     episode_iterator_func, 
                     device,
                     cfg,
                     action_discretizer,
                     state_discretizer=None
                     ):
    
    dtype = model.dtype
    prompt_template=cfg.model.agent.get("prompt_template", "Question: <question> Answer:")
    answer_template=cfg.model.agent.get("answer_template", "Answer: <answer>")
    epochs=cfg.training.bc.epochs
    log_frequency=cfg.wandb.log_frequency
    batch_size=cfg.training.bc.batch_size
    img_source = cfg.environment.get("img_source", "img")
    optimizer_step_freq=cfg.training.bc.get("optimizer_step_freq", 1)
    clip_eval=cfg.training.bc.clip_eval
    clip_checkpoint=cfg.training.bc.get("clip_checkpoint", None)
    state_string_input=cfg.training.bc.get("state_string_input", False)
    string_target = cfg.environment.get("target", "action_string")
    extra_questions_active = cfg.environment.get("extra_questions", False)
    action_first = cfg.environment.get("action_first", False)
    if extra_questions_active:
        batch_size = batch_size // 2

    ### possible early stopping
    stop_train_after = cfg.training.bc.get("stop_train_after", None)
    stop_val_after = cfg.training.bc.get("stop_val_after", None)
    stop_test_after = cfg.training.bc.get("stop_test_after", None)

    def processing_fn(images, text, answers, state=None, extra_questions=None, extra_answers=None):
        text = [prompt_template.replace("<question>", t) for t in text]
        if extra_questions:
            extra_questions = [prompt_template.split("<question>")[0] + t for t in extra_questions]
            text.extend(extra_questions)

        if state:
            text = [t.replace("<state>", state[i]) for i, t in enumerate(text)]
        if answers:
            answers = [answer_template.replace("<answer>", a) for a in answers]
            if extra_answers:
                extra_answers = [answer_template.replace("<answer>", a) for a in extra_answers]
                answers.extend(extra_answers)
            answers = [answers[i].replace("<instruction>", f"{t}.") for i, t in enumerate(text)]
        
        if extra_questions and extra_answers:
            # tmp = list(zip(text, answers, torch.cat((images, images))))
            # random.shuffle(tmp)
            # text, answers, images = zip(*tmp)
            images = torch.stack((images,images))

        inputs = process_batch(
            processor, images=images, text=text, answers=answers, training=True, device=device, dtype=dtype)
        return inputs
    
    def model_call(obs_action_pair, model):
        observation = obs_action_pair["observation"]
        images = observation[img_source]
        text = observation["instruction"]
        target_action = obs_action_pair[string_target]

        state = None
        if state_string_input:
            state = observation["robot_state_string"]

        extra_questions = None
        extra_answers = None
        if extra_questions_active:
            extra_questions = obs_action_pair["extra_questions"]
            extra_answers = obs_action_pair["extra_answers"]

        model_inputs = processing_fn(images=images, text=text, answers=target_action, state=state, extra_questions=extra_questions, extra_answers=extra_answers)
        
        outputs = model(**model_inputs)

        loss = outputs.loss

        if torch.isnan(loss):
            print(outputs)
            print(target_action)
            raise RuntimeError("NaN loss")
        
        return model, loss, model_inputs, outputs

    def model_generate(obs_action_pair, model, processor):
        observation = obs_action_pair["observation"]
        images = observation[img_source]
        text = observation["instruction"]

        state = None
        if state_string_input:
            state = observation["robot_state_string"]

        extra_questions = None
        extra_answers = None
        if extra_questions_active:
            extra_questions = obs_action_pair["extra_questions"]
            extra_answers = obs_action_pair["extra_answers"]

        model_inputs = processing_fn(images=images, text=text, answers=None, state=state, extra_questions=extra_questions, extra_answers=extra_answers)
        output = model.generate(**model_inputs, max_new_tokens=30, do_sample=True, num_beams=1)
        output = decode_output(processor, output)

        return output

    def train_step(obs_action_pair, global_train_step, epoch, model=model):
        model, loss, _, _ = model_call(obs_action_pair, model)

        loss.backward()
        
        ## Accumulate gradients
        if global_train_step%optimizer_step_freq==0:
            optimizer.step()
            optimizer.zero_grad()

        log_metrics(loss, "loss", global_train_step, epoch, "train", log_frequency)
        return model
    
    def inference_step(obs_action_pair, global_inference_step, epoch, model=model, stage="val"):
        model, loss, inputs, _ = model_call(obs_action_pair, model)

        log_metrics(loss, "loss", global_inference_step, epoch, stage, log_frequency)

        if hasattr(model, "generate") and callable(model.generate):
            output = model_generate(obs_action_pair, model, processor)

            if extra_questions_active:
                extra_outputs = [t.split("?", 1)[-1].strip() for t in output if prompt_template.split(" ")[-1] not in t]
                extra_answers = obs_action_pair["extra_answers"]
                log_binary_accuracy(decoded_output=extra_outputs, answers=extra_answers, step=global_inference_step, epoch=epoch, log_frequency=log_frequency, stage=stage)
                output = [t for t in output if prompt_template.split(" ")[-1] in t]
                
            output = [ t.split(prompt_template.split(" ")[-1], 1)[-1].strip() for t in output ]

            if clip_eval:
                clip_score = clip_sim(text=output, images=inputs[img_source])
                log_metrics(clip_score, "CLIP",
                            global_inference_step, epoch, stage, log_frequency)
            
            target_action = obs_action_pair[string_target]
        
            log_language_metrics(decoded_output=output, answers=target_action, step=global_inference_step, epoch=epoch, log_frequency=log_frequency, stage=stage)
            evaluate_action_tokens(output=output, target_actions=obs_action_pair["action"], action_discretizer=action_discretizer, step=global_inference_step, epoch=epoch, stage=stage, log_frequency=log_frequency)
        
            if i == 0 and j ==0:
                observation = obs_action_pair["observation"]
                log_vqa_pair(output, None, observation[img_source], observation["instruction"],                
                                    obs_action_pair[string_target], global_inference_step, epoch, stage, True, max_rows=20)
        return model

    if clip_eval:
        clip_sim = get_clip_similarity_score(cp=clip_checkpoint)

        ### Sanity Check
    print("\nPerforming Sanity Check.")
    model.eval()
    episode_batch = batch_of_dicts_to_outer_dict(next(iter(val_dataset)))
    obs_action_pair = next(episode_iterator_func(batch=episode_batch, device=device, batch_size=batch_size, shuffle=True, action_discretizer=action_discretizer, state_discretizer=state_discretizer, action_first=action_first))
    i, j = 0,0
    model = inference_step(obs_action_pair=obs_action_pair, global_inference_step=-1,
                                   epoch=-1, model=model)
    print("Sanity Check Complete.")
    
    
    global_train_step = 0
    global_val_step = 0
    for epoch in range(epochs):
        ### Training Loop ###
        model.train(True)
        optimizer.zero_grad()

        for i, episode_batch in enumerate(train_dataset):
            episode_batch = batch_of_dicts_to_outer_dict(episode_batch)
            optimizer.zero_grad()

            for j, obs_action_pair in enumerate(episode_iterator_func(batch=episode_batch, device=device, batch_size=batch_size, shuffle=True, action_discretizer=action_discretizer, state_discretizer=state_discretizer, action_first=action_first)):
                model = train_step(obs_action_pair=obs_action_pair, global_train_step=global_train_step, epoch=epoch, model=model)
                global_train_step += 1

                if stop_train_after and global_train_step // (epoch +1) > stop_train_after:
                    break
            
            if stop_train_after and global_train_step // (epoch +1) > stop_train_after:
                break
        
        ### Validation Loop ###
        model.eval()
        with torch.no_grad():
            for i, episode_batch in enumerate(val_dataset):
                episode_batch = batch_of_dicts_to_outer_dict(episode_batch)

                for j, obs_action_pair in enumerate(episode_iterator_func(batch=episode_batch, device=device, batch_size=batch_size, shuffle=True, action_discretizer=action_discretizer, state_discretizer=state_discretizer, action_first=action_first)):
                    model = inference_step(obs_action_pair=obs_action_pair, global_inference_step=global_val_step,
                                   epoch=epoch, model=model, stage="val")
                    global_val_step += 1

                    if stop_val_after and global_val_step // (epoch +1) > stop_val_after:
                        break
                if stop_val_after and global_val_step // (epoch +1) > stop_val_after:
                    break
    ### Test ###
    model.eval()
    global_test_step = 0
    with torch.no_grad():
        for i, episode_batch in enumerate(test_dataset):
            episode_batch = batch_of_dicts_to_outer_dict(episode_batch)
            for j, obs_action_pair in enumerate(episode_iterator_func(batch=episode_batch, device=device, batch_size=batch_size, shuffle=True, action_discretizer=action_discretizer, state_discretizer=state_discretizer, action_first=action_first)):
                model = inference_step(obs_action_pair=obs_action_pair, global_inference_step=global_test_step,
                                   epoch=epoch, model=model, stage="test")
                global_test_step += 1
                if stop_test_after and global_test_step > stop_test_after:
                    break
            
            if stop_test_after and global_test_step > stop_test_after:
                break
    
    return model