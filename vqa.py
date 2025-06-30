import wandb
import sys
import torch

from data import decode_raw_vqa_batch
from loss import get_loss_fn
from utils import DotDict, InputWrapper, batch_of_dicts_to_outer_dict, bleu, rouge, meteor

from model.vqa import decode_output, process_batch
from model.clip import get_clip_similarity_score
from log_metrics import log_language_metrics, log_metrics, log_vqa_pair

from einops import rearrange, reduce


def finetune_vlm(model,
                 processor,
                 train_dataset,
                 val_dataset,
                 optimizer,
                 device,
                 dtype,
                 cfg
                 ):

    epochs=cfg.training.vqa.epochs
    loss_fn_id=cfg.training.vqa.get("loss_function")
    log_frequency=cfg.wandb.log_frequency
    single_image=cfg.model.single_image
    num_images=cfg.model.get("num_images", 1)
    use_context=cfg.training.vqa.use_context,
    optimizer_step_freq=cfg.training.vqa.get("optimizer_step_freq", 1)
    prompt_template=cfg.model.get("prompt_template", "Question: <question> Answer:")
    answer_template=cfg.model.get("answer_template", "Answer: <answer>")
    clip_eval=cfg.training.vqa.clip_eval
    clip_checkpoint=cfg.training.vqa.get("clip_checkpoint", None)


    if clip_eval:
        clip_sim = get_clip_similarity_score(cp=clip_checkpoint)

    loss_fn = None
    if loss_fn_id != "builtin":
        loss_fn = get_loss_fn(id=loss_fn_id)

    def train_step(model_inputs, step, epoch):
        output = model(**model_inputs)
        if loss_fn:
            loss = loss_fn(prediction=output, target=model_inputs["labels"])
        else:
            loss = output.loss

        loss.backward()
        log_metrics(loss, "loss", step, epoch, "train", log_frequency)

        ## Accumulate gradients
        if step%optimizer_step_freq==0:
            optimizer.step()
            optimizer.zero_grad()

        return loss

    def inference_step(model_inputs, step, epoch, text, images):
        with torch.no_grad():
            output = model(**model_inputs)
            if loss_fn:
                loss = loss_fn(prediction=output, target=model_inputs["labels"])
            else:
                loss = output.loss
            log_metrics(loss, "loss", step, epoch, "val", log_frequency)

            # run inference
            if model.backbone == "instructblip":
                batch_size = len(text)
                output = []

                if images.ndim == 5:
                    pixel_values = process_batch(processor, images=images, text=None, answers=None, training=False, device=device, dtype=dtype).pixel_values

                for b in range(batch_size):
                    if images.ndim == 5:
                        model_inputs = processor(images=None, text=text[b], return_tensors="pt", max_length=256, truncation=True).to(dtype)
                        model_inputs["pixel_values"] = torch.unsqueeze(pixel_values[b])
                    else:
                        model_inputs = processor(images=images[b], text=text[b], return_tensors="pt", max_length=256, truncation=True).to(dtype)
                    model_inputs = model_inputs.to(device)
                    output.append(model.generate(
                                    model_inputs,
                                    do_sample=False,
                                    num_beams=5,
                                    max_length=256,
                                    min_length=1,
                                    top_p=0.9,
                                    repetition_penalty=1.5,
                                    length_penalty=1.0,
                                    temperature=1)[0])
            else:
                model_inputs.pop("labels", None)
                model_inputs.pop("decoder_attention_mask", None)
                current_length = model_inputs["input_ids"].shape[1]
                model_inputs = process_batch(processor, images=images, text=text, answers=None, training=False, device=device, dtype=dtype)
                # add temperature for more variety
                output = model.generate(
                    **model_inputs, do_sample=False, max_length=current_length+15, min_length=1)
            if isinstance(output, dict):
                decoded_output = decode_output(
                    vqa_processor=processor, raw_outputs=output["outputs"])
            else:
                decoded_output = decode_output(
                    vqa_processor=processor, raw_outputs=output)

            log_language_metrics(decoded_output=decoded_output, answers=answers, step=step, epoch=epoch, log_frequency=log_frequency)

            if clip_eval:
                clip_score = clip_sim(text=decoded_output, images=images)
                log_metrics(clip_score, "CLIP",
                            step, epoch, "val", log_frequency)

            if step == 0 or step == -1:
                # Log first batch of validation set
                if isinstance(output, dict):
                    decoded_thought = decode_output(
                        vqa_processor=processor, raw_outputs=output["thought"])
                else:
                    decoded_thought = None

                log_vqa_pair(decoded_output, decoded_thought, images, text,
                                answers, step, epoch, "val", single_image, max_rows=20)

    def processing_fn(batch_raw):
        batch_raw = batch_of_dicts_to_outer_dict(batch_raw)
        text, images, answers = decode_raw_vqa_batch(
            batch_raw, num_images=num_images, single_image=single_image, use_context=use_context, prompt_template=prompt_template, answer_template=answer_template)
        inputs = process_batch(
            processor, images=images, text=text, answers=answers, training=True, device=device, dtype=dtype)
        return inputs, text, images, answers

    ### Sanity Check
    print("\nPerforming Sanity Check.")
    model.eval()
    batch_raw = next(iter(val_dataset))
    inputs, text, images, answers = processing_fn(batch_raw)
    inference_step(model_inputs=inputs, step=-1, epoch=-1, text=text, images=images)
    print("Sanity Check Complete.")

    if epochs > 0:
        global_val_step = 0
        global_train_step = 0
        for epoch in range(epochs):
            ### Training
            print(f"\nStart of epoch {epoch}")
            model.train(True)
            optimizer.zero_grad()
            for _, batch_raw in enumerate(train_dataset):
                inputs, _, _, _ = processing_fn(batch_raw)
                train_step(model_inputs=inputs, step=global_train_step, epoch=epoch)
                global_train_step += 1

            ### Validation
            model.eval()
            for _, batch_raw in enumerate(val_dataset):
                inputs, text, images, answers =  processing_fn(batch_raw)
                inference_step(model_inputs=inputs, step=global_val_step, epoch=epoch, text=text, images=images)
                global_val_step += 1
    else:
        ### Validation
        model.eval()
        for step, batch_raw in enumerate(val_dataset):
            inputs, text, images, answers =  processing_fn(batch_raw)
            inference_step(model_inputs=inputs, step=step, epoch=0, text=text, images=images)

    return model
