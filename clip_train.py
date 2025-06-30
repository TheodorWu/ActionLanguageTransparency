import torch

from data import decode_raw_vqa_batch
from utils import  batch_of_dicts_to_outer_dict

from model.vqa import process_batch
from log_metrics import log_metrics

def finetune_clip(model,
                 processor,
                 #  processing_fn,
                 train_dataset,
                 val_dataset,
                 epochs,
                 optimizer,
                 log_frequency,
                 single_image,
                 use_context,
                 device, 
                 dtype):

    def train_step(model_inputs, step, epoch):
        optimizer.zero_grad()

        output = model(return_loss=True, **model_inputs)
        loss = output.loss

        loss.backward()
        optimizer.step()
        log_metrics(loss, "loss", step, epoch, "train", log_frequency)
        return loss
    
    def inference_step(model_inputs, step, epoch, text, images):
        with torch.no_grad():
            output = model(return_loss=True, **model_inputs)
            loss = output.loss
            log_metrics(loss, "loss", step, epoch, "val", log_frequency)
    
    def processing_fn(batch_raw):
        batch_raw = batch_of_dicts_to_outer_dict(batch_raw)
        text, images, answers = decode_raw_vqa_batch(
            batch_raw, single_image=single_image, use_context=use_context)
        inputs = process_batch(
            processor, images=images, text=text, answers=answers, training=False, device=device, dtype=dtype)
        return inputs, text, images, answers

    ### Sanity Check
    print("\nPerforming Sanity Check.")
    model.eval()
    batch_raw = next(iter(val_dataset))
    inputs, text, images, answers = processing_fn(batch_raw)
    inference_step(model_inputs=inputs, step=-1, epoch=-1, text=text, images=images)
    print("Sanity Check Complete.")

    if epochs > 0:
        global_train_step = 0
        global_val_step = 0
        for epoch in range(epochs):
            ### Training
            print(f"\nStart of epoch {epoch}")
            model.train(True)
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