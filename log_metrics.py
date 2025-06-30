import wandb
from einops import rearrange
import sys
import torch
from utils import bleu, meteor, rouge
import re

def log_metrics(value, name, step, epoch, stage, log_frequency):
    if step % log_frequency == 0:
        print(f"{stage}/{name} after {step} steps: {value}", file=sys.stdout)
    if step >= 0:
        wandb.log({f"{stage}/{name}": value,
                    f"{stage}/step": step, f"{stage}/epoch": epoch})


def log_vqa_pair(output, thought, images, text, answer, step, epoch, stage, single_image, max_rows=5):
    # requires pip install moviepy imageio
    def image_or_gif(images, single_image):
        images *= 255
        images = images.to(dtype=torch.uint8)
        if single_image:
            return wandb.Image(images)
        else:
            # images = rearrange(images, "n w h c -> n c w h") no longer needed with PT
            return wandb.Video(images)

    cols = ["output", "thought", "image/gif", "input_text",
            "target", "step", "epoch", "stage"]
    table_data = []

    for i, _ in enumerate(output):
        if i < max_rows:  # limit number of logged rows per batch
            table_data.append([
                output[i],
                None if thought is None else thought[i],
                image_or_gif(images=images[i], single_image=single_image),
                text[i],
                answer[i],
                step,
                epoch,
                stage
            ])
    vqa_table = wandb.Table(columns=cols, data=table_data)
    wandb.log({f"{stage}_{epoch}/examples": vqa_table})


def log_action_obs_pair(output, target, loss_fn, obs_img, obs_instruction, obs_effectors, step, epoch, stage, max_rows=5, thought=None):
    cols = ["output", "target", "thought", "loss", "observation/img", "observation/instruction",
            "observation/effector_translation", "step", "epoch", "stage"]
    table_data = []

    for i, _ in enumerate(output):
        if i < max_rows:  # limit number of logged rows per batch
            table_data.append([
                str(output[i]),
                str(target[i]),
                thought[i] if thought else None,
                str(loss_fn(output[i], target[i])),
                wandb.Image((obs_img[i]*255).to(dtype=torch.uint8)),
                obs_instruction[i],
                str(obs_effectors[i]),
                step,
                epoch,
                stage
            ])
    table = wandb.Table(columns=cols, data=table_data)
    wandb.log({f"stage_{stage}/epoch_{epoch}/step_{step}/examples": table})

def log_language_metrics(decoded_output, answers, step, epoch, log_frequency, stage="val"):
    bleu_score = bleu(decoded_output, answers)
    log_metrics(bleu_score, "Sentence BLEU",
                step, epoch, stage, log_frequency)
    
    rouge_scores = rouge(decoded_output, answers)
    for interval, values in rouge_scores["rougeL"]._asdict().items():
        for metric, value in values._asdict().items():
            log_metrics(value, f"ROUGE L {metric} {interval}",
                        step, epoch, stage, log_frequency)
    for interval, values in rouge_scores["rouge1"]._asdict().items():
        for metric, value in values._asdict().items():
            log_metrics(value, f"ROUGE 1 {metric} {interval}",
                        step, epoch, stage, log_frequency)
            
    meteor_score = meteor(decoded_output, answers)
    log_metrics(meteor_score, "METEOR",
                step, epoch, stage, log_frequency)
    
def evaluate_action_tokens(output, target_actions, action_discretizer, step, epoch, stage, log_frequency):
    # output = [
    #     "[action] a22 a37 [action]",
    #     "[action] a10 a12 a13",
    #     "yes",
    #     "24 a24 [action]"]

    decoded_actions = [re.findall("a[0-9]{1,2}", o) for o in output]
    indices = []
    trajectories = []
    for i, decoded_action in enumerate(decoded_actions):
        decoded_action = [ int(a.strip("a")) for a in decoded_action]
        if len(decoded_action) >= 2:
            trajectories.append(decoded_action[:2])
            indices.append(i)
    
    if len(indices) > 0:
        decoded_actions = action_discretizer.reverse(torch.tensor(trajectories)).to(target_actions.device)

        target_actions = torch.index_select(target_actions, 0, torch.tensor(indices, device=target_actions.device))
        
        ## Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(decoded_actions, target_actions) # pylint: disable=not-callable
        cos_sim = torch.mean(cos_sim)
        log_metrics(cos_sim, "CosSim", step, epoch, stage, log_frequency)

        ### Euclidean distance
        eucdist = torch.cdist(decoded_actions, target_actions)
        eucdist = torch.mean(eucdist)
        log_metrics(eucdist, "EuclideanDist", step, epoch, stage, log_frequency)

        ### MSE
        mse = torch.nn.functional.mse_loss(decoded_actions, target_actions)
        log_metrics(mse, "MSE", step, epoch, stage, log_frequency)

def log_binary_accuracy(decoded_output, answers, step, epoch, log_frequency, stage="val"):
    total = len(answers)
    correct = len([t for i, t in enumerate(decoded_output) if t == answers[i]])
    accuracy = correct/total
    log_metrics(accuracy, "Binary Accuracy",
                step, epoch, stage, log_frequency)