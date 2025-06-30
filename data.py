from pathlib import Path
import re
import numpy as np
import torch
from torchdata.datapipes.iter import FileLister, FileOpener
import csv


from utils import batch_of_dicts_to_outer_dict, decode_image_batch

DIR_PATH = Path(__file__).parent.resolve()

def load_roboVQA(mode, folder="processed"):
    data_path = f"{DIR_PATH}/data/RoboVQA/{folder}/{mode}"

    assert Path(data_path).exists(), f"Directory to load RoboVQA dataset from does not exist: {data_path}"

    lister = FileLister(root=data_path, masks="*.tfrecord*")
    opener = FileOpener(lister, mode="b")
    tfrecord_loader_dp = opener.load_from_tfrecord()
    return tfrecord_loader_dp

def decode_raw_vqa_batch(batch_raw, single_image=False, num_images=1, use_context=True, prompt_template="Question: <question> Answer:", answer_template="Answer: <answer>"):
    answers = np.array(batch_raw["answers"]).flatten()
    answers = [ a.decode("utf-8", "ignore").replace("A: ", "") for a in answers ]
    answers = [ answer_template.replace("<answer>", a) for a in answers ]

    text = np.array(batch_raw["questions"]).flatten()
    text = [ re.sub(r"^.\ ", "",t.decode("utf-8", "ignore").replace("Q: ", "")) for t in text ]

    if use_context:
        context = np.array(batch_raw["context"]).flatten()
        context = [ c.decode("utf-8", "ignore")for c in context ]
        text = [f"{c.strip()}. {t}" if c != "" else t for c,t in zip(context, text)]

    for i in range(len(answers)):
        answers[i] = answers[i].replace("<question>", text[i])

    text = [prompt_template.replace("<question>", t) for t in text]

    images = decode_image_batch(batch_raw["images"])

    if num_images==1 or single_image:
        images = images[:,0,:,:,:] # will only take first image of sequence for now (b, n, h, w, c)
    else:
        total_frames = images.size()[1]
        indices = torch.arange(0, total_frames, total_frames / num_images, dtype=torch.int)
        images = torch.index_select(images, 1, indices)

    return text, images, answers

def count_robovqa():
    keys = ["train", "val"]
    with open("robovqa_stats.csv", "w", encoding="utf-8", newline="") as statfile:
        writer = csv.writer(statfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for key in keys:
            ds = load_roboVQA(mode=key, folder="freeform_planning")
            ds = ds.batch(batch_size=4)
            for sample in ds:
                text, _, answers = decode_raw_vqa_batch(batch_of_dicts_to_outer_dict(sample))
                mode = [key] * len(text)
                rows = list(zip(text, answers, mode))
                writer.writerows(rows)

if __name__=="__main__":
    count_robovqa()
    # ds = load_roboVQA(mode="test", folder="processed")
    # ds = ds.batch(batch_size=4)
    # for sample in ds:
    #     sample = batch_of_dicts_to_outer_dict(sample)
    #     print(decode_raw_vqa_batch(sample))
    #     break
