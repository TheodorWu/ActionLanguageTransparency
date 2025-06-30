from pathlib import Path
from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader, Mapper
import torchvision
from einops import rearrange
import pickle
import torch
from PIL import Image
import numpy as np
import glob

# from furniture_bench_rl import furniture_bench_get_obs_act_pair
from language_table import language_table_get_obs_act_pair, load_language_table
from lhmanip import lhmanip_get_obs_act_pair, load_lh_manip
# from arnold import load_arnold, arnold_get_obs_act_pair
from utils import decode_image_batch, decode_inst, batch_of_dicts_to_outer_dict

DIR_PATH = Path(__file__).parent.resolve()

def init_env_for_imitation(dataset_id, mode, **kwargs):
    if dataset_id =="language_table":
        buffer_size = kwargs.pop("buffer_size", 4)
        subfiles = kwargs.pop("subfiles", ["part1"])
        frames_per_episode = kwargs.pop("frames_per_episode", -1)
        ds = load_language_table(mode, buffer_size=buffer_size, subfiles=subfiles, frames_per_episode=frames_per_episode)
    elif dataset_id in ["furniture_bench"]:
        ds = load_from_local_directory(dataset_id, mode)
    elif dataset_id in ["lhmanip"]:
        tasks = kwargs.pop("tasks",["match_the_cups_with_the_appropriate_bowls"])
        ds = load_lh_manip(tasks=tasks, mode=mode, batch_size=2)
    # elif dataset_id in ["arnold"]:
    #     batch_size = kwargs.pop("batch_size", 2)
    #     task = kwargs.pop("task", "transfer_water")
    #     ds = load_arnold(task, mode, batch_size=batch_size)
    else:
        raise NameError(f"Dataset with id: {dataset_id} not found")

    return ds, get_pair_function(dataset_id)

def get_pair_function(dataset_id):
    if dataset_id == "language_table":
        return language_table_get_obs_act_pair
    # elif dataset_id == "furniture_bench":
    #     return furniture_bench_get_obs_act_pair
    elif dataset_id == "lhmanip":
        return lhmanip_get_obs_act_pair
    # elif dataset_id == "arnold":
    #     return arnold_get_obs_act_pair
    else:
        raise NameError(f"Pair function for dataset with id {dataset_id} not found")

def load_from_local_directory(dataset_id, mode):
    download_dir = f"{DIR_PATH}/data"
    data_path = f"{download_dir}/{dataset_id}"

    dataset_id_to_pattern = { "language_table": f"{data_path}/robotics/language_table/0.0.1/{mode}",
                             "furniture_bench": f"{data_path}/low/lamp/{mode}" }

    dataset_id_to_mask = { "language_table": "*.tfrecord*",
                          "furniture_bench": "*.pkl"}

    dataset_path = dataset_id_to_pattern[dataset_id]
    mask = dataset_id_to_mask[dataset_id]

    lister = FileLister(root=dataset_path, masks=mask)
    opener = FileOpener(lister, mode="b")
    if "tfrecord" in mask:
        tfrecord_loader_dp = opener.load_from_tfrecord()

        return tfrecord_loader_dp
    else:
        dp = StreamReader(opener)
        dp = Mapper(dp, lambda x: pickle.loads(x[1]))
        return dp

def test_env():
    # import base64
    device = torch.device("cpu")
    ds, pair_function = init_env_for_imitation("language_table", "train", subfiles=["part1", "part2"])
    for batch in ds:
        batch = batch_of_dicts_to_outer_dict(batch)
        # s = base64.b64decode(batch["episode_id"][0][0])
        for sample in pair_function(batch, batch_size=4, device=device, shuffle=True):
            print(sample[0]["observation"]["instruction"])
        # break
    # print(ds.element_spec)

if __name__=="__main__":
    test_env()
