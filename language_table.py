from einops import rearrange
from pathlib import Path
import numpy as np
import json
import torch
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
import re
from torchdata.datapipes.iter import FileLister, FileOpener
import uuid
import h5py
import sys, os
import itertools
from einops import rearrange
import csv

from utils import batch_of_dicts_to_outer_dict, decode_image_batch, decode_inst, decode_image_sequence

DIR_PATH = Path(__file__).parent.resolve()

def language_table_get_obs_act_pair(batch, batch_size, shuffle, device, action_discretizer=None, action_string_start="<caption>. Perform Action: ", state_discretizer=None, action_first=False):
    if action_first:
        action_string_start = "Perform Action: "

    for k, v in batch.items():
        if torch.is_tensor(v[0]):
            batch[k] = torch.cat(v)
        else:
            batch[k] = list(itertools.chain.from_iterable(v))

    if shuffle:
        steps = torch.randperm(len(batch["captions"]))
    else:
        steps = torch.arange(len(batch["captions"]))

    for idxs in torch.tensor_split(steps, len(steps) // batch_size):
        actions = torch.index_select(batch["action"], 0, idxs)
        captions = [batch["captions"][idx] for idx in idxs]
        rewards = torch.index_select(batch["reward"], 0, idxs)

        states = torch.index_select(batch["effector_translation"], 0, idxs)

        if state_discretizer:
            discrete_states = state_discretizer(states)
            state_strings = []
            for discrete_state in discrete_states:
                state_string = "[state]"
                for d in discrete_state:
                    state_string += f" s{d}"
                state_strings.append(f"{state_string} [action]")
        else:
            state_strings = [f"[state] {str(state.tolist()).replace('[', '').replace(']', '')} [state]" for state in states]

        if action_discretizer:
            discrete_actions = action_discretizer(actions)
            action_strings = []
            plain_action_strings = []
            for discrete_action in discrete_actions:
                action_string = "[action]"
                for d in discrete_action:
                    action_string += f" a{d}"
                action_strings.append(f"{action_string_start} {action_string} [action]")
                plain_action_strings.append(f"{action_string} [action]")
        else:
            action_strings = [f"{action_string_start} {str(action.tolist()).replace('[', '').replace(']', '')} [action]" for action in actions]

        if action_first:
            action_strings = [f"{act_str} <caption>" for act_str in action_strings]

        pair = {
            "observation": {
                "instruction": [batch["instruction"][idx] for idx in idxs],
                "image": (torch.index_select(batch["frames"], 0, idxs) / 255 ).to(device),
                "effector_translation": torch.index_select(batch["effector_translation"], 0, idxs).to(device),
                "robot_state_string": state_strings
            },
            "caption": captions,
            "action": actions.to(device),
            "action_string": [ act_str.replace("<caption>", captions[i]) for i, act_str in enumerate(action_strings)],
            "action_only_string": plain_action_strings,
            "rewards": rewards.to(device),
            "extra_questions": ["Update verbal statement?" for _ in rewards],
            "extra_answers": [str(bool(reward==1)) for reward in rewards],
        }

        yield pair

def get_captions_loader():
    download_dir = f"{DIR_PATH}/data"
    data_path = f"{download_dir}/language_table"

    dataset_path = f"{data_path}/robotics/language_table/captions/"
    mask = "*.tfrecord*"

    lister = FileLister(root=dataset_path, masks=mask)
    opener = FileOpener(lister, mode="b")
    return opener.load_from_tfrecord()

def extract_instructions():
    tfrecord_loader_dp = get_captions_loader()
    episodes = {
        "episodes": []
    }

    with h5py.File("./data/language_table/captions.hdf5", "w") as f:
        for i, batch in enumerate(tfrecord_loader_dp):

            if i % 10 < 8:
                subset = "train"
            elif i % 10 < 9:
                subset = "validation"
            else:
                subset = "test"

            episode_id = str(uuid.uuid4())
            captions = [ c[0].decode("utf-8") for c in batch["captions"]]
            entry = {
                    "captions": captions,
                    "long_horizon_instructions": batch["long_horizon_instruction"][0].decode("utf-8"),
                    "id": episode_id,
                    "subset": subset
                }
            episodes["episodes"].append(entry)

            # store in hdf5 file under id/<caption>_{caption_idx}
            grp = f.create_group(episode_id)
            for i, cap in enumerate(captions):
                frame_idx = int(batch["start_times"][i])
                img = np.frombuffer(batch["frames"][frame_idx][0], dtype=np.uint8)
                grp.create_dataset(f"{cap.replace(' ', '_')}_{i}", data=img)

    with open("./data/language_table/captions.json", "w", encoding="utf-8") as file:
        json.dump(episodes, file, indent=4)

def get_action_ranges():
    keys = ["part1", "part2", "part3", "part4"]
    min_action = np.Infinity
    max_action = -np.Infinity
    with open("actions.csv", "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for key in keys:
            tfrecord_loader_dp = get_standard_loader(key)
            for batch in tfrecord_loader_dp:
                action = np.array(batch["steps/action"])

                episode_min_action = np.min(action)
                episode_max_action = np.max(action)

                if episode_min_action < min_action:
                    min_action = episode_min_action

                if episode_max_action > max_action:
                    max_action = episode_max_action

                action  = rearrange(action, "(n d) -> n d", d=2)
                writer.writerows(action)

    print(f"Min Action: {min_action}") # Min Action: -0.23736150562763214
    print(f"Max Action: {max_action}") # Max Action: 0.24496802687644958

def get_state_ranges():
    keys = ["part1", "part2", "part3", "part4"]
    min_state = np.Infinity
    max_state = -np.Infinity
    with open("states.csv", "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for key in keys:
            tfrecord_loader_dp = get_standard_loader(key)
            for batch in tfrecord_loader_dp:
                state = np.array(batch["steps/observation/effector_translation"])
                episode_min_state = np.min(state)
                episode_max_state = np.max(state)

                if episode_min_state < min_state:
                    min_state = episode_min_state

                if episode_max_state > max_state:
                    max_state = episode_max_state

                state  = rearrange(state, "(n d) -> n d", d=2)
                writer.writerows(state)

    print(f"Min state: {min_state}") # Min state: -0.23736150562763214
    print(f"Max state: {max_state}") # Max Action: 0.24496802687644958

def psnr(img1, img2):
    mse = np.mean(np.square(np.subtract(img1.astype(np.int16),
                                        img2.astype(np.int16))))
    if mse == 0:
        return np.Inf
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX) - 10 * np.log10(mse)

def decode_inst_no_filter(inst):
    """Utility to decode encoded language instruction"""
    return bytes(inst.tolist()).decode("utf-8")

def get_standard_loader(key):
    download_dir = f"{DIR_PATH}/data"
    data_path = f"{download_dir}/language_table"

    dataset_path = f"{data_path}/robotics/language_table/0.1.0/{key}"
    mask = "*.tfrecord*"

    lister = FileLister(root=dataset_path, masks=mask)
    opener = FileOpener(lister, mode="b")
    tfrecord_loader_dp = opener.load_from_tfrecord()
    return tfrecord_loader_dp

def extract_instructions_and_ids(key):
    tfrecord_loader_dp = get_standard_loader(key)
    unique_counter = 0

    with open("./data/language_table/captions.json", "r", encoding="utf-8") as caption_file:
        caption_dict = json.load(caption_file)

        regex = re.compile(r"(\x00)+")
        for batch in tfrecord_loader_dp:
            episode_id = batch["episode_id"][0].decode("utf-8")
            possible_instructions = []

            captions = decode_inst_no_filter(batch["steps/observation/instruction"])
            captions = re.sub(regex, ", ", captions).split(", ")
            captions = [c for c in set(captions) if c !="" ]

            for caption in captions:
                for episode in caption_dict["episodes"]:
                    if caption in episode["captions"]:
                        possible_instructions.append(episode["long_horizon_instructions"])

            if len(possible_instructions) == 1:
                unique_counter += 1

            record = { "captions": captions,
                      "episode_id": episode_id,
                      "possible_instructions": possible_instructions }

            with open(f"./data/language_table/ids.json", "a", encoding="utf-8") as file:
                json.dump(record, file)
                file.write("\n")
    print(f"Uniquely identified instructions: {unique_counter}")

def merge_datasets(key):
    download_dir = f"{DIR_PATH}/data"
    data_path = f"{download_dir}/language_table"

    dataset_path = f"{data_path}/robotics/language_table/0.1.0/{key}/"
    mask = "*.tfrecord*"

    lister = FileLister(root=dataset_path, masks=mask)
    opener = FileOpener(lister, mode="b")
    tfrecord_loader_dp = opener.load_from_tfrecord()
    unique_counter = 0
    not_found_counter = 0
    with h5py.File("./data/language_table/captions.hdf5", "r") as caption_data_file:
        with h5py.File(f"./data/language_table/language_table_{key}.hdf5", "w") as merged_file:
            dataset = {
                "test": merged_file.create_group("test"),
                "train": merged_file.create_group("train"),
                "validation": merged_file.create_group("validation")
            }

            with open("./data/language_table/captions.json", "r", encoding="utf-8") as caption_file:
                caption_dict = json.load(caption_file)

                regex = re.compile(r"(\x00)+")
                for batch in tfrecord_loader_dp:
                    possible_instructions = []
                    possible_modes = []
                    possible_ids = []
                    similarities = []

                    captions = decode_inst_no_filter(batch["steps/observation/instruction"])
                    captions = re.sub(regex, ", ", captions).split(", ")
                    captions = [c for c in set(captions) if c !="" ]

                    goal_img = decode_image(torch.tensor(np.frombuffer(batch["steps/observation/rgb"][0], dtype=np.uint8))).numpy()

                    for caption in captions:
                        for episode in caption_dict["episodes"]:
                            episode_captions = episode["captions"]

                            if caption in episode_captions:
                                instruction = episode["long_horizon_instructions"]
                                mode = episode["subset"]
                                episode_id = episode["id"]

                                cap_idx = episode_captions.index(caption)

                                ds_path = f"{episode_id}/{caption.replace(' ', '_')}_{cap_idx}"

                                compare_img = caption_data_file[ds_path][()]
                                compare_img = decode_image(torch.tensor(compare_img)).numpy()

                                similarity = psnr(goal_img, compare_img)
                                similarities.append(similarity)

                                possible_modes.append(mode)
                                possible_ids.append(episode_id)
                                possible_instructions.append(instruction)

                                if similarity > 30:
                                    # we found the image
                                    break


                    if len(similarities) > 0:
                        unique_counter += 1
                        sim_idx = np.argmax(similarities)
                        lh_instruction = possible_instructions[sim_idx]
                        subset = possible_modes[sim_idx]
                        episode_id = possible_ids[sim_idx]

                        # record = { "captions": captions,
                        #             "long_horizon_instruction": lh_instruction,
                        #             "possible_instructions": possible_instructions,
                        #             "similarities": similarities
                        #             }

                        clip_id = str(uuid.uuid4())
                        grp = dataset[subset].create_group(clip_id)

                        captions = ", ".join(captions).encode("utf-8")
                        grp.create_dataset("captions", data=captions)

                        lh_instruction = lh_instruction.encode("utf-8")
                        grp.create_dataset("instruction", data=lh_instruction)

                        episode_id = episode_id.encode("utf-8")
                        grp.create_dataset("episode_id", data=episode_id)

                        frames = batch["steps/observation/rgb"]
                        grp.create_dataset("frames", data=frames)

                        effector_translation = batch["steps/observation/effector_translation"]
                        grp.create_dataset("effector_translation", data=effector_translation)

                        effector_target_translation = batch["steps/observation/effector_target_translation"]
                        grp.create_dataset("effector_target_translation", data=effector_target_translation)

                        reward = batch["steps/reward"]
                        grp.create_dataset("reward", data=reward)

                        action = batch["steps/action"]
                        grp.create_dataset("action", data=action)
                    else:
                        not_found_counter += 1
                        # record = { "captions": captions,
                        #             "long_horizon_instruction": [],
                        #             "possible_instructions": possible_instructions,
                        #             "similarities": similarities
                        #             }

                    # with open(f"./data/language_table/merged.json", "a", encoding="utf-8") as file:
                    #     json.dump(record, file, indent=4)
                    #     file.write("\n###\n")
                print(f"Uniquely identified instructions: {unique_counter}")
                print(f"Not found instructions: {not_found_counter}")

class LanguageTable(Dataset):
    def __init__(self, subset="train", subfiles=["part1", "part2"], frames_per_episode=-1) -> None:
        super().__init__()

        self.frames_per_episode = frames_per_episode # -1 refers to all frames taken from episode
        self.subset = subset
        self.data_path = f"{DIR_PATH}/data/language_table/"
        self.h5_files = {subfile: h5py.File(f"{self.data_path}/language_table_{subfile}.hdf5", "r") for subfile in subfiles}
        self.groups = { key: self.h5_files[key][self.subset] for key in self.h5_files.keys() }
        self._register_idxs()

    def _register_idxs(self):
        file_path = f"{DIR_PATH}/data/language_table/idxs_{self.subset}.json"
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as idxs_file:
                self.index_dict = json.load(idxs_file)
        else:
            self.index_dict = {}
            counter = 0
            for part, group in self.groups.items():
                for episode_id in group.keys():
                    self.index_dict[f"{counter}"] = {
                        "part": part,
                        "group": episode_id
                        }
                    counter += 1
            with open(file_path, "w", encoding="utf-8") as idxs_file:
                json.dump(self.index_dict, idxs_file)

    def __getitem__(self, index):
        key = self.index_dict[f"{index}"]["group"]
        part = self.index_dict[f"{index}"]["part"]
        episode = self.groups[part][key]
        frames = decode_image_sequence(episode["frames"][()])

        steps = len(frames)

        effector_translation = torch.tensor(episode["effector_translation"][()])
        effector_translation = rearrange(effector_translation, "(n e) -> n e", e=2)
        effector_target_translation =  torch.tensor(episode["effector_target_translation"][()])
        effector_target_translation = rearrange(effector_target_translation, "(n e) -> n e", e=2)
        reward = torch.tensor(episode["reward"][()])
        action = rearrange(torch.tensor(episode["action"][()]), "(n a) -> n a", a=2)

        if self.frames_per_episode > 0:
            indices = torch.linspace(start=0, end=steps-2, steps = self.frames_per_episode, dtype=torch.int)

            captions = [episode["captions"][()].decode("utf-8")] * self.frames_per_episode
            instruction = [episode["instruction"][()].decode("utf-8")] * self.frames_per_episode

            frames = torch.index_select(frames, 0, indices)
            effector_translation = torch.index_select(effector_translation, 0, indices)
            effector_target_translation = torch.index_select(effector_target_translation, 0, indices)
            reward = torch.index_select(reward, 0, indices)
            action = torch.index_select(action, 0, indices)

        else:
            captions = [episode["captions"][()].decode("utf-8")] * steps
            instruction = [episode["instruction"][()].decode("utf-8")] * steps

        sample = {
            "captions": captions,
            "instruction": instruction,
            "frames": frames,
            "effector_translation": effector_translation,
            "effector_target_translation": effector_target_translation,
            "reward": reward,
            "action": action
        }
        return sample

    def __len__(self):
        return len(self.index_dict)


def language_table_collate(data):
    outer = {}
    first_dictionary = data[0]

    for k in first_dictionary.keys():
        outer[k] = [dictionary[k] for dictionary in data]
    return outer

def load_language_table(mode, subfiles=["part1", "part2"], buffer_size=4, frames_per_episode=-1):
    ds = LanguageTable(subset=mode, subfiles=subfiles, frames_per_episode=frames_per_episode)
    dl = DataLoader(ds, batch_size=buffer_size, shuffle=mode=="train", collate_fn=language_table_collate)
    return dl

if __name__=="__main__":
    # part  = sys.argv[1]
    get_state_ranges()
    # merge_datasets(part)
    # extract_instructions()
    # extract_instructions_and_ids()
    # ids = []
    # with open("./data/language_table/ids.json", "r", encoding="utf-8") as file:
    #     for line in file:
    #         d = json.loads(line.rstrip())
    #         ids.append(d["episode_id"])

    # print(len(ids))
    # print(len(set(ids)))
