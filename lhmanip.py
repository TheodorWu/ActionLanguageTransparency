import glob
import os
from pathlib import Path
import pickle
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from utils import batch_of_dicts_to_outer_dict

DIR_PATH = Path(__file__).parent.resolve()

class LHManipDataset(Dataset):
    def __init__(self, tasks, mode):

        if mode == "train":
            self.episode_ids = range(7)
        elif mode == "val":
            self.episode_ids = [7,8]
        else:
            self.episode_ids  = [9]
        # Create list of paths to all episodes
        data_path = f"{DIR_PATH}/data/long_horizon_manipulation_dataset/"
        self.tasks_paths = []
        for task_path in glob.glob(f"{data_path}/*"):
            if task_path.split("/")[-1] in tasks:
                self.tasks_paths.append(task_path)

        self.episode_paths = []
        for task_path in self.tasks_paths:
            for i in self.episode_ids:
                self.episode_paths.append(f"{task_path}/{str(i)}")

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        return self.episode_paths[idx]
    
def load_lh_manip(tasks, mode, batch_size):
    ds = LHManipDataset(tasks, mode)
    shuffle = mode == "train"

    return DataLoader(dataset=ds, shuffle=shuffle, batch_size=batch_size)

def lhmanip_get_obs_act_pair(batch, batch_size, device, shuffle=False, action_discretizer=None, action_string="Perform Action: [action]", state_discretizer=None):
    # Read data for each episode
    action_string_start = action_string
    for episode_path in batch:
        with open(f"{episode_path}/data.pkl", "rb") as f:
            data = pickle.load(f)
        num_steps = len(glob.glob(f"{episode_path}/images/*.png"))

        if action_discretizer:
            data['actions'] = action_discretizer(data['actions'])
        else:
            data['actions'] = torch.tensor(data['actions'])

        if state_discretizer:
            data['observations']['states'] = state_discretizer(data['observations']['states'][:,:23])
        else:
            data['observations']['states'] =  torch.tensor(data['observations']['states'])[:,:23]

        batch_of_pairs = []
        if shuffle:
            step_list = torch.randperm(num_steps)
        else:
            step_list = range(num_steps)

        for i in step_list:
            if len(batch_of_pairs) % batch_size == 0 and len(batch_of_pairs) > 0:
                batch_of_pairs = batch_of_dicts_to_outer_dict(batch_of_pairs)
                batch_of_pairs["observation"] = batch_of_dicts_to_outer_dict(batch_of_pairs["observation"])
                yield batch_of_pairs
                batch_of_pairs = []

            robot_state_string = ""
            if state_discretizer:
                discrete_state = data['observations']['states'][i]
                for d in discrete_state:
                    robot_state_string += f" s{d}"
            else:
                robot_state_string = f"{data['observations']['states'][i]}"
            
            action_string = action_string_start
            if action_discretizer:
                discrete_action = data['actions'][i]
                for d in discrete_action:
                    action_string += f" a{d}"
            else:
                action_string += f" {data['actions'][i]}"
            action_string += " [action]"
            
            pair = {
                'observation': {
                    'image': (torchvision.io.read_image(f"{episode_path}/images/{i}.png")/255).to(device),
                    'secondary_image': (torchvision.io.read_image(f"{episode_path}/images_left/{i}.png")/255).to(device),
                    'wrist_image': (torchvision.io.read_image(f"{episode_path}/images_wrist/{i}.png")/255).to(device),
                    # 'depth': (torch.tensor(np.asarray(Image.open(f"{episode_path}/depth/{i}.png")))*data['depth_scales'][i*3]).to(device),
                    # 'secondary_depth': (torch.tensor(np.asarray(Image.open(f"{episode_path}/depth_left/{i}.png")))*data['depth_scales'][i*3+1]).to(device),
                    # 'wrist_depth': (torch.tensor(np.asarray(Image.open(f"{episode_path}/depth_wrist/{i}.png")))*data['depth_scales'][i*3+2]).to(device),
                    'robot_state': data['observations']['states'][i].to(device),
                    'robot_state_string': robot_state_string.strip(),
                    'instruction': data['language_instruction'][i],
                },
                'action': data['actions'][i].to(device),
                'action_string': action_string
            }
            batch_of_pairs.append(pair)