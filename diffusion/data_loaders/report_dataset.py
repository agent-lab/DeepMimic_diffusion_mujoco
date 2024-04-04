import torch
import os
from torch.utils.data import Dataset
from utils.mocap_v2 import MocapDM
# from diffusion.utils.mocap_v2 import MocapDM
import numpy as np

class SpinkickPureFramesDataset(Dataset):
    def __init__(self, motion_src_path):
        self.motion_data = []
        self.mocap_dm = MocapDM()

        self.mocap_dm.load_mocap(filepath=motion_src_path)
        self.motion_data = torch.tensor(self.mocap_dm.frames_raw)
        print(self.motion_data.shape)
        self.motion_data = self.motion_data.float().unsqueeze(0).repeat(16,1,1)

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, idx):
        return self.motion_data[idx]

if __name__ == '__main__':
    dataset = SpinkickPureFramesDataset("/home/kenji/Fyp/DeepMimic_mujoco/diffusion/data/motions/humanoid3d_spinkick.txt")
    print(dataset.motion_data.shape)
    print(dataset.motion_data[0][1] == dataset.motion_data[1][0])
    print(dataset.motion_data)