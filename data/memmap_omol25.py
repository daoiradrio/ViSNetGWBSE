import warnings
warnings.filterwarnings("ignore")

import torch

import numpy as np

from torch.utils.data import Dataset



np.random.seed(42)



N_MAX = 185



class OMol25Dataset(Dataset):

    def __init__(self, split_set):
        super().__init__()
        self.split_set = split_set
        self.num_data = int(np.loadtxt(f"OMol25/{self.split_set}/num_data.dat"))
        self.Z = np.memmap(f"OMol25/{self.split_set}/Z.npy", dtype="int32", mode="r", shape=(self.num_data, N_MAX))
        self.R = np.memmap(f"OMol25/{self.split_set}/R.npy", dtype="float32", mode="r", shape=(self.num_data, N_MAX, 3))
        self.M = np.memmap(f"OMol25/{self.split_set}/M.npy", dtype="bool", mode="r", shape=(self.num_data, N_MAX))
        self.N = np.memmap(f"OMol25/{self.split_set}/N.npy", dtype="int32", mode="r", shape=(self.num_data,))
        self.E = np.memmap(f"OMol25/{self.split_set}/HOMO.npy", dtype="float32", mode="r", shape=(self.num_data,))


    def __len__(self):
        return self.num_data


    def __getitem__(self, i):
        Z = torch.from_numpy(self.Z[i, :].copy())
        R = torch.from_numpy(self.R[i, :, :].copy())
        M = torch.from_numpy(self.M[i, :].copy())
        N = torch.tensor([self.N[i]])
        E = torch.tensor([self.E[i]])

        Z = Z[M]
        R = R[M, :]

        return N, Z, R, E



class OMol25(torch.nn.Module):

    def __init__(self, num_train, num_val, dataset):
        super().__init__()
        self.train_dataset = OMol25Dataset("train")
        self.val_dataset = OMol25Dataset("val")


    def collate_fn(self, batch):
        curr_batch_size = len(batch)
        N, Z, R, E = zip(*batch)
        N = torch.concatenate(N)
        Z = torch.concatenate(Z)
        R = torch.concatenate(R)
        E = torch.concatenate(E).reshape(-1, 1)
        B = torch.repeat_interleave(torch.arange(curr_batch_size), N)
        return {"z": Z, "pos": R, "batch": B}, E
