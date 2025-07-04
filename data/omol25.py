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
        N_memmap = np.memmap(f"OMol25/{self.split_set}/N.npy", dtype="int32", mode="r", shape=(self.num_data,))
        Z_memmap = np.memmap(f"OMol25/{self.split_set}/Z.npy", dtype="int32", mode="r", shape=(self.num_data, N_MAX))
        R_memmap = np.memmap(f"OMol25/{self.split_set}/R.npy", dtype="float32", mode="r", shape=(self.num_data, N_MAX, 3))
        M_memmap = np.memmap(f"OMol25/{self.split_set}/M.npy", dtype="bool", mode="r", shape=(self.num_data, N_MAX))
        E_memmap = np.memmap(f"OMol25/{self.split_set}/HOMO.npy", dtype="float32", mode="r", shape=(self.num_data,))
        self.N = torch.from_numpy(np.asarray(N_memmap).copy())
        self.Z = torch.from_numpy(np.asarray(Z_memmap).copy())
        self.R = torch.from_numpy(np.asarray(R_memmap).copy())
        self.M = torch.from_numpy(np.asarray(M_memmap).copy())
        self.E = torch.from_numpy(np.asarray(E_memmap).copy())
        del N_memmap
        del Z_memmap
        del R_memmap
        del M_memmap
        del E_memmap


    def __len__(self):
        return self.num_data


    def __getitem__(self, i):
        Z = self.Z[i, :]
        R = self.R[i, :, :]
        M = self.M[i, :]
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
