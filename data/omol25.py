import warnings
warnings.filterwarnings("ignore")

import torch

import numpy as np

from torch.utils.data import Dataset



np.random.seed(42)



N_MAX = 185



class OMol25Dataset(Dataset):

    def __init__(self, data_path, split_set, num_samples, target):
        super().__init__()
        self.num_samples = num_samples
        num_data = int(np.loadtxt(f"{data_path}/{split_set}/num_data.dat"))
        N_memmap = np.memmap(f"{data_path}/{split_set}/N.npy", dtype="int32", mode="r", shape=(num_data,))
        Z_memmap = np.memmap(f"{data_path}/{split_set}/Z.npy", dtype="int32", mode="r", shape=(num_data, N_MAX))
        R_memmap = np.memmap(f"{data_path}/{split_set}/R.npy", dtype="float32", mode="r", shape=(num_data, N_MAX, 3))
        M_memmap = np.memmap(f"{data_path}/{split_set}/M.npy", dtype="bool", mode="r", shape=(num_data, N_MAX))
        if target == "HOMO" or target == "GAP" or target == "E":
            E_memmap = np.memmap(f"{data_path}/{split_set}/{target}.npy", dtype="float32", mode="r", shape=(num_data,))
        elif target == "LUMO":
            HOMO_memmap = np.memmap(f"{data_path}/{split_set}/HOMO.npy", dtype="float32", mode="r", shape=(num_data,))
            GAP_memmap = np.memmap(f"{data_path}/{split_set}/GAP.npy", dtype="float32", mode="r", shape=(num_data,))
            E_memmap = GAP_memmap + HOMO_memmap
        self.N = torch.from_numpy(np.asarray(N_memmap)[:num_samples].copy())
        self.Z = torch.from_numpy(np.asarray(Z_memmap)[:num_samples].copy())
        self.R = torch.from_numpy(np.asarray(R_memmap)[:num_samples].copy())
        self.M = torch.from_numpy(np.asarray(M_memmap)[:num_samples].copy())
        self.E = torch.from_numpy(np.asarray(E_memmap)[:num_samples].copy())
        del N_memmap
        del Z_memmap
        del R_memmap
        del M_memmap
        del E_memmap


    def __len__(self):
        return self.num_samples


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

    def __init__(self, num_train, num_val, dataset, data_path, target):
        super().__init__()
        self.train_dataset = OMol25Dataset(data_path=data_path, split_set="train", num_samples=num_train, target=target)
        self.val_dataset = OMol25Dataset(data_path=data_path, split_set="val", num_samples=num_val, target=target)


    def collate_fn(self, batch):
        curr_batch_size = len(batch)
        N, Z, R, E = zip(*batch)
        N = torch.concatenate(N)
        Z = torch.concatenate(Z)
        R = torch.concatenate(R)
        E = torch.concatenate(E).reshape(-1, 1)
        B = torch.repeat_interleave(torch.arange(curr_batch_size), N)
        return {"z": Z, "pos": R, "batch": B}, E
