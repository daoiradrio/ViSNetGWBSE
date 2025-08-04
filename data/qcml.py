import warnings
warnings.filterwarnings("ignore")

import torch

import numpy as np

from torch.utils.data import Dataset



np.random.seed(42)



N_MAX = 33
HARTREE_TO_EV = 27.2114



class QCMLDataset(Dataset):

    def __init__(self, data_path, split_set, num_samples):
        super().__init__()
        self.num_samples = num_samples
        num_data = int(np.loadtxt(f"{data_path}/{split_set}/num_data.dat"))
        Z_memmap = np.memmap(f"{data_path}/{split_set}/Z.npy", dtype="int32", mode="r", shape=(num_data, N_MAX))
        R_memmap = np.memmap(f"{data_path}/{split_set}/R.npy", dtype="float32", mode="r", shape=(num_data, N_MAX, 3))
        M_memmap = np.memmap(f"{data_path}/{split_set}/M.npy", dtype="bool", mode="r", shape=(num_data, N_MAX))
        E_memmap = np.memmap(f"{data_path}/{split_set}/HOMO.npy", dtype="float32", mode="r", shape=(num_data,))
        self.Z = torch.from_numpy(np.asarray(Z_memmap)[:num_samples].copy())
        self.R = torch.from_numpy(np.asarray(R_memmap)[:num_samples].copy())
        self.M = torch.from_numpy(np.asarray(M_memmap)[:num_samples].copy())
        self.E = torch.from_numpy(np.asarray(E_memmap)[:num_samples].copy())
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
        E = torch.tensor([self.E[i] * HARTREE_TO_EV])

        Z = Z[M]
        R = R[M, :]

        N = torch.tensor([Z.shape[0]])

        return N, Z, R, E



class QCML(torch.nn.Module):

    def __init__(self, num_train, num_val, dataset, data_path, target):
        super().__init__()
        self.train_dataset = QCMLDataset(data_path=data_path, split_set="train", num_samples=num_train)
        self.val_dataset = QCMLDataset(data_path=data_path, split_set="val", num_samples=num_val)


    def collate_fn(self, batch):
        curr_batch_size = len(batch)
        N, Z, R, E = zip(*batch)
        N = torch.concatenate(N)
        Z = torch.concatenate(Z)
        R = torch.concatenate(R)
        E = torch.concatenate(E).reshape(-1, 1)
        B = torch.repeat_interleave(torch.arange(start=0, end=curr_batch_size, step=1), repeats=N)
        return {"z": Z, "pos": R, "batch": B}, E
