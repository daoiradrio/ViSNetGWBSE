import os
import torch
import h5py
import json
import random

import pandas as pd

from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.functional import pad



random.seed(42)



QCDGE_N_MAX = 32



class QCDGEDataset(Dataset):

    def __init__(self, num_samples, N, Z, R, M, E):
        super().__init__()
        self.num_samples = num_samples
        self.N = N
        self.Z = Z
        self.R = R
        self.M = M
        self.E = E
        

    def __len__(self):
        return self.num_samples


    def __getitem__(self, i):
        N_sample = self.N[i, :]
        Z_sample = self.Z[i, :]
        R_sample = self.R[i, :, :]
        M_sample = self.M[i, :]
        E_sample = self.E[i, :]

        Z_sample = Z_sample[M_sample]
        R_sample = R_sample[M_sample, :]

        return N_sample, Z_sample, R_sample, E_sample



class QCDGE(torch.nn.Module):

    def __init__(
        self,
        num_train,
        num_val,
        num_test,
        dataset,
        data_path,
        target,
        remove_charged
    ):
        super().__init__()
        if not os.path.exists(data_path):
            self._prepare_data(data_path, num_train)
        self._read_data(data_path)
    

    def _read_data(self, data_path):
        train_path = os.path.join(data_path, "train")
        val_path = os.path.join(data_path, "val")
        
        num_train = torch.load(os.path.join(train_path, "num_samples.pt"))
        N_train = torch.load(os.path.join(train_path, "N.pt"))
        Z_train = torch.load(os.path.join(train_path, "Z.pt"))
        R_train = torch.load(os.path.join(train_path, "R.pt"))
        M_train = torch.load(os.path.join(train_path, "M.pt"))
        E_train = torch.load(os.path.join(train_path, "E.pt"))

        num_val = torch.load(os.path.join(val_path, "num_samples.pt"))
        N_val = torch.load(os.path.join(val_path, "N.pt"))
        Z_val = torch.load(os.path.join(val_path, "Z.pt"))
        R_val = torch.load(os.path.join(val_path, "R.pt"))
        M_val = torch.load(os.path.join(val_path, "M.pt"))
        E_val = torch.load(os.path.join(val_path, "E.pt"))

        self.train_dataset = QCDGEDataset(num_train, N_train, Z_train, R_train, M_train, E_train)
        self.val_dataset = QCDGEDataset(num_val, N_val, Z_val, R_val, M_val, E_val)
    

    def _prepare_data(self, data_path, num_train):
        mols_list = pd.read_csv("/Volumes/LaCie/QCDGE/final_all.csv")

        num_mols = len(mols_list)
        idx = [i for i in range(num_mols)]
        random.shuffle(idx)
        train_idx = idx[:num_train]
        val_idx = idx[num_train:]

        print()
        print(f"{len(idx)} samples in total.")

        split_sets = {"train": train_idx, "val": val_idx}
        for split_set, split_idx in split_sets.items():
            all_N = []
            all_Z = []
            all_R = []
            all_M = []
            all_E = []
            split_path = os.path.join(data_path, split_set)
            os.makedirs(split_path)
            print(f"Preparing {split_set} data...")
            with h5py.File("/Volumes/LaCie/QCDGE/final_all.hdf5", "r") as f:
                for i in tqdm(split_idx, leave=False):
                    atomic_numbers = f[mols_list.iloc[i]["Index"]]["ground_state"]["labels"][()][0]
                    N = torch.tensor([atomic_numbers.size])
                    Z = pad(
                        torch.from_numpy(atomic_numbers),
                        pad=(0, QCDGE_N_MAX - atomic_numbers.size)
                    )
                    coords = f[mols_list.iloc[i]["Index"]]["ground_state"]["coords"][()]
                    R = pad(
                        torch.from_numpy(coords),
                        pad=(0, 0, 0, QCDGE_N_MAX - atomic_numbers.size)
                    )
                    exc_state_raw_bytes = f[mols_list.iloc[i]["Index"]]["excited_state"]["Info_of_AllExcitedStates"][()][0]
                    exc_state_raw_str = exc_state_raw_bytes.decode("utf-8")
                    exc_state_data = json.loads(exc_state_raw_str)
                    E = torch.tensor([float(exc_state_data["1"]["excitation_e_eV"][:-3])])
                    M = torch.where(Z > 0, True, False)
                    all_N.append(N)
                    all_Z.append(Z)
                    all_R.append(R)
                    all_M.append(M)
                    all_E.append(E)
                N = torch.stack(all_N)
                Z = torch.stack(all_Z)
                R = torch.stack(all_R).to(dtype=torch.float32)
                M = torch.stack(all_M)
                E = torch.stack(all_E).to(dtype=torch.float32)
                torch.save(torch.tensor([len(split_idx)]), os.path.join(split_path, "num_samples.pt"))
                torch.save(N, os.path.join(split_path, "N.pt"))
                torch.save(Z, os.path.join(split_path, "Z.pt"))
                torch.save(R, os.path.join(split_path, "R.pt"))
                torch.save(M, os.path.join(split_path, "M.pt"))
                torch.save(E, os.path.join(split_path, "E.pt"))
                print("Done.")
        print()


    def collate_fn(self, batch):
        curr_batch_size = len(batch)
        N, Z, R, E = zip(*batch)
        N = torch.concatenate(N)
        Z = torch.concatenate(Z)
        R = torch.concatenate(R)
        E = torch.concatenate(E).reshape(-1, 1)
        B = torch.repeat_interleave(
            torch.arange(start=0, end=curr_batch_size, step=1),
            repeats=N
        )
        return {"z": Z, "pos": R, "batch": B}, E
