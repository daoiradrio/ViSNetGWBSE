import os
import torch
import random

import numpy as np

from torch.utils.data import Dataset
from torch.nn.functional import pad
from ase.io import read
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from tqdm import tqdm



random.seed(42)
np.random.seed(42)



QM9_SIZE = 133885
QM9_N_MAX = 29



class GWSetDataset(Dataset):

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



class GWSet(torch.nn.Module):

    def __init__(
        self,
        num_train,
        num_val,
        dataset,
        data_path
    ):
        super().__init__()
        self.data_path = data_path
        if not os.path.exists(os.path.join(os.getcwd(), data_path)):
            self._prepare_data(num_train, num_val)
        self._read_data(num_train, num_val)


    def _read_data(self, num_train, num_val):
        train_path = os.path.join(self.data_path, "train")
        val_path = os.path.join(self.data_path, "val")
        
        N_train = torch.load(os.path.join(train_path, "N.pt"))
        Z_train = torch.load(os.path.join(train_path, "Z.pt"))
        R_train = torch.load(os.path.join(train_path, "R.pt"))
        M_train = torch.load(os.path.join(train_path, "M.pt"))
        E_train = torch.load(os.path.join(train_path, "E.pt"))
        N_val = torch.load(os.path.join(val_path, "N.pt"))
        Z_val = torch.load(os.path.join(val_path, "Z.pt"))
        R_val = torch.load(os.path.join(val_path, "R.pt"))
        M_val = torch.load(os.path.join(val_path, "M.pt"))
        E_val = torch.load(os.path.join(val_path, "E.pt"))

        self.train_dataset = GWSetDataset(num_train, N_train, Z_train, R_train, M_train, E_train)
        self.val_dataset = GWSetDataset(num_val, N_val, Z_val, R_val, M_val, E_val)


    def _prepare_data(self, num_train, num_val):
        xyz_path = "/Users/dario/datasets/GWSet/QM9/QM9_xyz_files"
        results_path = "/Users/dario/datasets/GWSet/results"
        eqp_path = f"{results_path}/E_qp"
        homo_path = f"{results_path}/homo_idx"

        #idx = [i for i in range(1, QM9_SIZE + 1)]
        idx = []
        print()
        print("Checking molecules...")
        for i in tqdm(range(1, QM9_SIZE + 1), leave=False):
            try:
                raw_mol = Chem.MolFromXYZFile(f"{xyz_path}/mol_{i}.xyz")
                mol = Chem.Mol(raw_mol)
                rdDetermineBonds.DetermineBonds(mol)
                for atom in mol.GetAtoms():
                    if atom.GetFormalCharge() != 0:
                        continue
                idx.append(i)
            except:
                pass
        print("Done.")
        random.shuffle(idx)
        train_idx = idx[:num_train]
        val_idx = idx[num_train:min(len(idx), num_train+num_val)]

        all_N = []
        all_Z = []
        all_R = []
        all_M = []
        all_E = []
        train_path = os.path.join(self.data_path, "train")
        print("Preparing training data...")
        os.makedirs(os.path.join(self.data_path, "train"))
        for i in tqdm(train_idx, leave=False):
            mol = f"mol_{i}"
            atoms = read(f"{xyz_path}/{mol}.xyz", format="xyz")
            homo_idx = np.loadtxt(f"{homo_path}/{mol}.dat", dtype=int)
            N = torch.tensor([len(atoms)])
            Z = pad(
                torch.from_numpy(atoms.get_atomic_numbers()),
                pad=(0, QM9_N_MAX - N)
            )
            R = pad(
                torch.from_numpy(atoms.get_positions()),
                pad=(0, 0, 0, QM9_N_MAX - N)
            )
            M = torch.where(Z > 0, True, False)
            E = torch.tensor([np.loadtxt(f"{eqp_path}/{mol}.dat")[homo_idx]])
            all_N.append(N)
            all_Z.append(Z)
            all_R.append(R)
            all_M.append(M)
            all_E.append(E)
        N_train = torch.stack(all_N)
        Z_train = torch.stack(all_Z)
        R_train = torch.stack(all_R).to(dtype=torch.float32)
        M_train = torch.stack(all_M)
        E_train = torch.stack(all_E).to(dtype=torch.float32)
        torch.save(torch.tensor([len(train_idx)]), os.path.join(train_path, "num_samples.pt"))
        torch.save(N_train, os.path.join(train_path, "N.pt"))
        torch.save(Z_train, os.path.join(train_path, "Z.pt"))
        torch.save(R_train, os.path.join(train_path, "R.pt"))
        torch.save(M_train, os.path.join(train_path, "M.pt"))
        torch.save(E_train, os.path.join(train_path, "E.pt"))
        print("Done.")

        all_N = []
        all_Z = []
        all_R = []
        all_M = []
        all_E = []
        val_path = os.path.join(self.data_path, "val")
        print("Preparing validation data...")
        os.makedirs(os.path.join(self.data_path, "val"))
        for i in tqdm(val_idx, leave=False):
            mol = f"mol_{i}"
            atoms = read(f"{xyz_path}/{mol}.xyz", format="xyz")
            homo_idx = np.loadtxt(f"{homo_path}/{mol}.dat", dtype=int)
            N = torch.tensor([len(atoms)])
            Z = pad(
                torch.from_numpy(atoms.get_atomic_numbers()),
                pad=(0, QM9_N_MAX - N)
            )
            R = pad(
                torch.from_numpy(atoms.get_positions()),
                pad=(0, 0, 0, QM9_N_MAX - N)
            )
            M = torch.where(Z > 0, True, False)
            E = torch.tensor([np.loadtxt(f"{eqp_path}/{mol}.dat")[homo_idx]])
            all_N.append(N)
            all_Z.append(Z)
            all_R.append(R)
            all_M.append(M)
            all_E.append(E)
        N_val = torch.stack(all_N)
        Z_val = torch.stack(all_Z)
        R_val = torch.stack(all_R).to(dtype=torch.float32)
        M_val = torch.stack(all_M)
        E_val = torch.stack(all_E).to(dtype=torch.float32)
        torch.save(torch.tensor([len(val_idx)]), os.path.join(val_path, "num_samples.pt"))
        torch.save(N_val, os.path.join(val_path, "N.pt"))
        torch.save(Z_val, os.path.join(val_path, "Z.pt"))
        torch.save(R_val, os.path.join(val_path, "R.pt"))
        torch.save(M_val, os.path.join(val_path, "M.pt"))
        torch.save(E_val, os.path.join(val_path, "E.pt"))
        print("Done.")


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
