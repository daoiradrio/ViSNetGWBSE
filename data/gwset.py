import os
import torch
import random

import numpy as np

from torch.utils.data import Dataset
from ase.io import read
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds



random.seed(42)
np.random.seed(42)



QM9_SIZE = 133885
QM9_N_MAX = 29



class GWSetDataset(Dataset):

    def __init__(self, data_dir, split_set):
        super().__init__()        
        self.data_dir = data_dir
        self.split_set = split_set
        self.num_data = int(np.loadtxt(f"{self.data_dir}/{self.split_set}/num_data.dat"))
        self.Z = np.memmap(
            f"{self.data_dir}/{self.split_set}/Z.npy",
            dtype="int32",
            mode="r",
            shape=(self.num_data, QM9_N_MAX)
        )
        self.R = np.memmap(
            f"{self.data_dir}/{self.split_set}/R.npy",
            dtype="float32",
            mode="r",
            shape=(self.num_data, QM9_N_MAX, 3)
        )
        self.M = np.memmap(
            f"{self.data_dir}/{self.split_set}/M.npy",
            dtype="bool",
            mode="r",
            shape=(self.num_data, QM9_N_MAX)
        )
        self.N = np.memmap(
            f"{self.data_dir}/{self.split_set}/N.npy",
            dtype="int32",
            mode="r",
            shape=(self.num_data)
        )
        self.E = np.memmap(
            f"{self.data_dir}/{self.split_set}/E.npy",
            dtype="float32",
            mode="r",
            shape=(self.num_data)
        )


    def __len__(self):
        return self.num_data


    def __getitem__(self, i):
        Z = torch.from_numpy(self.Z[i, :].copy())
        R = torch.from_numpy(self.R[i, :, :].copy())
        M = torch.from_numpy(self.M[i, :].copy())
        N = torch.tensor([self.N[i].copy()])
        E = torch.tensor([self.E[i].copy()])

        Z = Z[M]
        R = R[M, :]

        return N, Z, R, E



class GWSet(torch.nn.Module):

    def __init__(
        self,
        data_dir,
        batch_size,
        device,
        num_train=None,
        num_val=None
    ):
        super().__init__()

        self.batch_size = batch_size
        self.device = device
        
        if not os.path.exists(data_dir):
            assert num_train is not None and num_val is not None, "Number of training and validation samples needed for setting up split sets"
            self._prepare_split_sets(data_dir, num_train, num_val)
        
        self.train_dataset = GWSetDataset(data_dir, split_set="train")
        self.val_dataset = GWSetDataset(data_dir, split_set="val")
    

    def setup(self, stage):
        return
    

    def _prepare_split_sets(self, data_dir, num_train, num_val):
        xyz_path = "/Users/dario/datasets/GWSet/QM9/QM9_xyz_files"
        results_path = "/Users/dario/datasets/GWSet/results"
        eqp_path = f"{results_path}/E_qp"
        homo_path = f"{results_path}/homo_idx"

        #idx = [i for i in range(1, num_train + num_val + 1)]
        #'''
        idx = []
        for i in range(1, num_train + num_val + 1):
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
        random.shuffle(idx)
        #'''

        train_idx = idx[:num_train]
        val_idx = idx[num_train : num_train+num_val]

        os.makedirs(data_dir)

        train_dir = os.path.join(data_dir, "train")
        os.makedirs(train_dir)
        
        Z_mm = np.memmap(
            os.path.join(train_dir, "Z.npy"),
            dtype="int32", mode='w+',
            shape=(num_train, QM9_N_MAX)
        )
        R_mm = np.memmap(
            os.path.join(train_dir, "R.npy"),
            dtype="float32",
            mode='w+',
            shape=(num_train, QM9_N_MAX, 3)
        )
        M_mm = np.memmap(
            os.path.join(train_dir, "M.npy"),
            dtype="bool",
            mode='w+',
            shape=(num_train, QM9_N_MAX)
        )
        N_mm = np.memmap(
            os.path.join(train_dir, "N.npy"),
            dtype="int32",
            mode='w+',
            shape=(num_train,)
        )
        E_mm = np.memmap(
            os.path.join(train_dir, "E.npy"),
            dtype="float32",
            mode='w+',
            shape=(num_train,)
        )

        np.savetxt(
            os.path.join(train_dir, f"num_data.dat"),
            X=np.array([len(train_idx)]),
            fmt="%d"
        )

        for n, idx in enumerate(train_idx):
            mol = f"mol_{idx}"
            atoms = read(f"{xyz_path}/{mol}.xyz", format="xyz")
            homo_idx = np.loadtxt(f"{homo_path}/{mol}.dat", dtype=int)
            E = np.loadtxt(f"{eqp_path}/{mol}.dat")[homo_idx]
            N = len(atoms)
            Z = np.pad(
                atoms.get_atomic_numbers(),
                pad_width=((0, QM9_N_MAX - N))
            )
            R = np.pad(
                atoms.get_positions(),
                pad_width=((0, QM9_N_MAX - N), (0, 0))
            )
            M = np.where(Z > 0, True, False)
            Z_mm[n, :] = Z
            R_mm[n, :, :] = R
            M_mm[n, :] = M
            N_mm[n] = N
            E_mm[n] = E
            if n % 30 == 0:
                Z_mm.flush()
                R_mm.flush()
                M_mm.flush()
                N_mm.flush()
                E_mm.flush()
            Z_mm.flush()
            R_mm.flush()
            M_mm.flush()
            N_mm.flush()
            E_mm.flush()

        val_dir= os.path.join(data_dir, "val")
        os.makedirs(val_dir)

        Z_mm = np.memmap(
            os.path.join(val_dir, "Z.npy"),
            dtype="int32", mode='w+',
            shape=(num_val, QM9_N_MAX)
        )
        R_mm = np.memmap(
            os.path.join(val_dir, "R.npy"),
            dtype="float32",
            mode='w+',
            shape=(num_val, QM9_N_MAX, 3)
        )
        M_mm = np.memmap(
            os.path.join(val_dir, "M.npy"),
            dtype="bool",
            mode='w+',
            shape=(num_val, QM9_N_MAX)
        )
        N_mm = np.memmap(
            os.path.join(val_dir, "N.npy"),
            dtype="int32",
            mode='w+',
            shape=(num_train,)
        )
        E_mm = np.memmap(
            os.path.join(val_dir, "E.npy"),
            dtype="float32",
            mode='w+',
            shape=(num_val,)
        )

        np.savetxt(
            os.path.join(val_dir, f"num_data.dat"),
            X=np.array([len(val_idx)]),
            fmt="%d"
        )

        for n, idx in enumerate(val_idx):
            mol = f"mol_{idx}"
            atoms = read(f"{xyz_path}/{mol}.xyz", format="xyz")
            homo_idx = np.loadtxt(f"{homo_path}/{mol}.dat", dtype=int)
            E = np.loadtxt(f"{eqp_path}/{mol}.dat")[homo_idx]
            N = len(atoms)
            Z = np.pad(
                atoms.get_atomic_numbers(),
                pad_width=((0, QM9_N_MAX - N))
            )
            R = np.pad(
                atoms.get_positions(),
                pad_width=((0, QM9_N_MAX - N), (0, 0))
            )
            M = np.where(Z > 0, True, False)
            Z_mm[n, :] = Z
            R_mm[n, :, :] = R
            M_mm[n, :] = M
            N_mm[n] = N
            E_mm[n] = E
            if n % 31 == 0:
                Z_mm.flush()
                R_mm.flush()
                M_mm.flush()
                N_mm.flush()
                E_mm.flush()
            Z_mm.flush()
            R_mm.flush()
            M_mm.flush()
            N_mm.flush()
            E_mm.flush()
    

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
