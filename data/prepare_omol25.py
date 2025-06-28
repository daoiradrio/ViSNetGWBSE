#import multiprocessing
#multiprocessing.set_start_method("fork")
import warnings
warnings.filterwarnings("ignore")

import os
import random

import numpy as np

from fairchem.core.datasets import AseDBDataset
from multiprocessing import Pool
from tqdm import tqdm
from math import ceil
from omegaconf import OmegaConf



# MAX. NUM. ATOMS
#
# val: 
# 110
#
# train:
#          0 - 10 000 000: 176
# 10 000 000 - 20 000 000: 181
# 20 000 000 - 30 000 000: 181
# 30 000 000 - 34 335 828: 181



random.seed(42)
np.random.seed(42)



N_MAX = 185
CHUNK_SIZE = 5000
train_data = None
val_data = None



def process_train_batch(indices):
    Z_list, R_list, HOMO_list, GAP_list, N_list = [], [], [], [], []

    for i in indices:
        atoms = train_data.get_atoms(i)
        Z = atoms.get_atomic_numbers()
        R = atoms.get_positions()
        HOMO = atoms.info["homo_energy"][0]
        GAP = atoms.info["homo_lumo_gap"][0]
        N = len(Z)
        Z_list.append(Z)
        R_list.append(R)
        HOMO_list.append(HOMO)
        GAP_list.append(GAP)
        N_list.append(N)

    batch_size = len(indices)
    Z_padded = np.zeros((batch_size, N_MAX), dtype=np.int32)
    R_padded = np.zeros((batch_size, N_MAX, 3), dtype=np.float32)
    M_padded = np.zeros((batch_size, N_MAX), dtype=bool)
    N_array = np.array(N_list, dtype=np.int32)
    HOMO_array = np.array(HOMO_list, dtype=np.float32)
    GAP_array = np.array(GAP_list, dtype=np.float32)

    for n, (Z, R, N) in enumerate(zip(Z_list, R_list, N_list)):
        Z_padded[n, :N] = Z
        R_padded[n, :N, :] = R
        M_padded[n, :N] = True

    return Z_padded, R_padded, M_padded, N_array, HOMO_array, GAP_array



def main():
    global train_data
    global val_data

    train_data = AseDBDataset({"src": "/home/dbaum1/neutral_train"})
    val_data = AseDBDataset({"src": "/home/dbaum1/neutral_val"})

    cfg = OmegaConf.load(os.path.join(os.getcwd(), "config.yaml"))

    idxs = list(range(len(train_data)))
    random.shuffle(idxs)
    train_idxs = idxs[:cfg.data.num_train]

    os.makedirs(cfg.data.data_dir, exist_ok=True)
    split_dir = os.path.join(cfg.data.data_dir, "train")
    os.makedirs(split_dir, exist_ok=True)
    np.savetxt(os.path.join(split_dir, "num_data.dat"), np.array([cfg.data.num_train]), fmt="%d")

    Z_mm_train = np.memmap(os.path.join(split_dir, "Z.npy"), dtype="int32", mode="w+", shape=(cfg.data.num_train, N_MAX))
    R_mm_train = np.memmap(os.path.join(split_dir, "R.npy"), dtype="float32", mode="w+", shape=(cfg.data.num_train, N_MAX, 3))
    M_mm_train = np.memmap(os.path.join(split_dir, "M.npy"), dtype="bool", mode="w+", shape=(cfg.data.num_train, N_MAX))
    N_mm_train = np.memmap(os.path.join(split_dir, "N.npy"), dtype="int32", mode="w+", shape=(cfg.data.num_train,))
    HOMO_mm_train = np.memmap(os.path.join(split_dir, "HOMO.npy"), dtype="float32", mode="w+", shape=(cfg.data.num_train,))
    GAP_mm_train = np.memmap(os.path.join(split_dir, "GAP.npy"), dtype="float32", mode="w+", shape=(cfg.data.num_train,))

    num_batches = ceil(len(train_idxs) / CHUNK_SIZE)
    index_batches = [train_idxs[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE] for i in range(num_batches)]

    #with Pool(processes=6) as pool:
    #    results = list(tqdm(pool.imap_unordered(process_train_batch, index_batches), total=num_batches))
    
    with Pool(processes=15) as pool:
        imap_it = pool.imap_unordered(process_train_batch, index_batches)
        i = 0
        for results in tqdm(imap_it, total=num_batches):
            Z_batch, R_batch, M_batch, N_batch, HOMO_batch, GAP_batch = results
            bsize = len(N_batch)
            Z_mm_train[i:i+bsize] = Z_batch
            R_mm_train[i:i+bsize] = R_batch
            M_mm_train[i:i+bsize] = M_batch
            N_mm_train[i:i+bsize] = N_batch
            HOMO_mm_train[i:i+bsize] = HOMO_batch
            GAP_mm_train[i:i+bsize] = GAP_batch
            Z_mm_train.flush()
            R_mm_train.flush()
            M_mm_train.flush()
            N_mm_train.flush()
            HOMO_mm_train.flush()
            GAP_mm_train.flush()
            i += bsize
    
    '''
    i = 0
    for Z_batch, R_batch, M_batch, N_batch, HOMO_batch, GAP_batch in results:
        bsize = len(N_batch)
        Z_mm_train[i:i+bsize] = Z_batch
        R_mm_train[i:i+bsize] = R_batch
        M_mm_train[i:i+bsize] = M_batch
        N_mm_train[i:i+bsize] = N_batch
        HOMO_mm_train[i:i+bsize] = HOMO_batch
        GAP_mm_train[i:i+bsize] = GAP_batch
        i += bsize
    '''

    Z_mm_train.flush()
    R_mm_train.flush()
    M_mm_train.flush()
    N_mm_train.flush()
    HOMO_mm_train.flush()
    GAP_mm_train.flush()

    idxs = list(range(len(val_data)))
    random.shuffle(idxs)
    val_idxs = idxs[:cfg.data.num_val]

    split_dir = os.path.join(cfg.data.data_dir, "val")
    os.makedirs(split_dir, exist_ok=True)
    np.savetxt(os.path.join(split_dir, "num_data.dat"), np.array([cfg.data.num_val]), fmt="%d")

    Z_mm_valid = np.memmap(os.path.join(split_dir, "Z.npy"), dtype="int32", mode="w+", shape=(cfg.data.num_val, N_MAX))
    R_mm_valid = np.memmap(os.path.join(split_dir, "R.npy"), dtype="float32", mode="w+", shape=(cfg.data.num_val, N_MAX, 3))
    M_mm_valid = np.memmap(os.path.join(split_dir, "M.npy"), dtype="bool", mode="w+", shape=(cfg.data.num_val, N_MAX))
    N_mm_valid = np.memmap(os.path.join(split_dir, "N.npy"), dtype="int32", mode="w+", shape=(cfg.data.num_val,))
    HOMO_mm_valid = np.memmap(os.path.join(split_dir, "HOMO.npy"), dtype="float32", mode="w+", shape=(cfg.data.num_val,))
    GAP_mm_valid = np.memmap(os.path.join(split_dir, "GAP.npy"), dtype="float32", mode="w+", shape=(cfg.data.num_val,))

    num_batches = ceil(len(val_idxs) / CHUNK_SIZE)
    index_batches = [val_idxs[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE] for i in range(num_batches)]

    #with Pool(processes=6) as pool:
    #    results = list(tqdm(pool.imap_unordered(process_train_batch, index_batches), total=num_batches))
    
    with Pool(processes=15) as pool:
        imap_it = pool.imap_unordered(process_train_batch, index_batches)
        i = 0
        for results in tqdm(imap_it, total=num_batches):
            Z_batch, R_batch, M_batch, N_batch, HOMO_batch, GAP_batch = results
            bsize = len(N_batch)
            Z_mm_valid[i:i+bsize] = Z_batch
            R_mm_valid[i:i+bsize] = R_batch
            M_mm_valid[i:i+bsize] = M_batch
            N_mm_valid[i:i+bsize] = N_batch
            HOMO_mm_valid[i:i+bsize] = HOMO_batch
            GAP_mm_valid[i:i+bsize] = GAP_batch
            Z_mm_valid.flush()
            R_mm_valid.flush()
            M_mm_valid.flush()
            N_mm_valid.flush()
            HOMO_mm_valid.flush()
            GAP_mm_valid.flush()
            i += bsize
    '''
    i = 0
    for Z_batch, R_batch, M_batch, N_batch, HOMO_batch, GAP_batch in results:
        bsize = len(N_batch)
        Z_mm_valid[i:i+bsize] = Z_batch
        R_mm_valid[i:i+bsize] = R_batch
        M_mm_valid[i:i+bsize] = M_batch
        N_mm_valid[i:i+bsize] = N_batch
        HOMO_mm_valid[i:i+bsize] = HOMO_batch
        GAP_mm_valid[i:i+bsize] = GAP_batch
        i += bsize
    '''

    Z_mm_valid.flush()
    R_mm_valid.flush()
    M_mm_valid.flush()
    N_mm_valid.flush()
    HOMO_mm_valid.flush()
    GAP_mm_valid.flush()



if __name__ == "__main__":
    main()
