import os
import shutil

from ase.io import read



base_path = "/Users/dario/datasets/GWFit"
mol_sets = ["6Z_general", "more_train_general", "test_general", "tm8_OE62", "tm8_bruneval", "train_general"]

mols = []
datas = []
mem = []

for mol_set in mol_sets:
    files = os.listdir(os.path.join(base_path, "results", "qsGW", mol_set, "eqp", "TZ"))
    for file in files:
        mol = file[:-4]
        atoms = read(os.path.join(base_path, mol_set, f"{mol}.xyz"), format="xyz")
        n = 0
        for Z in atoms.get_atomic_numbers():
            if Z not in [1, 6, 7, 8, 9]:
                n += 1
        if n > 0:
            if not mol in mem:
                mols.append(os.path.join(base_path, mol_set, f"{mol}.xyz"))
                datas.append(os.path.join(base_path, "results", "qsGW", mol_set, "eqp", "TZ", file))
                mem.append(mol)

for mol, data in zip(mols, datas):
    shutil.copyfile(mol, os.path.join(os.getcwd(), "mols", os.path.basename(mol)))
    shutil.copyfile(data, os.path.join(os.getcwd(), "E_qp", os.path.basename(data)))
