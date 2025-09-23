import os

import pandas as pd
import h5py

from rdkit import Chem


class extractData():
    def __init__(self, mol_list=[]):
        option = int(input('Enter a specific number to choose a way to filter molecules \n1 - all molecules\n2 - a list of molecular index (You should provide a list using the keyword "mol_list" in code)\n3 - filter according to some criteria\n9 - HELP FOR PROPs\n'))

        self.option = option
        self.hdf5 = '/Volumes/LaCie/QCDGE/final_all.hdf5'
        # self.hdf5 = '/data/home/zhuyf/dataset_work/database/checkOptedGeoms/final_all.hdf5'
        self.csv = './final_all_remove_duplicate.csv'

        self.props
        self.props_list = self.collect_prop_list()

        if self.option == 2:
            if not mol_list:
                print('ERROR: NO MOL_LIST INPUT!\n')
                os._exit(0)
            self.mol_list = mol_list
        elif self.option == 3:
            self.mol_list = self.filter_molecules()


    def read_from_hdf5(self):
        data = {}
        with h5py.File(self.hdf5 , 'r') as f:
            for prop_ind in self.props_list:
                data[prop_ind] = []
                keyword = 'ground_state' if prop_ind <= 14 else 'excited_state'
                prop = self.data[str(prop_ind)]['Key']


                if self.option != 1:
                    for db_id in self.mol_list:
                        db_name = str(db_id)
                        if db_name in f:
                            data[prop_ind].append(f[db_name][keyword][prop][()].tolist())
                else:
                    for db_name in f:
                        data[prop_ind].append(f[db_name][keyword][prop][()])
            return data

    @property
    def props(self):
        from rich.table import Table
        from rich.console import Console
        self.data = {
            '1': {'Key': 'labels', 'Description': 'Atomic labels.'},
            '2': {'Key': 'coords', 'Description': 'Optimized Cartesian coordinates.'},
            '3': {'Key': 'Etot', 'Description': 'Total energy.'},
            '4': {'Key': 'e_homo_lumo', 'Description': 'Energy of HOMO and LUMO.'},
            '5': {'Key': 'polarizability', 'Description': 'Isotropic polarizability.'},
            '6': {'Key': 'dipole', 'Description': 'Dipole moment.'},
            '7': {'Key': 'quadrupole', 'Description': 'Quadrupole moment.'},
            '8': {'Key': 'zpve', 'Description': 'Zero point vibrational energy.'},
            '9': {'Key': 'rot_constants', 'Description': 'Rotational constant.'},
            '10': {'Key': 'elec_spatial_ext',
            'Description': 'Electronic spatial extent.'},
            '11': {'Key': 'thermal', 'Description': 'Thermal properties at 298.15 K.'},
            '12': {'Key': 'freqs', 'Description': 'Harmonic vibrational frequencies.'},
            '13': {'Key': 'mulliken', 'Description': 'Mulliken charges.'},
            '14': {'Key': 'cv', 'Description': 'Heat capacity at 298.15 K.'},
            '15': {'Key': 'Etot', 'Description': 'Total energy.'},
            '16': {'Key': 'e_homo_lumo', 'Description': 'Energy of HOMO and LUMO.'},
            '17': {'Key': 'dipole', 'Description': 'Dipole moment.'},
            '18': {'Key': 'quadrupole', 'Description': 'Quadrupole moment.'},
            '19': {'Key': 'rot_constants', 'Description': 'Rotational constant.'},
            '20': {'Key': 'elec_spatial_ext',
            'Description': 'Electronic spatial extent.'},
            '21': {'Key': 'mulliken', 'Description': 'Mulliken charges.'},
            '22': {'Key': 'transition_electric_DM',
            'Description': 'Transition electric dipole moments.'},
            '23': {'Key': 'transition_velocity_DM',
            'Description': 'Transition velocity dipole moments.'},
            '24': {'Key': 'transition_magnetic_DM',
            'Description': 'Transition magnetic dipole moments.'},
            '25': {'Key': 'transition_velocity_QM',
            'Description': 'Transition velocity quadrupole moments.'},
            '26': {'Key': 'OrbNum_HomoLumo',
            'Description': 'The orbital number of HOMO and LUMO.'},
            '27': {'Key': 'Info_of_AllExcitedStates',
            'Description': 'The transition contribution of 10 singlet and 10 triplet transition states.'}
        }
        console = Console()
        # ground-state
        console.print('Props from ground-state calculation (b3lyp/6-31g*+BJD3):')
        gs_table = Table(show_header=True, header_style="bold green", border_style = "bold green")
        gs_table.add_column("No.", style="dim")
        gs_table.add_column("Key")
        gs_table.add_column("Description")
        for no in range(1, 15):
            no_str = str(no)
            if no_str in self.data:
                item = self.data[no_str]
                gs_table.add_row(no_str, item["Key"], item["Description"])
        console.print(gs_table)

        #excited-state
        console.print('\nProps from excited-state calculation (wb97xd/6-31g*):')
        es_table = Table(show_header=True, header_style="bold green", border_style = "bold green")
        es_table.add_column("No.", style="dim")
        es_table.add_column("Key")
        es_table.add_column("Description")
        for no in range(15, 28):
            no_str = str(no)
            if no_str in self.data:
                item = self.data[no_str]
                es_table.add_row(no_str, item["Key"], item["Description"])
        console.print(es_table)

    @staticmethod
    def collect_prop_list():
        props = []
        print("\nPlease enter the numbers, separated by the enter key, ending with '#'. If you enter '*', all props are included:\n")

        while True:
            user_input = input()
            if user_input == '*':
                props = list(range(1,28))
                break
            elif user_input == '#':
                break
            try:
                prop = int(user_input)
                props.append(prop)
            except ValueError:
                print("Invalid input, please enter a number or '#' to end.\n")
        print(props)
        return props

    def filter_molecules(self):
        heavy_atom_threshold = input('\nPlease enter the number of heavy atoms in the desired molecule.\ne.g.:\n8 - One number represents the number of heavy atoms less than or equal to that number. Here, molecules with <=8 heavy atoms will be selected.\n2,6 - two numbers separated by a comma represent an interval [2,6].)\n')

        element_input = input('\nHeavy atom type: 6 - C, 7 - N, 8 - O, 9 - F.\nPlease enter the numbers, separated by the commas.\ne.g.:6,7,8 means mols contains C,N,O)\n').split()
        elements_list={
            6:'C',
            7:'N',
            8:'O',
            9:'F',
        }
        elements = [elements_list[int(e)] for e in element_input]

        df = pd.read_csv(self.csv)
        thr = heavy_atom_threshold.split(',')
        if len(thr) ==2:
            threshold1 = int(heavy_atom_threshold.split(',')[0])
            threshold2 = int(heavy_atom_threshold.split(',')[1])
            selected_df = df[(df['HeavyAtomCount'] > threshold1) & (df['HeavyAtomCount'] < threshold2)]
        else:
            selected_df = df[(df['HeavyAtomCount'] < int(heavy_atom_threshold))]

        self.mol_list = []
        for _, row in selected_df.iterrows():

            molecule = Chem.MolFromSmiles(row['Smiles_rdkit']) if Chem.MolFromSmiles(row['Smiles_rdkit']) else Chem.MolFromInchi(row['InchI_pybel'])

            if molecule is None:
                continue

            atom_elements = set(atom.GetSymbol() for atom in molecule.GetAtoms())

            if atom_elements.issubset(set(elements)):
                self.mol_list.append(row['Index'])
        return self.mol_list

if __name__ == '__main__':
    extractor = extractData(mol_list=[])
    data_dict = extractor.read_from_hdf5()


