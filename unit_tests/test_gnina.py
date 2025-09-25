import os
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
from get_mol_prop import calculate_binding_affinity

#if __name__ == "__main__":
#    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
#
#    gnina_path = "/home/server2/dash/LMLFStar/docking"        
#    config_path = "/home/server2/dash/LMLFStar/docking/2Z65/2Z65_config.txt"
#    temp_dir = "/tmp"
#
#    affinity, cnn_affinity = calculate_binding_affinity(
#        aspirin_smiles, gnina_path, config_path, temp_dir
#    )
#
#    print("Affinity:", affinity)
#    print("CNN Affinity:", cnn_affinity)

import get_mol_prop
#get_mol_prop.main(input_csv="data/2Z65.txt", gnina_path="./docking/", temp_dir="/tmp/", protein="2Z65")
get_mol_prop.main(input_csv="data/chembl1K.txt", gnina_path="./docking/", temp_dir="/tmp/", protein="2Z65")
