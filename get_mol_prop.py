import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import rdMolDescriptors as rdmd
import numpy as np
import os
import sys
import subprocess
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def calculate_sas_score(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            sa_score = sascorer.calculateScore(mol)
            return sa_score
        return None
    except Exception:
        return None

def calculate_qed_score(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return QED.qed(mol)
        return None
    except Exception:
        return None

def calculate_molecular_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            qed = calculate_qed_score(smiles)
            sas = calculate_sas_score(smiles)
            tpsa = Descriptors.TPSA(mol)
            h_acceptors = Descriptors.NumHAcceptors(mol)
            h_donors = Descriptors.NumHDonors(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            return {
                'SMILES': smiles,
                'Molecular Weight': mw,
                'LogP': logp,
                'QED Score': qed,
                'SAS Score': sas,
                'TPSA': tpsa,
                'H Acceptors': h_acceptors,
                'H Donors': h_donors,
                'Rotatable Bonds': rot_bonds
            }
        else:
            return {
                'SMILES': smiles,
                'Molecular Weight': None,
                'LogP': None,
                'QED Score': None,
                'SAS Score': None,
                'TPSA': None,
                'H Acceptors': None,
                'H Donors': None,
                'Rotatable Bonds': None
            }
    except Exception as e:
        return {
            'SMILES': smiles,
            'Error': str(e)
        }


def smiles_to_pdb(smiles, output_path):
    """
    Converts a SMILES string to a PDB file.

    Args:
        smiles (str): The SMILES string of the molecule.
        output_path (str): The path to save the PDB file.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            print(f"Invalid SMILES: {smiles}")
            return False

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        success = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if success != 0:
            print(f"Failed to generate 3D coordinates for SMILES: {smiles}")
            return False

        # Optimize the structure
        AllChem.MMFFOptimizeMolecule(mol)

        # Save as PDB
        Chem.MolToPDBFile(mol, output_path)
        return True
    except Exception as e:
        print(f"Error during SMILES-to-PDB conversion: {e}")
        return False


def calculate_binding_affinity(smiles, gnina_path, config_path, temp_dir):
    try:
        pdb_file = os.path.join(temp_dir, 'ligand.pdb')

        # Convert SMILES to PDB
        success = smiles_to_pdb(smiles, pdb_file)
        if not success:
            return None, None

        # Run Gnina
        cmd = [
            os.path.join(gnina_path, 'gnina'),
            '--config', config_path,
            '--ligand', pdb_file,
            '--seed', '0',
            '--cpu', '64'
        ]

        print(f"Running Gnina command: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Gnina output:\n{result.stdout}")

        if result.returncode != 0:
            print(f"Gnina error:\n{result.stderr}")
            return None, None

        # Parse the output for affinity and CNN affinity
        affinity = None
        cnn_affinity = None
        for line in result.stdout.splitlines():
            if "mode" in line or not line.strip():
                continue
            if "affinity" in line and "CNN" in line:
                parts = line.split()
                if len(parts) >= 4:
                    print(parts)
                    try:
                        affinity = float(parts[1])
                        cnn_affinity = float(parts[-1])
                    except ValueError:
                        print(f"Error parsing affinity values in line: {line}")
                break

        return affinity, cnn_affinity
    except Exception as e:
        print(f"Error during Gnina processing: {e}")
        return None, None


def compute_properties_with_affinity(input_data, gnina_path, config_path, temp_dir):
    """
    Compute molecular properties for SMILES strings, including binding affinity.

    Args:
        input_data: str or list. A file path to a CSV with 'SMILES' and 'label' columns or a list of SMILES strings.
        gnina_path: str. Path to the Gnina executable.
        config_path: str. Path to the Gnina configuration file.
        temp_dir: str. Temporary directory to store intermediate files.

    Returns:
        pd.DataFrame: A DataFrame containing the computed properties for each SMILES.
    """
    if isinstance(input_data, str):
        df = pd.read_csv(input_data, nrows=2)
        smiles_list = df['SMILES'].tolist()
        labels = df['Label'].tolist()
    elif isinstance(input_data, list):
        smiles_list = input_data
        labels = [None] * len(smiles_list)
    else:
        raise ValueError("Input must be a path to a CSV file or a list of SMILES strings.")

    properties = []

    for smiles, label in zip(smiles_list, labels):
        print(f"\nPROCESSING MOLECULE: {smiles} (Label: {label})")
        mol_props = calculate_molecular_properties(smiles)
        affinity, cnn_affinity = calculate_binding_affinity(smiles, gnina_path, config_path, temp_dir)
        mol_props.update({
            'Affinity': affinity,
            'CNNaffinity': cnn_affinity
            'Label': label
        })
        properties.append(mol_props)

    return pd.DataFrame(properties)


# RUN CONFIG
gnina_path = './docking/JAK2/'
config_path = './docking/JAK2/JAK2_config.txt'
temp_dir = '/tmp/'
os.makedirs(temp_dir, exist_ok=True)
input_csv = 'data/JAK2.txt'
properties_df = compute_properties_with_affinity(input_csv, gnina_path, config_path, temp_dir)
properties_df.to_csv("data/JAK2_with_properties_and_affinities.txt", index=False)

