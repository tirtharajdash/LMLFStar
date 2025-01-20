import time
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
                'MolWt': mw,
                'LogP': logp,
                'QED': qed,
                'SAS': sas,
                'TPSA': tpsa,
                'H_Acceptors': h_acceptors,
                'H_Donors': h_donors,
                'Rotatable_Bonds': rot_bonds
            }
        else:
            return {
                'SMILES': smiles,
                'MolWt': None,
                'LogP': None,
                'QED': None,
                'SAS': None,
                'TPSA': None,
                'H_Acceptors': None,
                'H_Donors': None,
                'Rotatable_Bonds': None
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


def parse_gnina_output(output):
    """
    Parses the Gnina stdout to extract affinity and CNN affinity values.

    Args:
        output (str): The stdout string from Gnina.

    Returns:
        tuple: (affinity, cnn_affinity) or (None, None) if not found.
    """
    lines = output.splitlines()
    for i, line in enumerate(lines):
        # Look for the header line indicating the docking results table
        if "affinity" in line.lower() and "cnn" in line.lower():
            for result_line in lines[i + 3:]:
                result_line = result_line.strip()
                if result_line:  
                    print(f"gnina output result line captured: {result_line}")
                    parts = result_line.split()
                    if len(parts) >= 5: # usually 5 columns
                        try:
                            affinity = float(parts[1])  # Affinity value
                            cnn_affinity = float(parts[4])  # CNN Affinity value
                            return affinity, cnn_affinity
                        except ValueError:
                            print(f"Error parsing values from line: {result_line}")
                            return None, None
    return None, None


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
        affinity, cnn_affinity = parse_gnina_output(result.stdout)
        return affinity, cnn_affinity
    except Exception as e:
        print(f"Error during Gnina processing: {e}")
        return None, None


def compute_properties_with_affinity(input_data, gnina_path, config_path, temp_dir):
    """
    Compute molecular properties for SMILES strings, including binding affinity.

    Args:
        input_data: str or list. A file path to a CSV with 'SMILES' column (and optional 'Label') or a list of SMILES strings.
        gnina_path: str. Path to the Gnina executable.
        config_path: str. Path to the Gnina configuration file.
        temp_dir: str. Temporary directory to store intermediate files.

    Returns:
        pd.DataFrame: A DataFrame containing the computed properties for each SMILES.
    """
    if isinstance(input_data, str):
        #df = pd.read_csv(input_data, nrows=2)  
        df = pd.read_csv(input_data)  
        smiles_list = df['SMILES'].tolist()
        if 'Label' in df.columns:
            labels = df['Label'].tolist()
        else:
            labels = [-1] * len(smiles_list)  # unknown label (-1)
    elif isinstance(input_data, list):
        smiles_list = input_data
        labels = [-1] * len(smiles_list)
    else:
        raise ValueError("Input must be a path to a CSV file or a list of SMILES strings.")

    properties = []

    for smiles, label in zip(smiles_list, labels):
        if label == -1:
            print(f"\nPROCESSING MOLECULE: {smiles}")
        else:
            print(f"\nPROCESSING MOLECULE: {smiles} (Label: {label})")
        mol_props = calculate_molecular_properties(smiles)
        affinity, cnn_affinity = calculate_binding_affinity(smiles, gnina_path, config_path, temp_dir)
        mol_props.update({
            'Affinity': affinity,
            'CNNaffinity': cnn_affinity,
            'Label': label
        })
        properties.append(mol_props)

    return pd.DataFrame(properties)


def infer_protein_from_filename(filename):
    """
    Infer protein from the filename if it is labeled; otherwise, return None.
    """
    known_proteins = ["JAK2", "DRD2", "DBH"]
    for protein in known_proteins:
        if protein in filename:
            return protein
    return None


def main(input_csv, gnina_path, temp_dir, protein=None):
    """
    Main function to compute molecular properties and binding affinity.
    """
    if protein is None:
        protein = infer_protein_from_filename(input_csv)
        if protein is None:
            protein = "DBH"  # Default protein if not inferred
            print(f"Protein not inferred from filename. Using default protein: {protein}")
        else:
            print(f"Protein inferred from filename: {protein}")

    config_path = f'./docking/{protein}/{protein}_config.txt'
    output_csv = f"{input_csv.split('.')[0]}_with_properties_{protein}binding.txt"
    os.makedirs(temp_dir, exist_ok=True)

    print(f"Running for protein: {protein}")
    print(f"Input file: {input_csv}")
    print(f"Output file: {output_csv}")

    start_time = time.time()
    properties_df = compute_properties_with_affinity(input_csv, gnina_path, config_path, temp_dir)
    properties_df.to_csv(output_csv, index=False)
    print(properties_df.head())
    end_time = time.time()

    print(f"\nData saved to: {output_csv}")
    print(f"Total time taken: {(end_time - start_time) / 60.0:.2f} minutes")


if __name__ == "__main__":
    main(
        input_csv='data/chembl1K.txt', #JAK2.txt, DRD2.txt, DBH.txt, unlabelled.txt, chembl10.txt, chembl1K.txt
        gnina_path='./docking/',
        temp_dir='/tmp/',
        protein='JAK2' #This msut be set to a protein (JAK2/DRD2/DBH) if input_csv is non_protein
    )


