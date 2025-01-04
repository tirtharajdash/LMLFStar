import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import rdMolDescriptors as rdmd
import numpy as np
import os
import sys
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def calculate_radscore(smiles):
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
            radscore = calculate_radscore(smiles)
            return {
                'SMILES': smiles,
                'Molecular Weight': mw,
                'LogP': logp,
                'QED Score': qed,
                'RAD Score': radscore
            }
        else:
            return {
                'SMILES': smiles,
                'Molecular Weight': None,
                'LogP': None,
                'QED Score': None,
                'RAD Score': None
            }
    except Exception as e:
        return {
            'SMILES': smiles,
            'Molecular Weight': None,
            'LogP': None,
            'QED Score': None,
            'RAD Score': None,
            'Error': str(e)
        }

def compute_properties_from_smiles(input_data):
    """
    Compute molecular properties for SMILES strings.

    Args:
        input_data: str or list. A file path to a CSV with a 'SMILES' column or a list of SMILES strings.

    Returns:
        pd.DataFrame: A DataFrame containing the computed properties for each SMILES.
    """
    if isinstance(input_data, str):
        # Assume it's a CSV file
        df = pd.read_csv(input_data)
        smiles_list = df['SMILES'].tolist()
    elif isinstance(input_data, list):
        smiles_list = input_data
    else:
        raise ValueError("Input must be a path to a CSV file or a list of SMILES strings.")

    properties = [calculate_molecular_properties(smiles) for smiles in smiles_list]
    return pd.DataFrame(properties)

# Example usage:
# For a list of SMILES strings
smiles_list = ['CCO', 'CCCC', 'CCN(CC)CC','C1=CC=CC=C1']
properties_df = compute_properties_from_smiles(smiles_list)
print(properties_df)

# For a CSV file containing SMILES strings
# properties_df = compute_properties_from_smiles('path_to_smiles_file.csv')
# print(properties_df)

