from datetime import datetime
import os
import pandas as pd
import ast
from rdkit import Chem
import anthropic
from env_utils import load_anthropic_api_key
from get_mol_prop import compute_properties_with_affinity
from mol_utils import calculate_similarity, sanitize_smiles, validate_smiles
from LMLFStar import generate_molecules_for_protein_multifactors_with_context_claude

if __name__ == "__main__":
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    print(date_time)

    api_key = load_anthropic_api_key()  # Load your Anthropic API key
    model_name = "claude-3-5-sonnet-20241022"  # or "claude-3-haiku-20240307" for faster/cheaper option

    # User parameters
    protein, affinity_range = "DBH", (6.08, 10.0)
    input_csv = f"data/{protein}.txt"

    # Gnina docking information
    gnina_path = "./docking"
    config_path = f"./docking/{protein}/{protein}_config.txt"
    temp_dir = "/tmp/molecule_generation"
    os.makedirs(temp_dir, exist_ok=True)

    print("*" * 50)
    print(f"    PROGRAM: LMLFSTAR with Claude (TIMESTAMP: {date_time})")
    print(f"    PROTEIN: {protein}")
    print("*" * 50)

    # Example usage with Claude
    output_dir = f"results_LMLFStar_Claude/MFC/{protein}/{model_name.replace('-', '_')}/{date_time}"
    
    generate_molecules_for_protein_multifactors_with_context_claude(
        protein=protein,
        input_csv=input_csv,
        output_dir=output_dir,
        api_key=api_key,
        model_name=model_name,
        gnina_path=gnina_path,
        config_path=config_path,
        temp_dir=temp_dir,
        parameter_ranges={'CNNaffinity': affinity_range, 'MolWt': (100, 500)},
        target_size=10,
        max_iterations=10,
        max_samples=5
    )
