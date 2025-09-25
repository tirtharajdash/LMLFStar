import pandas as pd
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from get_mol_prop import compute_properties_with_affinity
from mol_utils import calculate_similarity, validate_smiles
from datetime import datetime
import os
from LMLFStar import generate_molecules_for_protein_multifactors_using_ZincGPT2


if __name__ == "__main__":
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    print(date_time)

    # Test proteins
    proteins = ["JAK2", "DRD2", "DBH"]
    affinity_ranges = {
        "JAK2": (6.95, 10.0),
        "DRD2": (6.95, 10.0), 
        "DBH": (4, 10.0)
    }

    # Gnina docking information
    gnina_path = "./docking"
    temp_dir = "/tmp/molecule_generation"
    os.makedirs(temp_dir, exist_ok=True)

    print("*" * 50)
    print(f"    PROGRAM: GPT2 Molecule Generation (TIMESTAMP: {date_time})")
    print("*" * 50)

    for protein in proteins[2:]:
        print(f"\n{'='*20} Processing {protein} {'='*20}")
        
        input_csv = f"data/{protein}.txt"
        config_path = f"./docking/{protein}/{protein}_config.txt"
        output_dir = f"results_ZincGPT2/GPT2_MF/{protein}/gpt2_zinc_87m/{date_time}"
        
        generate_molecules_for_protein_multifactors_using_ZincGPT2(
            protein=protein,
            input_csv=input_csv,
            output_dir=output_dir,
            gnina_path=gnina_path,
            config_path=config_path,
            temp_dir=temp_dir,
            parameter_ranges={'CNNaffinity': affinity_ranges[protein], 'MolWt': (100, 500)},
            target_size=5,
            max_iterations=5,
            molecules_per_scaffold=5
        )
        
        print(f"Completed processing for {protein}")

    print("\nAll proteins processed successfully!")