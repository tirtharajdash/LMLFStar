from datetime import datetime
import os
import pandas as pd
import ast
from rdkit import Chem
from openai import OpenAI
from get_mol_prop import compute_properties_with_affinity
from mol_utils import calculate_similarity, sanitize_smiles

def generate_molecules_for_protein(protein, input_csv, output_dir, api_key, model_engine, gnina_path, config_path, temp_dir, affinity_range, target_size=5, max_iterations=10, max_samples=5):
    """
    Generates molecules similar to positive labeled ones for a specific protein target while ensuring they meet a specified CNNaffinity range.

    Args:
        protein (str): Target protein name.
        input_csv (str): Path to the input CSV with 'SMILES' and 'Label'.
        output_dir (str): Path to save generated molecules and other relevant files.
        api_key (str): OpenAI API key.
        model_engine (str): OpenAI model engine (gpt-4o, gpt-4o-mini, gpt-3.5-turbo).
        gnina_path (str): Path to the Gnina executable.
        config_path (str): Path to the Gnina configuration file.
        temp_dir (str): Temporary directory for intermediate files.
        affinity_range (tuple): The user-defined CNNaffinity range (min, max).
        max_iterations (int): Maximum number of iterations for generation.
        max_samples (int): Maximum number of samples to generate per iteration.

    Returns:
        None
    """
    client = OpenAI(api_key=api_key)

    data = pd.read_csv(input_csv)
    target_positive_molecules = data[data['Label'] == 1]['SMILES'].tolist()

    positive_molecules = target_positive_molecules[0:target_size]

    if not positive_molecules:
        print(f"No positive molecules found for {protein} in {input_csv}")
        return

    generated_molecules = []

    for iteration in range(1, max_iterations + 1):
        messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are a scientist specialising in chemistry and drug design. "
                        "Your task is to generate valid SMILES strings in the form of a Python list. " 
                        "The response must be formatted exactly as follows: ['SMILES1', 'SMILES2', ...]. Avoid any extra text or explanations."
                        )
                },
                {
                    "role": "user", 
                    "content": (
                        f"Generate up to {max_samples} novel valid molecules similar to the following positive molecules: {positive_molecules}. "
                        f"Ensure the molecules are chemically feasible and suitable for binding to {protein}."
                        )
                }
            ]

        try:
            response = client.chat.completions.create(
                model=model_engine,
                messages=messages,
                max_tokens=128 * max_samples,
                temperature=0.7,
                n=1
            )
            
            raw_generated_smiles = response.choices[0].message.content.strip()
            
            print(f"Iteration {iteration}:")
            print(f"\tGenerated: {raw_generated_smiles}")
            try:
                parsed_smiles_list = ast.literal_eval(raw_generated_smiles.replace("'", '"'))
                valid_smiles = [s for s in parsed_smiles_list if sanitize_smiles(s)]
                print(f"\tValid: {valid_smiles} (Total: {len(valid_smiles)})")
                generated_molecules.extend(valid_smiles)
            except Exception as e:
                print(f"Error parsing generated SMILES: {e}")
                continue
        except Exception as e:
            print(f"Error during molecule generation: {e}")
            continue

    generated_molecules = list(set(generated_molecules))

    if len(generated_molecules) > 0:
        print(f"Total unique valid molecules generated: {len(generated_molecules)}")
        properties_df = compute_properties_with_affinity(
            input_data=generated_molecules,
            gnina_path=gnina_path,
            config_path=config_path,
            temp_dir=temp_dir
            )

        feasible_df = properties_df[(properties_df['CNNaffinity'] >= affinity_range[0]) &
                                    (properties_df['CNNaffinity'] <= affinity_range[1])]
        
        if len(feasible_df) > 0: 
            feasible_molecules = feasible_df['SMILES'].tolist()
        
            print(f"Number of feasible molecules within affinity range {affinity_range}: {len(feasible_molecules)}")
        
            print(f"Calculate Jaccard (Tanimoto) similarities between generated molecules and a target database of molecules in {input_csv}.")
            sim_df = calculate_similarity(target_positive_molecules, feasible_molecules)
        
            #save results
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Directory created: {output_dir}")
            else:
                print(f"Directory already exists: {output_dir}")
        
            gen_csv = f"{output_dir}/generated.csv"
            sim_csv = f"{output_dir}/tanimoto.csv"
            feasible_df.to_csv(gen_csv, index=False)
            sim_df.to_csv(sim_csv, index=False)
            print(f"Results saved inside {output_dir}.")
        else:
            print("Error: No valid molecule generated.")
    else:
        print("Error: No valid molecule generated.")


if __name__ == "__main__":
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    print(date_time)

    api_key = "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"
    model_engine = "gpt-3.5-turbo" #"gpt-3.5-turbo", gpt-4o-mini, gpt-4o
    
    # User parameters: protein and its affinity range for potential inhibitors (obtained from search.py)
    #protein, affinity_range = "JAK2", (6.95, 10.0)
    #protein, affinity_range = "DBH", (8.96, 10.0)
    protein, affinity_range = "DRD2", (5.76, 10.0)
    
    input_csv = f"data/{protein}.txt"
    output_dir = f"results/{protein}/{model_engine}/{date_time}"

    # Gnina docking information
    gnina_path = "./docking"
    config_path = f"./docking/{protein}/{protein}_config.txt"
    temp_dir = "/tmp/molecule_generation"
    os.makedirs(temp_dir, exist_ok=True)

    generate_molecules_for_protein(
        protein=protein,
        input_csv=input_csv,
        output_dir=output_dir,
        api_key=api_key,
        model_engine=model_engine,
        gnina_path=gnina_path,
        config_path=config_path,
        temp_dir=temp_dir,
        affinity_range=affinity_range,
        target_size=10,
        max_iterations=10,
        max_samples=5
    )

