import os
import pandas as pd
import ast
from rdkit import Chem
from openai import OpenAI
from get_mol_prop import compute_properties_with_affinity

def generate_molecules_for_protein(protein, input_csv, output_csv, api_key, model_engine, gnina_path, config_path, temp_dir, affinity_range, max_iterations=10, max_samples=5):
    """
    Generates molecules similar to positive labeled ones for a specific protein target while ensuring they meet a specified CNNaffinity range.

    Args:
        protein (str): Target protein name.
        input_csv (str): Path to the input CSV with 'SMILES' and 'Label'.
        output_csv (str): Path to save generated molecules.
        api_key (str): OpenAI API key.
        model_engine (str): OpenAI model engine (e.g., gpt-4).
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
    positive_molecules = data[data['Label'] == 1]['SMILES'].tolist()

    if not positive_molecules:
        print(f"No positive molecules found for {protein} in {input_csv}")
        return

    generated_molecules = []

    for iteration in range(1, max_iterations + 1):
        messages = [
            {"role": "system", "content": "You are a science expert specialising in chemistry and drug design. Your final answer should be valid SMILES strings for molecules. Do not generate any extra text descriptions."},
            {"role": "user", "content": (
                f"Generate up to {max_samples} novel valid molecules similar to the following positive molecules: {positive_molecules}.\n"
                f"Ensure the molecules are chemically feasible and suitable for binding to {protein}."
            )}
        ]

        try:
            response = client.chat.completions.create(
                model=model_engine,
                messages=messages,
                max_tokens=100 * max_samples,
                temperature=0.7,
                n=1
            )
            raw_generated_smiles = response.choices[0].message.content.strip()
            print(raw_generated_smiles[0])
            print(type(raw_generated_smiles[0]))
            print(f"Iteration {iteration}: {raw_generated_smiles}")
            # Parse and validate SMILES
            try:
                parsed_smiles_list = ast.literal_eval(raw_generated_smiles.replace("'", '"'))
                valid_smiles = [s for s in parsed_smiles_list if Chem.MolFromSmiles(s)]
                print(f"Iteration {iteration}: Valid molecules: {valid_smiles}")
                generated_molecules.extend(valid_smiles)
            except Exception as e:
                print(f"Error parsing generated SMILES: {e}")
                continue
        except Exception as e:
            print(f"Error during molecule generation: {e}")
            continue

    # Remove duplicates
    generated_molecules = list(set(generated_molecules))
    print(f"Total unique valid molecules generated: {len(generated_molecules)}")

    # Compute properties and filter based on CNNaffinity
    properties_df = compute_properties_with_affinity(
        input_data=generated_molecules,
        gnina_path=gnina_path,
        config_path=config_path,
        temp_dir=temp_dir
    )
    
    valid_df = properties_df[(properties_df['CNNaffinity'] >= affinity_range[0]) &
                             (properties_df['CNNaffinity'] <= affinity_range[1])]
    valid_molecules = valid_df['SMILES'].tolist()

    print(f"Number of valid molecules within affinity range {affinity_range}: {len(valid_molecules)}")

    # Save the valid molecules to the output CSV
    valid_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    # User parameters
    protein = "DBH"
    input_csv = f"data/{protein}.txt"
    output_csv = f"results/gen_{protein}.csv"

    # OpenAI API details
    api_key = "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"
    model_engine = "gpt-3.5-turbo" #"gpt-4o-mini"

    # Gnina docking details
    gnina_path = "./docking"
    config_path = f"./docking/{protein}/{protein}_config.txt"
    temp_dir = "/tmp/molecule_generation"
    os.makedirs(temp_dir, exist_ok=True)

    # CNNaffinity range (user-defined hypothesis)
    affinity_range = (5.0, 10.0)

    generate_molecules_for_protein(
        protein=protein,
        input_csv=input_csv,
        output_csv=output_csv,
        api_key=api_key,
        model_engine=model_engine,
        gnina_path=gnina_path,
        config_path=config_path,
        temp_dir=temp_dir,
        affinity_range=affinity_range,
        max_iterations=5,
        max_samples=5
    )


