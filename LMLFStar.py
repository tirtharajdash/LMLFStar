"""
LMLFStar.py: The classical non-interleaved version of LMLFStar.
It takes an optimal search interval (or intervals for more than one params) and does the molecule generation in iteration.
"""


from datetime import datetime
import os
import pandas as pd
import ast
from rdkit import Chem
from openai import OpenAI

from env_utils import load_api_key
from get_mol_prop import compute_properties_with_affinity
from mol_utils import calculate_similarity, sanitize_smiles, validate_smiles

def generate_molecules_for_protein(protein, input_csv, output_dir, api_key, model_engine, gnina_path, config_path, temp_dir, 
                                   affinity_range, target_size=5, max_iterations=10, max_samples=5):
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
        
    gen_prompt = f"Generate up to {max_samples} novel valid molecules"
    if target_size > 0:
        positive_molecules = target_positive_molecules[0:target_size]
        if not positive_molecules:
            print(f"No positive molecules found for {protein} in {input_csv}")
            return
        gen_prompt = f"{gen_prompt} similar to the following positive molecules: {positive_molecules}"

    generated_molecules = []

    for iteration in range(1, max_iterations + 1):
        messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are a scientist specialising in chemistry and drug design. "
                        "Your task is to generate valid SMILES strings as a comma-separated list inside square brackets. "
                        "Return the response as plain text without any formatting, backticks, or explanations. "
                        "The response must be formatted exactly as follows: ['SMILES1', 'SMILES2', ...]. Avoid any extra text or explanations. "
                        "Example output: ['SMILES1', 'SMILES2', 'SMILES3']" 
                        )
                },
                {
                    "role": "user", 
                    "content": (
                        f"{gen_prompt}. "
                        f"Ensure the molecules are chemically feasible and require minimal steps for synthesis."
                        )
                }
            ]
        
        valid_smiles = []

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
                #valid_smiles = [s for s in parsed_smiles_list if sanitize_smiles(s)]
                valid_smiles = [s for s in parsed_smiles_list if validate_smiles(s)]
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


def generate_molecules_for_protein_multifactors(protein, input_csv, output_dir, api_key, model_engine, gnina_path, config_path, temp_dir, 
                                                parameter_ranges, target_size=5, max_iterations=10, max_samples=5):
    """
    Generates molecules similar to positive labeled ones for a specific protein target while ensuring they meet specified parameter ranges.

    Args:
        protein (str): Target protein name.
        input_csv (str): Path to the input CSV with 'SMILES' and 'Label'.
        output_dir (str): Path to save generated molecules and other relevant files.
        api_key (str): OpenAI API key.
        model_engine (str): OpenAI model engine (gpt-4o, gpt-4o-mini, gpt-3.5-turbo).
        gnina_path (str): Path to the Gnina executable.
        config_path (str): Path to the Gnina configuration file.
        temp_dir (str): Temporary directory for intermediate files.
        parameter_ranges (dict): A dictionary of parameter ranges (e.g., {'CNNaffinity': (2, 10), 'MolWt': (100, 500)}).
        max_iterations (int): Maximum number of iterations for generation.
        max_samples (int): Maximum number of samples to generate per iteration.

    Returns:
        None
    """
    client = OpenAI(api_key=api_key)
    
    data = pd.read_csv(input_csv)
    target_positive_molecules = data[data['Label'] == 1]['SMILES'].tolist()
        
    gen_prompt = f"Generate up to {max_samples} novel valid molecules"
    if target_size > 0:
        positive_molecules = target_positive_molecules[0:target_size]
        if not positive_molecules:
            print(f"No positive molecules found for {protein} in {input_csv}")
            return
        gen_prompt = f"{gen_prompt} similar to the following positive molecules: {positive_molecules}"

    generated_molecules = []

    for iteration in range(1, max_iterations + 1):
        messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are a scientist specialising in chemistry and drug design. "
                        "Your task is to generate valid SMILES strings as a comma-separated list inside square brackets. "
                        "Return the response as plain text without any formatting, backticks, or explanations. "
                        "The response must be formatted exactly as follows: ['SMILES1', 'SMILES2', ...]. Avoid any extra text or explanations. "
                        "Example output: ['SMILES1', 'SMILES2', 'SMILES3']" 
                        )
                },
                {
                    "role": "user", 
                    "content": (
                        f"{gen_prompt}. "
                        f"Ensure the molecules are chemically feasible and require minimal steps for synthesis."
                        )
                }
            ]
        
        valid_smiles = []

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
                #valid_smiles = [s for s in parsed_smiles_list if sanitize_smiles(s)]
                valid_smiles = [s for s in parsed_smiles_list if validate_smiles(s)]
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

        for param, (min_val, max_val) in parameter_ranges.items():
            properties_df = properties_df[(properties_df[param] >= min_val) &
                                          (properties_df[param] <= max_val)]
        
        if len(properties_df) > 0: 
            feasible_molecules = properties_df['SMILES'].tolist()
        
            print(f"Number of feasible molecules within specified ranges: {len(feasible_molecules)}")
        
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
            properties_df.to_csv(gen_csv, index=False)
            sim_df.to_csv(sim_csv, index=False)
            print(f"Results saved inside {output_dir}.")
        else:
            print("Error: No valid molecule generated within specified ranges.")
    else:
        print("Error: No valid molecule generated.")


def generate_molecules_for_protein_with_context(protein, input_csv, output_dir, api_key, model_engine, gnina_path, config_path, temp_dir, 
                                                affinity_range, target_size=5, max_iterations=10, max_samples=5):
    """
    Generates molecules similar to positive labeled ones for a specific protein target while ensuring they
    meet a specified CNNaffinity range. In each iteration, the molecules that pass the feasibility constraint
    are added as context for subsequent generations.

    Args:
        protein (str): Target protein name.
        input_csv (str): Path to the input CSV with 'SMILES' and 'Label'.
        output_dir (str): Path to save generated molecules and other relevant files.
        api_key (str): OpenAI API key.
        model_engine (str): OpenAI model engine (e.g., 'gpt-4o', 'gpt-3.5-turbo').
        gnina_path (str): Path to the Gnina executable.
        config_path (str): Path to the Gnina configuration file.
        temp_dir (str): Temporary directory for intermediate files.
        affinity_range (tuple): The CNNaffinity range as (min, max).
        target_size (int): Number of positive molecules to use as a starting point.
        max_iterations (int): Maximum number of iterations for generation.
        max_samples (int): Maximum number of samples to generate per iteration.

    Returns:
        None
    """
    client = OpenAI(api_key=api_key)
    
    data = pd.read_csv(input_csv)
    target_positive_molecules = data[data['Label'] == 1]['SMILES'].tolist()
        
    gen_prompt = f"Generate up to {max_samples} novel valid molecules"
    if target_size > 0:
        positive_molecules = target_positive_molecules[0:target_size]
        if not positive_molecules:
            print(f"No positive molecules found for {protein} in {input_csv}")
            return
        gen_prompt = f"{gen_prompt} similar to the following positive molecules: {positive_molecules}"

    generated_molecules = []
    context_feasible = [] 

    for iteration in range(1, max_iterations + 1):
        context_text = f" Additionally, consider these previously generated feasible molecules: {context_feasible}." if context_feasible else ""
               
        messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are a scientist specialising in chemistry and drug design. "
                        "Your task is to generate valid SMILES strings as a comma-separated list inside square brackets. "
                        "Return the response as plain text without any formatting, backticks, or explanations. "
                        "The response must be formatted exactly as follows: ['SMILES1', 'SMILES2', ...]. Avoid any extra text or explanations. "
                        "Example output: ['SMILES1', 'SMILES2', 'SMILES3']" 
                        )
                },
                {
                    "role": "user", 
                    "content": (
                        f"{gen_prompt}. "
                        f"{context_text} "
                        f"Ensure the molecules are chemically feasible and require minimal steps for synthesis."
                        )
                }
            ]
        
        valid_smiles = []  

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
                # Parse the generated SMILES list.
                parsed_smiles_list = ast.literal_eval(raw_generated_smiles.replace("'", '"'))
                #valid_smiles = [s for s in parsed_smiles_list if sanitize_smiles(s)]
                valid_smiles = [s for s in parsed_smiles_list if validate_smiles(s)]
                print(f"\tValid molecules: {valid_smiles} (Total: {len(valid_smiles)})")
                generated_molecules.extend(valid_smiles)
            except Exception as e:
                print(f"Error parsing generated SMILES: {e}")
                continue
        except Exception as e:
            print(f"Error during molecule generation: {e}")
            continue

        # Remove duplicates.
        generated_molecules = list(set(generated_molecules))

        # Evaluate newly generated molecules for feasibility.
        if valid_smiles:
            properties_df = compute_properties_with_affinity(
                input_data=valid_smiles,
                gnina_path=gnina_path,
                config_path=config_path,
                temp_dir=temp_dir
            )
            feasible_df = properties_df[
                (properties_df['CNNaffinity'] >= affinity_range[0]) &
                (properties_df['CNNaffinity'] <= affinity_range[1])
            ]
            if not feasible_df.empty:
                new_feasible = feasible_df['SMILES'].tolist()
                print(f"\tFeasible molecules in iteration {iteration}: {new_feasible}")
                context_feasible.extend(new_feasible)
                context_feasible = list(set(context_feasible))
            else:
                print(f"\tNo feasible molecules found in iteration {iteration}.")

    # Final processing.
    if generated_molecules:
        print(f"Total unique generated molecules: {len(generated_molecules)}")
        properties_df = compute_properties_with_affinity(
            input_data=generated_molecules,
            gnina_path=gnina_path,
            config_path=config_path,
            temp_dir=temp_dir
        )
        feasible_df = properties_df[
            (properties_df['CNNaffinity'] >= affinity_range[0]) &
            (properties_df['CNNaffinity'] <= affinity_range[1])
        ]
        if not feasible_df.empty:
            feasible_molecules = feasible_df['SMILES'].tolist()
            print(f"Number of feasible molecules within affinity range {affinity_range}: {len(feasible_molecules)}")
            sim_df = calculate_similarity(target_positive_molecules, feasible_molecules)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Directory created: {output_dir}")
            else:
                print(f"Directory already exists: {output_dir}")

            gen_csv = os.path.join(output_dir, "generated.csv")
            sim_csv = os.path.join(output_dir, "tanimoto.csv")
            feasible_df.to_csv(gen_csv, index=False)
            sim_df.to_csv(sim_csv, index=False)
            print(f"Results saved in {output_dir}.")
        else:
            print("Error: No feasible molecules generated within the specified CNNaffinity range.")
    else:
        print("Error: No valid molecules generated.")


def generate_molecules_for_protein_multifactors_with_context(protein, input_csv, output_dir, api_key, model_engine, 
                                                             gnina_path, config_path, temp_dir, parameter_ranges, 
                                                             target_size=5, max_iterations=10, max_samples=5):
    """
    Generates molecules similar to positive labeled ones for a specific protein target while ensuring they
    meet specified parameter ranges. In each iteration, the molecules that pass the feasibility constraints
    are added as context for subsequent generations.

    Args:
        protein (str): Target protein name.
        input_csv (str): Path to the input CSV with 'SMILES' and 'Label'.
        output_dir (str): Path to save generated molecules and other relevant files.
        api_key (str): OpenAI API key.
        model_engine (str): OpenAI model engine (e.g., 'gpt-4o', 'gpt-3.5-turbo').
        gnina_path (str): Path to the Gnina executable.
        config_path (str): Path to the Gnina configuration file.
        temp_dir (str): Temporary directory for intermediate files.
        parameter_ranges (dict): A dictionary of parameter ranges (e.g., {'CNNaffinity': (2, 10), 'MolWt': (100, 500)}).
        target_size (int): Number of positive molecules to use as a starting point.
        max_iterations (int): Maximum number of iterations for generation.
        max_samples (int): Maximum number of samples to generate per iteration.

    Returns:
        None
    """
    client = OpenAI(api_key=api_key)
    
    data = pd.read_csv(input_csv)
    target_positive_molecules = data[data['Label'] == 1]['SMILES'].tolist()
        
    gen_prompt = f"Generate up to {max_samples} novel valid molecules"
    if target_size > 0:
        positive_molecules = target_positive_molecules[0:target_size]
        if not positive_molecules:
            print(f"No positive molecules found for {protein} in {input_csv}")
            return
        gen_prompt = f"{gen_prompt} similar to the following positive molecules: {positive_molecules}"

    generated_molecules = []
    context_feasible = []

    for iteration in range(1, max_iterations + 1):
        context_text = f" Additionally, consider these previously generated feasible molecules: {context_feasible}." if context_feasible else ""
        
        messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are a scientist specialising in chemistry and drug design. "
                        "Your task is to generate valid SMILES strings as a comma-separated list inside square brackets. "
                        "Your generated SMILES strings must not be available in any known chemical databases such as ChEMBL, PubChem, ZINC, etc. " #added on 27032025
                        "Return the response as plain text without any formatting, backticks, or explanations. "
                        "The response must be formatted exactly as follows: ['SMILES1', 'SMILES2', ...]. Avoid any extra text or explanations. "
                        "Example output: ['SMILES1', 'SMILES2', 'SMILES3']" 
                        )
                },
                {
                    "role": "user", 
                    "content": (
                        f"{gen_prompt}. "
                        f"{context_text} "
                        f"Ensure the molecules are chemically feasible and require minimal steps for synthesis."
                        )
                }
            ]
        
        valid_smiles = []  

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
                #valid_smiles = [s for s in parsed_smiles_list if sanitize_smiles(s)]
                valid_smiles = [s for s in parsed_smiles_list if validate_smiles(s)]
                print(f"\tValid molecules: {valid_smiles} (Total: {len(valid_smiles)})")
                generated_molecules.extend(valid_smiles)
            except Exception as e:
                print(f"Error parsing generated SMILES: {e}")
                continue
        except Exception as e:
            print(f"Error during molecule generation: {e}")
            continue

        # Remove duplicates.
        generated_molecules = list(set(generated_molecules))

        # Evaluate the newly generated molecules for feasibility.
        if valid_smiles:
            properties_df = compute_properties_with_affinity(
                input_data=valid_smiles,
                gnina_path=gnina_path,
                config_path=config_path,
                temp_dir=temp_dir
            )

            # Apply each parameter filter sequentially.
            for param, (min_val, max_val) in parameter_ranges.items():
                properties_df = properties_df[(properties_df[param] >= min_val) & (properties_df[param] <= max_val)]

            if not properties_df.empty:
                new_feasible = properties_df['SMILES'].tolist()
                print(f"\tFeasible molecules in iteration {iteration}: {new_feasible}")
                context_feasible.extend(new_feasible)
                context_feasible = list(set(context_feasible))
            else:
                print(f"\tNo feasible molecules found in iteration {iteration}.")

    # Final processing.
    if generated_molecules:
        print(f"Total unique generated molecules: {len(generated_molecules)}")
        properties_df = compute_properties_with_affinity(
            input_data=generated_molecules,
            gnina_path=gnina_path,
            config_path=config_path,
            temp_dir=temp_dir
        )

        for param, (min_val, max_val) in parameter_ranges.items():
            properties_df = properties_df[(properties_df[param] >= min_val) & (properties_df[param] <= max_val)]

        if not properties_df.empty:
            feasible_molecules = properties_df['SMILES'].tolist()
            print(f"Number of feasible molecules meeting specified ranges: {len(feasible_molecules)}")
            sim_df = calculate_similarity(target_positive_molecules, feasible_molecules)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Directory created: {output_dir}")
            else:
                print(f"Directory already exists: {output_dir}")

            gen_csv = os.path.join(output_dir, "generated.csv")
            sim_csv = os.path.join(output_dir, "tanimoto.csv")
            properties_df.to_csv(gen_csv, index=False)
            sim_df.to_csv(sim_csv, index=False)
            print(f"Results saved in {output_dir}.")
        else:
            print("Error: No feasible molecules generated within the specified parameter ranges.")
    else:
        print("Error: No valid molecules generated.")


if __name__ == "__main__":
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    print(date_time)

    api_key = load_api_key()  # Load the API key without revealing it publicly
    model_engine = "gpt-4o"  # "gpt-3.5-turbo", gpt-4o-mini, gpt-4o

    # User parameters: protein and its affinity range for potential inhibitors (obtained from search.py)
    # protein, affinity_range = "JAK2", (6.95, 10.0)
    # protein, affinity_range = "DRD2", (6.95, 10.0)
    protein, affinity_range = "DBH", (6.08, 10.0)
    input_csv = f"data/{protein}.txt"

    # Gnina docking information
    gnina_path = "./docking"
    config_path = f"./docking/{protein}/{protein}_config.txt"
    temp_dir = "/tmp/molecule_generation"
    os.makedirs(temp_dir, exist_ok=True)

    print("*" * 50)
    print(f"    PROGRAM: LMLFSTAR (TIMESTAMP: {date_time})")
    print(f"    PROTEIN: {protein}")
    print("*" * 50)

    # Let the user choose which generation mode to test
    print("Choose the generation mode:")
    print("1. Single factor (SF)")
    print("2. Multifactor (MF)")
    print("3. Single factor with context (SFC)")
    print("4. Multifactor with context (MFC)")
    print("0. Abort")
    mode = input("Enter the mode number (1-4): ").strip()

    # Build the output directory based on the chosen mode
    if mode == "1":
        subpath = "SF"
        output_dir = f"results_LMLFStar/{subpath}/{protein}/{model_engine}/{date_time}"
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
    elif mode == "2":
        subpath = "MF"
        output_dir = f"results_LMLFStar/{subpath}/{protein}/{model_engine}/{date_time}"
        generate_molecules_for_protein_multifactors(
            protein=protein,
            input_csv=input_csv,
            output_dir=output_dir,
            api_key=api_key,
            model_engine=model_engine,
            gnina_path=gnina_path,
            config_path=config_path,
            temp_dir=temp_dir,
            parameter_ranges={'CNNaffinity': affinity_range, 'MolWt': (100, 500)},
            target_size=10,
            max_iterations=10,
            max_samples=5
        )
    elif mode == "3":
        subpath = "SFC"
        output_dir = f"results_LMLFStar/{subpath}/{protein}/{model_engine}/{date_time}"
        generate_molecules_for_protein_with_context(
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
    elif mode == "4":
        subpath = "MFC"
        output_dir = f"results_LMLFStar/{subpath}/{protein}/{model_engine}/{date_time}"
        generate_molecules_for_protein_multifactors_with_context(
            protein=protein,
            input_csv=input_csv,
            output_dir=output_dir,
            api_key=api_key,
            model_engine=model_engine,
            gnina_path=gnina_path,
            config_path=config_path,
            temp_dir=temp_dir,
            parameter_ranges={'CNNaffinity': affinity_range, 'MolWt': (100, 500)},
            target_size=10,
            max_iterations=10,
            max_samples=5
        )
    else:
        print("Invalid mode selected. Exiting.")



