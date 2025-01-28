
import random
import math
import numpy as np
from datetime import datetime
import os
from search import Hypothesis, compute_Q, construct_file_paths
from LMLFStar import generate_molecules_for_protein
import pandas as pd

# Seeding for reproducibility
random.seed(0)
np.random.seed(0)


def factor(x, constraints):
    """
    Checks if a molecule satisfies the specified constraints.

    Args:
        x (dict): Molecule properties.
        constraints (list): List of constraints where each constraint is a dictionary with:
                            - 'parameter': str, property name (e.g., 'CNNaffinity', 'MolWt').
                            - 'operator': str, '<', '>', or 'range'.
                            - 'value': float or list, value for comparison or range [min, max].

    Returns:
        bool: True if molecule satisfies all constraints, else False.
    """
    for constraint in constraints:
        param, operator, value = constraint['parameter'], constraint['operator'], constraint['value']
        if param not in x:
            continue  # Skip if parameter is missing

        if operator == '<':
            if not (x[param] < value):
                return False
        elif operator == '>':
            if not (x[param] > value):
                return False
        elif operator == 'range':
            if not (value[0] <= x[param] <= value[1]):
                return False
    return True


def sample_intervals(ranges, num_quantiles, num_samples):
    """
    Samples intervals for each parameter based on quantiles.

    Args:
        ranges (dict): Dictionary of current ranges for each parameter (e.g., {'CNNaffinity': [2, 10]}).
        num_quantiles (int): Number of quantiles for splitting the range.
        num_samples (int): Number of intervals to sample per quantile.

    Returns:
        list: List of sampled intervals for all parameters.
    """
    sampled_intervals = []
    for param, (lower, upper) in ranges.items():
        quantiles = list(np.linspace(lower, upper, num_quantiles + 1))
        param_intervals = [
            {param: [random.uniform(quantiles[q], quantiles[q + 1]), upper] if lower == 0
             else [lower, random.uniform(quantiles[q], quantiles[q + 1])]}
            for q in range(num_quantiles)
            for _ in range(num_samples // num_quantiles)
        ]
        sampled_intervals.extend(param_intervals)
    return sampled_intervals


def interleaved_LMLFStar(protein, labelled_data, unlabelled_data, initial_ranges, constraints, api_key, model_engine,
                         gnina_path, config_path, temp_dir, output_dir, s=4, n=10, max_samples=5, final_k=10):
    """
    Interleaved search and molecule generation with multiple parameters.

    Args:
        protein (str): Target protein name.
        labelled_data (list): Labelled dataset (list of dictionaries).
        unlabelled_data (list): Unlabelled dataset (list of dictionaries).
        initial_ranges (dict): Initial ranges for parameters (e.g., {'CNNaffinity': [2, 10]}).
        constraints (list): Fixed constraints for filtering feasible molecules.
        api_key (str): OpenAI API key.
        model_engine (str): Model engine (e.g., gpt-3.5-turbo).
        gnina_path (str): Path to Gnina executable.
        config_path (str): Path to Gnina config file.
        temp_dir (str): Path for temporary files.
        output_dir (str): Directory to save results.
        s (int): Number of samples per interval.
        n (int): Maximum number of iterations.
        max_samples (int): Maximum number of molecules to generate per iteration.
        final_k (int): Number of molecules to generate at the final node.

    Returns:
        None
    """
    e_0 = initial_ranges.copy()  # Current search space
    k = 1
    interval_history = [e_0]
    search_tree = []
    Q_values = []

    intermediate_csv = os.path.join(output_dir, "intermediate.csv")
    intermediate_data = []

    while k <= n:
        print(f"\nIteration {k}: Current Ranges {e_0}")

        # Sample intervals for each parameter
        E_k = sample_intervals(e_0, num_quantiles=4, num_samples=s)

        S = []
        for sampled_interval in E_k:
            h_k = Hypothesis([lambda x, param=param: x.get(param, None) for param in sampled_interval.keys()],
                             [[sampled_interval[param][0], sampled_interval[param][1]] for param in sampled_interval.keys()])
            Q_k = compute_Q(h_k, "Background Knowledge", labelled_data)
            S.append((Q_k, sampled_interval))

        # Add to search tree
        search_tree.append({
            "iteration": k,
            "current_ranges": e_0,
            "children": [{"interval": sampled_interval, "Q_score": Q_k} for Q_k, sampled_interval in S]
        })

        # Sort by Q-scores
        sorted_S = sorted(S, key=lambda x: x[0], reverse=True)
        feasible_node_found = False

        for (Q_k, interval) in sorted_S:
            print(f"Evaluating interval {interval} with Q-score {Q_k:.4f}")

            # Generate molecules for the sampled interval
            generate_molecules_for_protein(
                protein=protein,
                input_csv=f"data/{protein}.txt",
                output_dir=output_dir,
                api_key=api_key,
                model_engine=model_engine,
                gnina_path=gnina_path,
                config_path=config_path,
                temp_dir=temp_dir,
                affinity_range=interval.get('CNNaffinity', [0, 10]),
                target_size=5,
                max_iterations=1,
                max_samples=max_samples
            )

            # Check feasibility of molecules
            gen_csv = os.path.join(output_dir, "generated.csv")
            if os.path.exists(gen_csv):
                properties_df = pd.read_csv(gen_csv)
                feasible_df = properties_df[
                    properties_df.apply(lambda row: factor(row, constraints), axis=1)
                ]

                if len(feasible_df) > 0:
                    print(f"Feasible molecules found in interval {interval}")
                    w_k = Q_k * len(feasible_df)
                    Q_values.append(w_k)

                    # Shrink search space for next iteration
                    for param in interval:
                        e_0[param] = interval[param]

                    interval_history.append(e_0)
                    intermediate_data.extend(feasible_df.to_dict(orient="records"))
                    feasible_node_found = True
                    break
            else:
                print(f"No molecules generated for interval {interval}")

        if not feasible_node_found:
            print("No feasible intervals found. Ending search.")
            break

        k += 1

    # Save intermediate results
    if intermediate_data:
        pd.DataFrame(intermediate_data).drop_duplicates().to_csv(intermediate_csv, index=False)
        print(f"Intermediate feasible molecules saved to {intermediate_csv}")

    # Generate final molecules
    print("\nGenerating final molecules for the last interval.")
    generate_molecules_for_protein(
        protein=protein,
        input_csv=f"data/{protein}.txt",
        output_dir=output_dir,
        api_key=api_key,
        model_engine=model_engine,
        gnina_path=gnina_path,
        config_path=config_path,
        temp_dir=temp_dir,
        affinity_range=e_0.get('CNNaffinity', [0, 10]),
        target_size=5,
        max_iterations=1,
        max_samples=final_k
    )

    print("\nFinal Search Space:", e_0)
    print("Search Tree:")
    for node in search_tree:
        print(f"Iteration {node['iteration']}: {node['current_ranges']}")
        for child in node['children']:
            print(f"  Child: {child['interval']} | Q-score: {child['Q_score']:.4f}")


if __name__ == "__main__":
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    protein = "DBH"
    initial_ranges = {
        'CNNaffinity': [2, 10],
        'MolWt': [0, 500],
        'SAS': [0, 10]
    }
    constraints = [
        {'parameter': 'CNNaffinity', 'operator': 'range', 'value': [2, 10]},
        {'parameter': 'MolWt', 'operator': '<', 'value': 500},
        {'parameter': 'SAS', 'operator': '>', 'value': 2.5}
    ]

    # Paths and initialization
    data_path = "data"
    labelled_file, unlabelled_file = construct_file_paths(data_path, protein)

    labelled_data = pd.read_csv(labelled_file).to_dict(orient="records")
    unlabelled_data = pd.read_csv(unlabelled_file).to_dict(orient="records")

    api_key = "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"
    model_engine = "gpt-4o-mini"
    gnina_path = "./docking"
    config_path = f"./docking/{protein}/{protein}_config.txt"
    temp_dir = "/tmp/molecule_generation"
    output_dir = f"results_i_mpmc/{protein}/{model_engine}/{date_time}"

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    interleaved_LMLFStar(
        protein=protein,
        labelled_data=labelled_data,
        unlabelled_data=unlabelled_data,
        initial_ranges=initial_ranges,
        constraints=constraints,
        api_key=api_key,
        model_engine=model_engine,
        gnina_path=gnina_path,
        config_path=config_path,
        temp_dir=temp_dir,
        output_dir=output_dir,
        s=4,
        n=10,
        max_samples=5,
        final_k=10
    )

