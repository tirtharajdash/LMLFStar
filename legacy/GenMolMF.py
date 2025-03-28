"""
GenMolMF.py: Generate molecules while doing the LMLFStar search with multiple constraint (CNNaffinity, MolWt, SAS, etc.)
The code searches for an optimal parameter ranges (CNNaffinity in [x,10], ...) while interleaving it with generating molecules from an LLM.
Feasibility of generated molecules are checked all provided factors. That is, each factor should be within the obtained optimal range from search.
"""

import random
import math
import numpy as np
from datetime import datetime
import os

from env_utils import load_api_key
from search import Hypothesis, compute_Q, construct_file_paths
from LMLFStar import generate_molecules_for_protein_multifactors, generate_molecules_for_protein_multifactors_with_context
import pandas as pd

# Seeding for reproducibility
random.seed(0)
np.random.seed(0)

def interleaved_LMLFStar(protein, labelled_data, unlabelled_data, initial_intervals, api_key, model_engine, gnina_path, config_path, temp_dir, output_dir, s=4, n=10, max_samples=5, final_k=10, context=False):
    """
    Interleaved search and molecule generation.

    Args:
        protein (str): Target protein name.
        labelled_data (list): Labelled dataset (list of dictionaries).
        unlabelled_data (list): Unlabelled dataset (list of dictionaries).
        initial_intervals (dict): Initial intervals for each parameter as a dictionary.
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
    # Set initial hypothesis
    factors = [lambda x, p=param: x.get(p) for param in initial_intervals.keys()]
    e_0 = [initial_intervals[param] for param in initial_intervals]
    h_0 = Hypothesis(factors, e_0)

    theta_ext_h_default = len(unlabelled_data) / (len(labelled_data) + len(unlabelled_data))
    w_0 = compute_Q(h_0, "Background Knowledge", labelled_data, theta_ext_h_approx=theta_ext_h_default)

    k = 1
    interval_history = [e_0]
    Q_values = [w_0]
    search_tree = []

    # Create an intermediate CSV file for saving feasible molecules
    intermediate_csv = os.path.join(output_dir, "intermediate.csv")
    intermediate_data = []

    while k <= n:
        print(f"\nIteration {k}: Current Intervals {e_0} | Q-score {w_0:.4f}\n")

        E_k = []
        for dim, (lower, upper) in enumerate(e_0):
            quantiles = list(map(float, np.linspace(lower, upper, 5)))
            for q in range(len(quantiles) - 1):
                sub_interval = e_0[:]
                sub_interval[dim] = [quantiles[q], quantiles[q + 1]]
                E_k.append(sub_interval)

        S = []
        for e in E_k:
            h_k = Hypothesis(factors, e)
            Q_k = compute_Q(h_k, "Background Knowledge", labelled_data, theta_ext_h_approx=theta_ext_h_default)
            S.append((Q_k, e))

        print(f"Sampled intervals: {E_k}")
        print(f"Q-scores: {[q for q, e in S]}\n")

        # Add current node and its children to the search tree
        search_tree.append({
            "iteration": k,
            "current_interval": e_0,
            "Q_score": w_0,
            "children": [{"interval": e, "Q_score": q} for q, e in S]
        })

        # Select child node with highest Q-score
        sorted_S = sorted(S, key=lambda x: x[0], reverse=True)
        feasible_node_found = False

        for (Q_k, e_k) in sorted_S:
            print(f"Evaluating node with interval {e_k} and Q-score {Q_k:.4f}")

            parameter_ranges = {param: e_k[i] for i, param in enumerate(initial_intervals.keys())}

            # Generate molecules for the hyper-interval
            if context:
                generate_molecules_for_protein_multifactors_with_context(
                    protein=protein,
                    input_csv=f"data/{protein}.txt",
                    output_dir=output_dir,
                    api_key=api_key,
                    model_engine=model_engine,
                    gnina_path=gnina_path,
                    config_path=config_path,
                    temp_dir=temp_dir,
                    parameter_ranges=parameter_ranges,
                    target_size=5,
                    max_iterations=1,
                    max_samples=max_samples
                )
            else:
                generate_molecules_for_protein_multifactors(
                    protein=protein,
                    input_csv=f"data/{protein}.txt",
                    output_dir=output_dir,
                    api_key=api_key,
                    model_engine=model_engine,
                    gnina_path=gnina_path,
                    config_path=config_path,
                    temp_dir=temp_dir,
                    parameter_ranges=parameter_ranges,
                    target_size=5,
                    max_iterations=1,
                    max_samples=max_samples
                )

            # Check if generated molecules exist within the interval
            gen_csv = f"{output_dir}/generated.csv"

            if os.path.exists(gen_csv):
                properties_df = pd.read_csv(gen_csv)

                for param, bounds in parameter_ranges.items():
                    properties_df = properties_df[(properties_df[param] >= bounds[0]) &
                                                  (properties_df[param] <= bounds[1])]

                if len(properties_df) > 0:
                    print(f"Feasible molecules found in interval {e_k} with Q-score {Q_k:.4f}.")
                    w_k = Q_k * len(properties_df)
                    Q_values.append(w_k)
                    interval_history.append(e_k)

                    # Add feasible molecules to the intermediate data
                    intermediate_data.extend(properties_df.to_dict(orient="records"))

                    if (w_0 - w_k) / w_0 >= 0.10:  # Stop if no significant improvement
                        print(f"\nNo significant improvement in W-score: {w_k:.4f} <= {w_0:.4f}")
                        break

                    w_0 = w_k
                    e_0 = e_k
                    feasible_node_found = True
                    break
                else:
                    print(f"No feasible molecules in interval {e_k}.")
            else:
                print(f"No molecules generated for interval {e_k}.")

        if not feasible_node_found:
            print("No feasible child nodes found. Ending search.")
            break

        k += 1

    if intermediate_data:
        pd.DataFrame(intermediate_data).drop_duplicates().to_csv(intermediate_csv, index=False)
        print(f"Intermediate feasible molecules saved to {intermediate_csv}")

    # Generate k molecules at the final node
    print("\nGenerating final molecules for the last interval.")
    final_parameter_ranges = {param: interval_history[-1][i] for i, param in enumerate(initial_intervals.keys())}

    if context:
        generate_molecules_for_protein_multifactors_with_context(
            protein=protein,
            input_csv=f"data/{protein}.txt",
            output_dir=output_dir,
            api_key=api_key,
            model_engine=model_engine,
            gnina_path=gnina_path,
            config_path=config_path,
            temp_dir=temp_dir,
            parameter_ranges=final_parameter_ranges,
            target_size=5,
            max_iterations=1,
            max_samples=final_k
        )
    else:
        generate_molecules_for_protein_multifactors(
            protein=protein,
            input_csv=f"data/{protein}.txt",
            output_dir=output_dir,
            api_key=api_key,
            model_engine=model_engine,
            gnina_path=gnina_path,
            config_path=config_path,
            temp_dir=temp_dir,
            parameter_ranges=final_parameter_ranges,
            target_size=5,
            max_iterations=1,
            max_samples=final_k
        )

    print("\nFinal Hypothesis Intervals:", interval_history[-1])
    print("Q-value History:", Q_values)

    # Print the search tree
    print("\nSearch Tree:")
    for node in search_tree:
        print(f"Iteration {node['iteration']}: Interval {node['current_interval']} | Q-score {node['Q_score']:.4f}")
        for child in node['children']:
            print(f"\tChild Interval {child['interval']} | Q-score {child['Q_score']:.4f}")

if __name__ == "__main__":
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    print(date_time)

    protein = "DBH"

    initial_intervals = {
        "CNNaffinity": [3, 10],
        "MolWt": [100, 500],
        "SAS": [0, 4.0]
    }

    data_path = "data"
    labelled_file, unlabelled_file = construct_file_paths(data_path, protein)

    labelled_data = pd.read_csv(labelled_file).to_dict(orient="records")
    unlabelled_data = pd.read_csv(unlabelled_file).to_dict(orient="records")

    api_key = load_api_key() # api key
    model_engine = "gpt-4o"  # "gpt-3.5-turbo", gpt-4o-mini, gpt-4o
    gnina_path = "./docking"
    config_path = f"./docking/{protein}/{protein}_config.txt"
    temp_dir = "/tmp/molecule_generation"
    output_dir = f"results_GenMolMF/{protein}/{model_engine}/{date_time}"

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("*" * 64)
    print(f" PROGRAM: INTERLEAVED_LMLFSTAR (TIMESTAMP: {date_time})")
    print(f" PROTEIN: {protein}")
    print("*" * 64)

    print(f"Search interval:\n\t{initial_intervals}")
   
    search_params = {"s":5, "n":10, "max_samples":10, "final_k":100, "context":True}

    interleaved_LMLFStar(
        protein=protein,
        labelled_data=labelled_data,
        unlabelled_data=unlabelled_data,
        initial_intervals=initial_intervals,
        api_key=api_key,
        model_engine=model_engine,
        gnina_path=gnina_path,
        config_path=config_path,
        temp_dir=temp_dir,
        output_dir=output_dir,
        s=search_params["s"],
        n=search_params["n"],
        max_samples=search_params["max_samples"],
        final_k=search_params["final_k"],
        context=search_params["context"]
    )
    
    print("Run config:")
    print(f"Search intervals : {initial_intervals}")
    print(f"Search params    : {search_params}")

    print("DONE [GenMol.py]")
