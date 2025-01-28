"""
GenMol.py: Generate molecules while doing the LMLFStar search with single constraint (CNNaffinity)
The code searches for an optimal parameter range (CNNaffinity \in [x,10]) while interleaving it with generating molecules from an LLM
"""

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


def interleaved_LMLFStar(protein, labelled_data, unlabelled_data, initial_interval, api_key, model_engine, gnina_path, config_path, temp_dir, output_dir, s=4, n=10, max_samples=5, final_k=10):
    """
    Interleaved search and molecule generation.

    Args:
        protein (str): Target protein name.
        labelled_data (list): Labelled dataset (list of dictionaries).
        unlabelled_data (list): Unlabelled dataset (list of dictionaries).
        initial_interval (list): Initial interval for the parameter.
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
    factor = lambda x: x['CNNaffinity']
    e_0 = initial_interval
    h_0 = Hypothesis([factor], e_0)

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
        print(f"\nIteration {k}: Current Interval {e_0} | Q-score {w_0:.4f}\n")

        quantiles = list(map(float, np.linspace(e_0[0][0], e_0[0][1], 5)))
        E_k = [
            [[random.uniform(quantiles[q], quantiles[q + 1]), e_0[0][1]]]
            for q in range(len(quantiles) - 1) for _ in range(s // 4)
        ]

        S = []
        for e in E_k:
            h_k = Hypothesis([factor], e)
            Q_k = compute_Q(h_k, "Background Knowledge", labelled_data, theta_ext_h_approx=theta_ext_h_default)
            S.append((Q_k, e))

        print(f"Quantiles\t\t: {quantiles}")
        print(f"Sampled intervals\t: {E_k}")
        print(f"Q-scores\t\t: {[q for q, e in S]}\n")

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

            affinity_range = [float(e_k[0][0]), float(e_k[0][1])]

            # Generate molecules for the interval
            generate_molecules_for_protein(
                protein=protein,
                input_csv=f"data/{protein}.txt",
                output_dir=output_dir,
                api_key=api_key,
                model_engine=model_engine,
                gnina_path=gnina_path,
                config_path=config_path,
                temp_dir=temp_dir,
                affinity_range=affinity_range,
                target_size=5,
                max_iterations=1,
                max_samples=max_samples
            )

            # Check if generated molecules exist within the interval
            gen_csv = f"{output_dir}/generated.csv"

            if os.path.exists(gen_csv):
                properties_df = pd.read_csv(gen_csv)
                feasible_df = properties_df[(properties_df['CNNaffinity'] >= affinity_range[0]) &
                                            (properties_df['CNNaffinity'] <= affinity_range[1])]

                if len(feasible_df) > 0:
                    print(f"Feasible molecules found in interval {e_k} with Q-score {Q_k:.4f}.")
                    w_k = Q_k * len(feasible_df)
                    Q_values.append(w_k)
                    interval_history.append(e_k)

                    # Add feasible molecules to the intermediate data
                    intermediate_data.extend(feasible_df.to_dict(orient="records"))

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
    final_affinity_range = [float(interval_history[-1][0][0]), float(interval_history[-1][0][1])]

    generate_molecules_for_protein(
        protein=protein,
        input_csv=f"data/{protein}.txt",
        output_dir=output_dir,
        api_key=api_key,
        model_engine=model_engine,
        gnina_path=gnina_path,
        config_path=config_path,
        temp_dir=temp_dir,
        affinity_range=final_affinity_range,
        target_size=5,
        max_iterations=1,
        max_samples=final_k
    )

    print("\nFinal Hypothesis Interval:", interval_history[-1])
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
    initial_interval = [[2, 10]]

    data_path = "data"
    labelled_file, unlabelled_file = construct_file_paths(data_path, protein)

    labelled_data = pd.read_csv(labelled_file).to_dict(orient="records")
    unlabelled_data = pd.read_csv(unlabelled_file).to_dict(orient="records")

    api_key = "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"
    model_engine = "gpt-4o-mini" #"gpt-3.5-turbo", gpt-4o-mini, gpt-4o
    gnina_path = "./docking"
    config_path = f"./docking/{protein}/{protein}_config.txt"
    temp_dir = "/tmp/molecule_generation"
    output_dir = f"results_GenMol/{protein}/{model_engine}/{date_time}"

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("*" * 64)
    print(f" PROGRAM: INTERLEAVED_LMLFSTAR (TIMESTAMP: {date_time})")
    print(f" PROTEIN: {protein}")
    print("*" * 64)

    interleaved_LMLFStar(
        protein=protein,
        labelled_data=labelled_data,
        unlabelled_data=unlabelled_data,
        initial_interval=initial_interval,
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
    
    print("DONE [GenMol.py]")
