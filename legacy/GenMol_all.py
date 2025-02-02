#!/usr/bin/env python3
"""
This single script can run one of three different molecule‐generation pipelines,
depending on the user’s choice:

1. GenMol1F: Single‐factor search (CNNaffinity) with feasibility based on CNNaffinity.
2. GenMol1Fplus: Single‐factor search (CNNaffinity) with extended feasibility test
   (CNNaffinity plus MolWt and SAS constraints).
3. GenMolMF: Multi‐factor search (e.g. CNNaffinity, MolWt, SAS) with feasibility on all factors.
0. To abort the run

Usage:
    python GenMol.py --choice 1
    python GenMol.py --choice 2
    python GenMol.py --choice 3
"""

import argparse
import random
import math
import numpy as np
from datetime import datetime
import os
import pandas as pd

from env_utils import load_api_key
from search import Hypothesis, compute_Q, construct_file_paths

from LMLFStar import (
    generate_molecules_for_protein,
    generate_molecules_for_protein_with_context
)

from LMLFStar import (
    generate_molecules_for_protein_multifactors,
    generate_molecules_for_protein_multifactors_with_context
)


def GenMol1F(seed=0):
    """
    Pipeline 1: GenMol1F – Single-factor search for CNNaffinity.
    
    The search iteratively refines an interval for CNNaffinity, and at each iteration,
    the generated molecules are checked for feasibility (i.e. CNNaffinity lies in the interval).
    """
    random.seed(seed)
    np.random.seed(seed)

    def interleaved_LMLFStar(protein, labelled_data, unlabelled_data, initial_interval,
                             api_key, model_engine, gnina_path, config_path, temp_dir,
                             output_dir, s=4, n=10, max_samples=5, final_k=10, context=False):
        """
        Interleaved search and molecule generation.
        The feasibility of generated molecules is checked against one constraint:
            CNNaffinity in [optimal range]
        """
        factor = lambda x: x['CNNaffinity']
        e_0 = initial_interval
        h_0 = Hypothesis([factor], e_0)

        theta_ext_h_default = len(unlabelled_data) / (len(labelled_data) + len(unlabelled_data))
        w_0 = compute_Q(h_0, "Background Knowledge", labelled_data,
                        theta_ext_h_approx=theta_ext_h_default)

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
                Q_k = compute_Q(h_k, "Background Knowledge", labelled_data,
                                theta_ext_h_approx=theta_ext_h_default)
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
                if context:
                    generate_molecules_for_protein_with_context(
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
                else:
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
                    feasible_df = properties_df[
                        (properties_df['CNNaffinity'] >= affinity_range[0]) &
                        (properties_df['CNNaffinity'] <= affinity_range[1])
                    ]

                    if len(feasible_df) > 0:
                        print(f"Feasible molecules found in interval {e_k} with Q-score {Q_k:.4f}.")
                        w_k = Q_k * len(feasible_df)
                        Q_values.append(w_k)
                        interval_history.append(e_k)

                        # Save feasible molecules
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

        # Generate final molecules at the final node
        print("\nGenerating final molecules for the last interval.")
        final_affinity_range = [float(interval_history[-1][0][0]), float(interval_history[-1][0][1])]

        if context:
            generate_molecules_for_protein_with_context(
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
        else:
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
        print("\nSearch Tree:")
        for node in search_tree:
            print(f"Iteration {node['iteration']}: Interval {node['current_interval']} | Q-score {node['Q_score']:.4f}")
            for child in node['children']:
                print(f"\tChild Interval {child['interval']} | Q-score {child['Q_score']:.4f}")

    # -----------------------
    # Main block for GenMol1F
    # -----------------------
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    print(date_time)

    protein = "DBH"
    initial_interval = [[2, 10]]  # For CNNaffinity

    data_path = "data"
    labelled_file, unlabelled_file = construct_file_paths(data_path, protein)
    labelled_data = pd.read_csv(labelled_file).to_dict(orient="records")
    unlabelled_data = pd.read_csv(unlabelled_file).to_dict(orient="records")

    api_key = load_api_key()  # Load API key (assumed to be secure)
    model_engine = "gpt-4o-mini"  # or "gpt-3.5-turbo", etc.
    gnina_path = "./docking"
    config_path = f"./docking/{protein}/{protein}_config.txt"
    temp_dir = "/tmp/molecule_generation"
    output_dir = f"results/GenMol1F/{protein}/{model_engine}/{date_time}"

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("*" * 64)
    print(f" PROGRAM: INTERLEAVED_LMLFSTAR (TIMESTAMP: {date_time})")
    print(f" PROTEIN: {protein}")
    print("*" * 64)

    search_params = {"s": 4, "n": 10, "max_samples": 10, "final_k": 20, "context": True}

    interleaved_LMLFStar(protein=protein,
                         labelled_data=labelled_data,
                         unlabelled_data=unlabelled_data,
                         initial_interval=initial_interval,
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
                         context=search_params["context"])

    print("Run config:")
    print(f"Search interval  : CNNaffinity {initial_interval}")
    print(f"Search params    : {search_params}")
    print("DONE [GenMol1F]")


def GenMol1Fplus(seed=0):
    """
    Pipeline 2: GenMol1Fplus – Single-factor search with extended feasibility.
    
    The search is still over CNNaffinity, but the feasibility check now also requires that:
      - MolWt < 500
      - SAS < 5.0
    """
    random.seed(seed)
    np.random.seed(seed)

    def interleaved_LMLFStar(protein, labelled_data, unlabelled_data, initial_interval,
                             api_key, model_engine, gnina_path, config_path, temp_dir,
                             output_dir, s=4, n=10, max_samples=5, final_k=10, context=False):
        """
        Interleaved search and molecule generation with extended feasibility checks.
        Feasibility is determined by:
          1. CNNaffinity within the search interval.
          2. MolWt < 500.
          3. SAS < 5.0.
        """
        factor = lambda x: x['CNNaffinity']
        e_0 = initial_interval
        h_0 = Hypothesis([factor], e_0)

        theta_ext_h_default = len(unlabelled_data) / (len(labelled_data) + len(unlabelled_data))
        w_0 = compute_Q(h_0, "Background Knowledge", labelled_data,
                        theta_ext_h_approx=theta_ext_h_default)

        k = 1
        interval_history = [e_0]
        Q_values = [w_0]
        search_tree = []

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
                Q_k = compute_Q(h_k, "Background Knowledge", labelled_data,
                                theta_ext_h_approx=theta_ext_h_default)
                S.append((Q_k, e))

            print(f"Quantiles\t\t: {quantiles}")
            print(f"Sampled intervals\t: {E_k}")
            print(f"Q-scores\t\t: {[q for q, e in S]}\n")

            search_tree.append({
                "iteration": k,
                "current_interval": e_0,
                "Q_score": w_0,
                "children": [{"interval": e, "Q_score": q} for q, e in S]
            })

            sorted_S = sorted(S, key=lambda x: x[0], reverse=True)
            feasible_node_found = False

            for (Q_k, e_k) in sorted_S:
                print(f"Evaluating node with interval {e_k} and Q-score {Q_k:.4f}")

                affinity_range = [float(e_k[0][0]), float(e_k[0][1])]

                if context:
                    generate_molecules_for_protein_with_context(
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
                else:
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

                gen_csv = f"{output_dir}/generated.csv"

                if os.path.exists(gen_csv):
                    properties_df = pd.read_csv(gen_csv)

                    # Apply extended feasibility checks
                    feasible_df = properties_df[
                        (properties_df['CNNaffinity'] >= affinity_range[0]) &
                        (properties_df['CNNaffinity'] <= affinity_range[1]) &
                        (properties_df['MolWt'] < 500) &
                        (properties_df['SAS'] < 5.0)
                    ]

                    if len(feasible_df) > 0:
                        print(f"Feasible molecules found in interval {e_k} with Q-score {Q_k:.4f}.")
                        w_k = Q_k * len(feasible_df)
                        Q_values.append(w_k)
                        interval_history.append(e_k)

                        intermediate_data.extend(feasible_df.to_dict(orient="records"))

                        if (w_0 - w_k) / w_0 >= 0.10:
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

        print("\nGenerating final molecules for the last interval.")
        final_affinity_range = [float(interval_history[-1][0][0]), float(interval_history[-1][0][1])]

        if context:
            generate_molecules_for_protein_with_context(
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
        else:
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
        print("\nSearch Tree:")
        for node in search_tree:
            print(f"Iteration {node['iteration']}: Interval {node['current_interval']} | Q-score {node['Q_score']:.4f}")
            for child in node['children']:
                print(f"\tChild Interval {child['interval']} | Q-score {child['Q_score']:.4f}")

    # -----------------------------
    # Main block for GenMol1Fplus
    # -----------------------------
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    print(date_time)

    protein = "DBH"
    initial_interval = [[2, 10]]  # For CNNaffinity

    data_path = "data"
    labelled_file, unlabelled_file = construct_file_paths(data_path, protein)
    labelled_data = pd.read_csv(labelled_file).to_dict(orient="records")
    unlabelled_data = pd.read_csv(unlabelled_file).to_dict(orient="records")

    api_key = load_api_key()
    model_engine = "gpt-4o"  # or another engine
    gnina_path = "./docking"
    config_path = f"./docking/{protein}/{protein}_config.txt"
    temp_dir = "/tmp/molecule_generation"
    output_dir = f"results/GenMol1Fplus/{protein}/{model_engine}/{date_time}"

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("*" * 64)
    print(f" PROGRAM: INTERLEAVED_LMLFSTAR (TIMESTAMP: {date_time})")
    print(f" PROTEIN: {protein}")
    print("*" * 64)

    search_params = {"s": 5, "n": 10, "max_samples": 10, "final_k": 10, "context": True}

    interleaved_LMLFStar(protein=protein,
                         labelled_data=labelled_data,
                         unlabelled_data=unlabelled_data,
                         initial_interval=initial_interval,
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
                         context=search_params["context"])

    print("Run config:")
    print(f"Search interval  : CNNaffinity {initial_interval}")
    print(f"Search params    : {search_params}")
    print("DONE [GenMol1Fplus]")


def GenMolMF(seed=0):
    """
    Pipeline 3: GenMolMF – Multi-factor search.
    
    The algorithm searches for optimal parameter ranges for multiple factors (e.g. CNNaffinity,
    MolWt, SAS, etc.). Generated molecules are checked for feasibility on each factor.
    """
    random.seed(seed)
    np.random.seed(seed)

    def interleaved_LMLFStar(protein, labelled_data, unlabelled_data, initial_intervals,
                             api_key, model_engine, gnina_path, config_path, temp_dir,
                             output_dir, s=4, n=10, max_samples=5, final_k=10, context=False):
        """
        Interleaved search and molecule generation for multi-factors.
        Feasibility is checked such that for each parameter the molecule's property falls within
        the optimal range determined by the search.
        """
        # Create a list of functions—one for each parameter.
        factors = [lambda x, p=param: x.get(p) for param in initial_intervals.keys()]
        e_0 = [initial_intervals[param] for param in initial_intervals]
        h_0 = Hypothesis(factors, e_0)

        theta_ext_h_default = len(unlabelled_data) / (len(labelled_data) + len(unlabelled_data))
        w_0 = compute_Q(h_0, "Background Knowledge", labelled_data,
                        theta_ext_h_approx=theta_ext_h_default)

        k = 1
        interval_history = [e_0]
        Q_values = [w_0]
        search_tree = []

        intermediate_csv = os.path.join(output_dir, "intermediate.csv")
        intermediate_data = []

        while k <= n:
            print(f"\nIteration {k}: Current Intervals {e_0} | Q-score {w_0:.4f}\n")

            E_k = []
            for dim, (lower, upper) in enumerate(e_0):
                quantiles = list(map(float, np.linspace(lower, upper, 5)))
                for q in range(len(quantiles) - 1):
                    sub_interval = e_0[:]  # shallow copy of the list of intervals
                    sub_interval[dim] = [quantiles[q], quantiles[q + 1]]
                    E_k.append(sub_interval)

            S = []
            for e in E_k:
                h_k = Hypothesis(factors, e)
                Q_k = compute_Q(h_k, "Background Knowledge", labelled_data,
                                theta_ext_h_approx=theta_ext_h_default)
                S.append((Q_k, e))

            print(f"Sampled intervals: {E_k}")
            print(f"Q-scores: {[q for q, e in S]}\n")

            search_tree.append({
                "iteration": k,
                "current_interval": e_0,
                "Q_score": w_0,
                "children": [{"interval": e, "Q_score": q} for q, e in S]
            })

            sorted_S = sorted(S, key=lambda x: x[0], reverse=True)
            feasible_node_found = False

            for (Q_k, e_k) in sorted_S:
                print(f"Evaluating node with interval {e_k} and Q-score {Q_k:.4f}")

                parameter_ranges = {param: e_k[i] for i, param in enumerate(initial_intervals.keys())}

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

                gen_csv = f"{output_dir}/generated.csv"

                if os.path.exists(gen_csv):
                    properties_df = pd.read_csv(gen_csv)
                    for param, bounds in parameter_ranges.items():
                        properties_df = properties_df[
                            (properties_df[param] >= bounds[0]) &
                            (properties_df[param] <= bounds[1])
                        ]
                    if len(properties_df) > 0:
                        print(f"Feasible molecules found in interval {e_k} with Q-score {Q_k:.4f}.")
                        w_k = Q_k * len(properties_df)
                        Q_values.append(w_k)
                        interval_history.append(e_k)

                        intermediate_data.extend(properties_df.to_dict(orient="records"))

                        if (w_0 - w_k) / w_0 >= 0.10:
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
        print("\nSearch Tree:")
        for node in search_tree:
            print(f"Iteration {node['iteration']}: Interval {node['current_interval']} | Q-score {node['Q_score']:.4f}")
            for child in node['children']:
                print(f"\tChild Interval {child['interval']} | Q-score {child['Q_score']:.4f}")

    # ---------------------------
    # Main block for GenMolMF
    # ---------------------------
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

    api_key = load_api_key()
    model_engine = "gpt-4o"  # or another engine
    gnina_path = "./docking"
    config_path = f"./docking/{protein}/{protein}_config.txt"
    temp_dir = "/tmp/molecule_generation"
    output_dir = f"results/GenMolMF/{protein}/{model_engine}/{date_time}"

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("*" * 64)
    print(f" PROGRAM: INTERLEAVED_LMLFSTAR (TIMESTAMP: {date_time})")
    print(f" PROTEIN: {protein}")
    print("*" * 64)
    print(f"Search interval:\n\t{initial_intervals}")

    search_params = {"s": 5, "n": 10, "max_samples": 10, "final_k": 100, "context": True}

    interleaved_LMLFStar(protein=protein,
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
                         context=search_params["context"])

    print("Run config:")
    print(f"Search intervals : {initial_intervals}")
    print(f"Search params    : {search_params}")
    print("DONE [GenMolMF]")


def main():
    parser = argparse.ArgumentParser(
        description="Run molecule generation with different constraints and search strategies."
    )
    parser.add_argument("--choice", type=int, choices=[1, 2, 3, 0], required=True,
                        help="Choose 1 for GenMol1F, 2 for GenMol1Fplus, 3 for GenMolMF, 0 for Abort")
    args = parser.parse_args()

    if args.choice == 1:
        GenMol1F()
    elif args.choice == 2:
        GenMol1Fplus()
    elif args.choice == 3:
        GenMolMF()
    else:
        print(f"Run choice is {args.choice}. Aborting ...")
        return 1


if __name__ == "__main__":
    main()
