#!/usr/bin/env python3
"""
This single script can run one of three different molecule‐generation pipelines, depending on the user’s choice:

1. GenMol1F: Single‐factor search (CNNaffinity) with feasibility based on CNNaffinity.
2. GenMol1F with plus mode: Single‐factor search (CNNaffinity) with extended feasibility test (CNNaffinity plus MolWt and SAS constraints).
3. GenMolMF: Multi‐factor search (e.g. CNNaffinity, MolWt, SAS) with feasibility on all factors.
0. To abort the run

Example run:
    python GenMol.py --protein DBH --target_size 1 --choice mf --context True --model gpt-4o --final_k 100
"""

import argparse
import random
import math
import numpy as np
from datetime import datetime
import os
import json
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

from env_utils import load_api_key
from search import Hypothesis, compute_Q, construct_file_paths
from LMLFStar import (
    generate_molecules_for_protein,
    generate_molecules_for_protein_with_context,
    generate_molecules_for_protein_multifactors,
    generate_molecules_for_protein_multifactors_with_context
)


# =========================
# Helper: Environment Setup
# =========================
def setup_environment(protein, results_subdir, data_path="data", model_engine="gpt-4o"):
    """
    Sets up common parameters and directories.
    Returns a dictionary with:
      - date_time (timestamp)
      - labelled_data and unlabelled_data (from CSV files)
      - api_key, model_engine, gnina_path, config_path, temp_dir, output_dir
    """
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    labelled_file, unlabelled_file = construct_file_paths(data_path, protein)
    labelled_data = pd.read_csv(labelled_file).to_dict(orient="records")
    unlabelled_data = pd.read_csv(unlabelled_file).to_dict(orient="records")
    api_key = load_api_key()
    gnina_path = "./docking"
    config_path = f"./docking/{protein}/{protein}_config.txt"
    temp_dir = "/tmp/molecule_generation"
    output_dir = f"results/{results_subdir}/{protein}/{model_engine}/{date_time}"
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    return {
        "date_time": date_time,
        "labelled_data": labelled_data,
        "unlabelled_data": unlabelled_data,
        "api_key": api_key,
        "model_engine": model_engine,
        "gnina_path": gnina_path,
        "config_path": config_path,
        "temp_dir": temp_dir,
        "output_dir": output_dir
    }


# ====================================
# Pipeline 1: GenMol1F (Single-Factor)
# ====================================
def GenMol1F(seed=0, protein="DBH", target_size=5, final_k=20, context=False, model_engine="gpt-4o", plus_mode=False):
    """
    Single-factor search for CNNaffinity.
    Checks feasibility solely by verifying that the molecule’s CNNaffinity
    lies within the current search interval.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if plus_mode:
        env = setup_environment(protein, "GenMol1Fplus", model_engine=model_engine)
    else:
        env = setup_environment(protein, "GenMol1F", model_engine=model_engine)
        
    labelled_data = env["labelled_data"]
    unlabelled_data = env["unlabelled_data"]
    api_key = env["api_key"]
    gnina_path = env["gnina_path"]
    config_path = env["config_path"]
    temp_dir = env["temp_dir"]
    output_dir = env["output_dir"]
    
    def interleaved_LMLFStar(protein, labelled_data, unlabelled_data, initial_interval,
                             api_key, model_engine, gnina_path, config_path, temp_dir,
                             output_dir, s=4, n=10, max_samples=5, final_k=20, target_size=5, context=False):

        factor = lambda x: x['CNNaffinity']
        e_0 = initial_interval
        h_0 = Hypothesis([factor], e_0)
        theta_ext_h_default = len(unlabelled_data) / (len(labelled_data) + len(unlabelled_data))
        w_0 = compute_Q(h_0, "Background Knowledge", labelled_data, theta_ext_h_approx=theta_ext_h_default)
        
        best_w = w_0
        patience = 3
        patience_counter = 0
        
        iteration_numbers = []
        current_Q_history = []
        best_Q_history = []
        
        k = 1
        interval_history = [e_0]
        Q_values = [w_0]
        w_values = [w_0]
        search_tree = []
        intermediate_csv = os.path.join(output_dir, "intermediate.csv")
        intermediate_data = []
                
        while k <= n:
            lhs_samples = scipy.stats.qmc.LatinHypercube(d=1, seed=seed).random(n=s)
            quantiles = list(map(float, np.linspace(e_0[0][0], e_0[0][1], s + 1)))  # s partitions
            E_k = [
                [[float(quantiles[min(int(sample * s), s - 1)]), float(e_0[0][1])]]
                for sample in lhs_samples.flatten()
            ]
            
            S = []
            for e in E_k:
                h_k = Hypothesis([factor], e)
                Q_k = compute_Q(h_k, "Background Knowledge", labelled_data, theta_ext_h_approx=theta_ext_h_default)
                S.append((Q_k, e))
            
            print("----------------------------------------")
            print(f"Iteration {k}:")
            print(f"  Current Interval: {e_0}, Current Q = {w_0:.4f}, Best Q = {best_w:.4f}")
            print(f"  Candidate Intervals: {E_k}")
            print(f"  Candidate Q scores: {[round(q, 4) for q, _ in S]}")
            
            search_tree.append({
                "iteration": k,
                "current_interval": e_0,
                "Q_score": w_0,
                "children": [{"interval": e, "Q_score": q} for q, e in S]
            })
            
            sorted_S = sorted(S, key=lambda x: x[0], reverse=True)
            feasible_node_found = False
            w_k = 0 
            
            for (Q_k, e_k) in sorted_S:
                if Q_k < best_w * 0.8:
                    continue  # bad candidate
                
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
                if os.path.exists(gen_csv) and os.path.getsize(gen_csv) > 0:
                    properties_df = pd.read_csv(gen_csv)
                    
                    if plus_mode:
                        feasible_df = properties_df[
                            (properties_df['CNNaffinity'] >= affinity_range[0]) &
                            (properties_df['CNNaffinity'] <= affinity_range[1]) &
                            (properties_df['MolWt'] < 500) &
                            (properties_df['SAS'] < 5.0)
                        ]
                    else:
                        feasible_df = properties_df[
                            (properties_df['CNNaffinity'] >= e_k[0][0]) &
                            (properties_df['CNNaffinity'] <= e_k[0][1])
                        ]
                    
                    if len(feasible_df) > 0:
                        print(f"  Feasible molecules found in interval {e_k} with Q-score {Q_k:.4f}.")
                        intermediate_data.extend(feasible_df.to_dict(orient="records"))
                        best_w = max(best_w, Q_k)
                        w_k = Q_k
                        w_0 = Q_k
                        e_0 = e_k
                        patience_counter = 0
                        feasible_node_found = True
                        break
                else:
                    print(f"  No molecules generated for interval {e_k}.")
                    w_k = 0
                    patience_counter += 1
            
            if not feasible_node_found:
                print("No feasible candidate nodes found that meet the threshold. Ending search.")
                break

            iteration_numbers.append(k)
            current_Q_history.append(w_0)
            best_Q_history.append(best_w)
            interval_history.append(e_0)
            Q_values.append(w_0) 
            w_values.append(w_k) 
            
            if patience_counter >= patience:
                print("Patience limit reached without improvement. Ending search.")
                break
            
            k += 1
        
        if intermediate_data:
            pd.DataFrame(intermediate_data).drop_duplicates().to_csv(intermediate_csv, index=False)
            print(f"Intermediate feasible molecules saved to {intermediate_csv}")
        
        print("\nGenerating final molecules for the optimal interval.")
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
                target_size=target_size,
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
                target_size=target_size,
                max_iterations=1,
                max_samples=final_k
            )
        
        if iteration_numbers:
            plt.figure(figsize=(8, 6))
            plt.plot(iteration_numbers, current_Q_history, marker='o', label='Current Q Score')
            plt.plot(iteration_numbers, best_Q_history, marker='x', linestyle='--', label='Best Q Score')
            plt.xlabel('Iteration')
            plt.ylabel('Q Score')
            plt.title('Search Progression of Q Score')
            plt.legend()
            plt.grid(True)
            pdf_path = os.path.join(output_dir, "search_progress.pdf")
            plt.savefig(pdf_path)
            plt.close()
            print(f"Search progression plot saved to: {pdf_path}")
        
        # Print and save the search log.
        log_lines = []
        log_lines.append("Search Tree:")
        for node in search_tree:
            log_lines.append(f"Iteration {node['iteration']}: Interval {node['current_interval']} | Q-score {node['Q_score']:.4f}")
            for child in node['children']:
                log_lines.append(f"\tChild Interval {child['interval']} | Q-score {child['Q_score']:.4f}")

        log_lines.append(f"Selected Intervals: {interval_history}")
        log_lines.append(f"Selected Q-scores: {Q_values}")
        log_lines.append(f"Selected W-values: {w_values}") 
        log_lines.append(f"\nFinal Hypothesis Interval: {interval_history[-1]}")
        log_str = "\n".join(log_lines)
        log_file_path = os.path.join(output_dir, "log.txt")
        with open(log_file_path, "w") as log_file:
            log_file.write(log_str)
        print(f"Hypothesis search log saved to: {log_file_path}")
    
    # other run parameters: initial search space, search params, etc.
    initial_interval = [[2, 10]]
    search_params = {"s": 30, "n": 10, "max_samples": 10, "final_k": final_k, "context": context}
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
                         target_size=target_size,
                         context=search_params["context"])
    
    # Save configuration file.
    config_data = {
        "protein": protein,
        "target_size": target_size,
        "context": context,
        "model_engine": model_engine,
        "search_intervals": initial_interval,
        "search_params": search_params,
        "plus_mode": plus_mode
    }
    config_file_path = os.path.join(output_dir, "config.json")
    with open(config_file_path, "w") as f:
        json.dump(config_data, f, indent=4)
    print(f"Run configuration saved to: {config_file_path}")
    print("DONE [GenMol1F]")


# ====================================
# Pipeline 2: GenMolMF (Multi-Factor)
# ====================================
def GenMolMF(seed=0, protein="DBH", target_size=5, final_k=20, context=False, model_engine="gpt-4o"):
    """
    Multi-factor search.
    The algorithm searches for optimal parameter ranges for multiple properties
    (e.g. CNNaffinity, MolWt, SAS) and verifies that each molecule satisfies
    the corresponding constraints.
    """
    random.seed(seed)
    np.random.seed(seed)

    env = setup_environment(protein, "GenMolMF", model_engine=model_engine)
    labelled_data = env["labelled_data"]
    unlabelled_data = env["unlabelled_data"]
    api_key = env["api_key"]
    gnina_path = env["gnina_path"]
    config_path = env["config_path"]
    temp_dir = env["temp_dir"]
    output_dir = env["output_dir"]


    def interleaved_LMLFStar(protein, labelled_data, unlabelled_data, initial_intervals,
                             api_key, model_engine, gnina_path, config_path, temp_dir,
                             output_dir, s=4, n=10, max_samples=5, final_k=100, target_size=5, context=False):

        factors = [lambda x, p=param: x.get(p) for param in initial_intervals.keys()]
        e_0 = [initial_intervals[param] for param in initial_intervals]
        h_0 = Hypothesis(factors, e_0)

        theta_ext_h_default = len(unlabelled_data) / (len(labelled_data) + len(unlabelled_data))
        w_0 = compute_Q(h_0, "Background Knowledge", labelled_data, theta_ext_h_approx=theta_ext_h_default)

        best_w = w_0
        patience = 3
        patience_counter = 0

        iteration_numbers = []
        current_Q_history = []
        best_Q_history = []

        k = 1
        interval_history = [e_0]
        Q_values = [w_0]
        w_values = [w_0]
        search_tree = []
        intermediate_csv = os.path.join(output_dir, "intermediate.csv")
        intermediate_data = []

        while k <= n:
            lhs_samples = scipy.stats.qmc.LatinHypercube(d=len(initial_intervals), seed=seed).random(n=s)
            E_k = []
            for sample in lhs_samples:
                new_intervals = []
                for i, param in enumerate(initial_intervals.keys()):
                    quantiles = np.linspace(initial_intervals[param][0], initial_intervals[param][1], s + 1)
                    index = min(max(int(sample[i] * s), 0), s - 1)
                    if param == "CNNaffinity": #keep max end fixed
                        new_intervals.append([float(quantiles[index]), float(initial_intervals[param][1])])
                    elif param in ["MolWt", "SAS"]: #keep min end fixed
                        new_intervals.append([float(initial_intervals[param][0]), float(quantiles[index])])
                E_k.append(new_intervals)
            
            S = []
            for e in E_k:
                h_k = Hypothesis(factors, e)
                Q_k = compute_Q(h_k, "Background Knowledge", labelled_data, theta_ext_h_approx=theta_ext_h_default)
                S.append((Q_k, e))

            print("----------------------------------------")
            print(f"Iteration {k}: Current Intervals {e_0} | Q-score {w_0:.4f} (Best Q: {best_w:.4f})")
            print(f"  Candidate Intervals: {E_k}")
            print(f"  Candidate Q scores: {[round(q, 4) for q, _ in S]}")

            search_tree.append({
                "iteration": k,
                "current_interval": e_0,
                "Q_score": w_0,
                "children": [{"interval": e, "Q_score": q} for q, e in S]
            })

            sorted_S = sorted(S, key=lambda x: x[0], reverse=True)
            feasible_node_found = False
            w_k = 0

            for (Q_k, e_k) in sorted_S:
                if Q_k < best_w * 0.8:
                    continue

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
                if os.path.exists(gen_csv) and os.path.getsize(gen_csv) > 0:
                    properties_df = pd.read_csv(gen_csv)

                    for param, bounds in parameter_ranges.items():
                        properties_df = properties_df[(properties_df[param] >= bounds[0]) &
                                                      (properties_df[param] <= bounds[1])]

                    if len(properties_df) > 0:
                        print(f"  Feasible molecules found in interval {e_k} with Q-score {Q_k:.4f}.")
                        intermediate_data.extend(properties_df.to_dict(orient="records"))
                        best_w = max(best_w, Q_k)
                        w_k = Q_k
                        w_0 = Q_k
                        e_0 = e_k
                        patience_counter = 0
                        feasible_node_found = True
                        break
                else:
                    print(f"  No molecules generated for interval {e_k}.")
                    w_k = 0
                    patience_counter += 1

            if not feasible_node_found:
                print("No feasible candidate nodes found that meet the threshold. Ending search.")
                break

            iteration_numbers.append(k)
            current_Q_history.append(w_0)
            best_Q_history.append(best_w)
            interval_history.append(e_0)
            Q_values.append(w_0)
            w_values.append(w_k)

            if patience_counter >= patience:
                print("Patience limit reached without improvement. Ending search.")
                break

            k += 1

        if intermediate_data:
            pd.DataFrame(intermediate_data).drop_duplicates().to_csv(intermediate_csv, index=False)
            print(f"Intermediate feasible molecules saved to {intermediate_csv}")

        print("\nGenerating final molecules for the optimal interval.")
        final_parameter_ranges = {param: interval_history[-1][i] for i, param in enumerate(initial_intervals)}

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
                target_size=target_size,
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
                target_size=target_size,
                max_iterations=1,
                max_samples=final_k
            )
        
        if iteration_numbers:
            plt.figure(figsize=(8, 6))
            plt.plot(iteration_numbers, current_Q_history, marker='o', label='Current Q Score')
            plt.plot(iteration_numbers, best_Q_history, marker='x', linestyle='--', label='Best Q Score')
            plt.xlabel('Iteration')
            plt.ylabel('Q Score')
            plt.title('Search Progression of Q Score')
            plt.legend()
            plt.grid(True)
            pdf_path = os.path.join(output_dir, "search_progress.pdf")
            plt.savefig(pdf_path)
            plt.close()
            print(f"Search progression plot saved to: {pdf_path}")

        log_lines = []
        log_lines.append("Search Tree:")
        for node in search_tree:
            log_lines.append(f"Iteration {node['iteration']}: Interval {node['current_interval']} | Q-score {node['Q_score']:.4f}")
            for child in node['children']:
                log_lines.append(f"\tChild Interval {child['interval']} | Q-score {child['Q_score']:.4f}")
        log_lines.append(f"Q-value History: {Q_values}")
        log_lines.append(f"W-value History: {w_values}")
        log_lines.append(f"Final Hypothesis Interval: {interval_history[-1]}")
        log_str = "\n".join(log_lines)
        log_file_path = os.path.join(output_dir, "log.txt")
        with open(log_file_path, "w") as log_file:
            log_file.write(log_str)
        print(f"Hypothesis search log saved to: {log_file_path}")
        
    # other run parameters: initial search space, search params, etc.
    initial_intervals = {"CNNaffinity": [3, 10], "MolWt": [200, 700], "SAS": [0, 7.0]}
    #initial_intervals = {"CNNaffinity": [2, 10], "MolWt": [0, 500]}

    search_params = {"s": 10, "n": 10, "max_samples": 10, "final_k": final_k, "context": context}
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
                         target_size=target_size,
                         context=search_params["context"])

    config_data = {
        "protein": protein,
        "target_size": target_size,
        "context": context,
        "model_engine": model_engine,
        "search_intervals": initial_intervals,
        "search_params": search_params
    }
    config_file_path = os.path.join(output_dir, "config.json")
    with open(config_file_path, "w") as f:
        json.dump(config_data, f, indent=4)
    print(f"Run configuration saved to: {config_file_path}")
    print("DONE [GenMolMF]")


# ================================
# Main: Parsing arguments and run
# ================================
def main():
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    
    print("="*63)
    print(f"   TARGET-SPECIFIC LEAD DISCOVERY USING AN LLM [{date_time}]")
    print("="*63)
    
    parser = argparse.ArgumentParser(
        description="TARGET-SPECIFIC LEAD DISCOVERY USING AN LLM"
    )
    parser.add_argument("--choice", type=str, required=True,
                        help="Choice of pipeline: '1' (or '1f') for GenMol1F; '2' (or '1fplus') for GenMol1F with plus mode; '3' (or 'mf') for GenMolMF; '0' to abort")
    parser.add_argument("--protein", type=str, default="DBH", help="Target protein")
    parser.add_argument("--target_size", type=int, default=5, help="Target size for molecule generation")
    parser.add_argument("--context", type=str, default="False", help="Use context (True/False)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model engine to use")
    parser.add_argument("--final_k", type=int, default=20, help="Number of molecules to generate in the final step")
    args = parser.parse_args()
    
    context = args.context.lower() in ("true", "1", "yes")
    
    choice = args.choice.lower()
    print(args)

    if choice in ["1", "1f"]:
        print("Calling GenMol1F ...")
        GenMol1F(seed=0, 
                 protein=args.protein, 
                 target_size=args.target_size, 
                 final_k=args.final_k, 
                 context=context, 
                 model_engine=args.model)
    elif choice in ["2", "1fplus"]:        
        print("Calling GenMol1F with plus mode ...")
        GenMol1F(seed=0, 
                 protein=args.protein, 
                 target_size=args.target_size, 
                 final_k=args.final_k, 
                 context=context, 
                 model_engine=args.model,
                 plus_mode=True)
    elif choice in ["3", "mf"]:
        print("Calling GenMolMF ...")
        GenMolMF(seed=0, 
                 protein=args.protein,
                 target_size=args.target_size, 
                 final_k=args.final_k, 
                 context=context, 
                 model_engine=args.model)
    else:
        print(f"Choice {args.choice} is invalid. Aborting...")
        return 1

if __name__ == "__main__":
    main()
