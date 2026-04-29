#!/usr/bin/env python3
"""
GenMol_ZincGPT2.py

This script runs molecule generation pipelines using the ZincGPT2 model, following the same
structure as GenMol_claude.py but adapted for local GPT2-based generation.

Pipelines:
1. GenMol1F: Single‐factor search (CNNaffinity) with feasibility based on CNNaffinity.
2. GenMol1F with plus mode: Single‐factor search (CNNaffinity) with extended feasibility test.
3. GenMolMF: Multi‐factor search (e.g. CNNaffinity, MolWt, SAS) with feasibility on all factors.

Example run:
    # Multi-factor search for DBH protein
    python GenMol_ZincGPT2.py --choice mf --protein DBH --target_size 5 --final_k 100
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
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

from search import Hypothesis, compute_Q, construct_file_paths
from get_mol_prop import compute_properties_with_affinity
from mol_utils import calculate_similarity, validate_smiles
from LMLFStar import generate_molecules_for_protein_multifactors_using_ZincGPT2


# =========================
# Helper: Environment Setup
# =========================
def setup_environment(protein, results_subdir, data_path="data", model_engine="gpt2_zinc_87m"):
    """
    Sets up common parameters and directories.
    Returns a dictionary with common parameters.
    """
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    labelled_file, unlabelled_file = construct_file_paths(data_path, protein)
    labelled_data = pd.read_csv(labelled_file).to_dict(orient="records")
    unlabelled_data = pd.read_csv(unlabelled_file).to_dict(orient="records")
    gnina_path = "./docking"
    config_path = f"./docking/{protein}/{protein}_config.txt"
    temp_dir = "/tmp/molecule_generation"
    output_dir = f"results_ZincGPT2/{results_subdir}/{protein}/{model_engine}/{date_time}"
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    return {
        "date_time": date_time,
        "labelled_data": labelled_data,
        "unlabelled_data": unlabelled_data,
        "model_engine": model_engine,
        "gnina_path": gnina_path,
        "config_path": config_path,
        "temp_dir": temp_dir,
        "output_dir": output_dir
    }


# ====================================
# Pipeline 1: GenMol1F (Single-Factor)
# ====================================
def GenMol1F(seed=0, protein="DBH", target_size=5, final_k=20, context=False, model_engine="gpt2_zinc_87m", plus_mode=False):
    """
    Single-factor search using ZincGPT2.
    NOTE: This is a placeholder implementation - you may need to adapt based on your specific needs.
    """
    print(f"GenMol1F not fully implemented for ZincGPT2. Running GenMolMF instead.")
    GenMolMF(seed=seed, protein=protein, target_size=target_size, final_k=final_k, 
             context=context, model_engine=model_engine)


# ====================================
# Pipeline 2: GenMolMF (Multi-Factor)
# ====================================
def GenMolMF(seed=0, protein="DBH", target_size=5, final_k=20, context=False, model_engine="gpt2_zinc_87m"):
    """
    Multi-factor search using ZincGPT2.
    The algorithm searches for optimal parameter ranges for multiple properties
    and uses ZincGPT2 to generate molecules that satisfy the constraints.
    """
    random.seed(seed)
    np.random.seed(seed)

    env = setup_environment(protein, "GenMolMF", model_engine=model_engine)
    labelled_data = env["labelled_data"]
    unlabelled_data = env["unlabelled_data"]
    gnina_path = env["gnina_path"]
    config_path = env["config_path"]
    temp_dir = env["temp_dir"]
    output_dir = env["output_dir"]

    def interleaved_LMLFStar_ZincGPT2(protein, labelled_data, unlabelled_data, initial_intervals,
                                      gnina_path, config_path, temp_dir, output_dir, 
                                      s=4, n=10, molecules_per_scaffold=20, final_k=100, 
                                      target_size=5, context=False):

        factors = [lambda x, p=param: x.get(p) for param in initial_intervals.keys()]
        e_0 = [initial_intervals[param] for param in initial_intervals]
        h_0 = Hypothesis(factors, e_0)

        theta_ext_h_default = len(unlabelled_data) / (len(labelled_data) + len(unlabelled_data))
        w_0 = compute_Q(h_0, "Background Knowledge", labelled_data, epsilon=0.1, theta_ext_h_approx=theta_ext_h_default)

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
                    if param == "CNNaffinity":  # keep max end fixed
                        new_intervals.append([float(quantiles[index]), float(initial_intervals[param][1])])
                    elif param in ["MolWt", "SAS"]:  # keep min end fixed
                        new_intervals.append([float(initial_intervals[param][0]), float(quantiles[index])])
                E_k.append(new_intervals)
            
            S = []
            for e in E_k:
                h_k = Hypothesis(factors, e)
                Q_k = compute_Q(h_k, "Background Knowledge", labelled_data, epsilon=0.1, theta_ext_h_approx=theta_ext_h_default)
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
                if abs(Q_k) < abs(best_w) * 0.8:
                    continue

                print(f"Evaluating node with interval {e_k} and Q-score {Q_k:.4f}")

                parameter_ranges = {param: e_k[i] for i, param in enumerate(initial_intervals.keys())}

                # Use ZincGPT2 to generate molecules
                generate_molecules_for_protein_multifactors_using_ZincGPT2(
                    protein=protein,
                    input_csv=f"data/{protein}.txt",
                    output_dir=output_dir,
                    gnina_path=gnina_path,
                    config_path=config_path,
                    temp_dir=temp_dir,
                    parameter_ranges=parameter_ranges,
                    target_size=target_size,
                    max_iterations=1,
                    molecules_per_scaffold=molecules_per_scaffold
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

        # Final generation with more molecules
        generate_molecules_for_protein_multifactors_using_ZincGPT2(
            protein=protein,
            input_csv=f"data/{protein}.txt",
            output_dir=output_dir,
            gnina_path=gnina_path,
            config_path=config_path,
            temp_dir=temp_dir,
            parameter_ranges=final_parameter_ranges,
            target_size=target_size,
            max_iterations=1,
            molecules_per_scaffold=final_k // 5  # Distribute across scaffolds
        )
        
        # Generate plots and logs (same as original)
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
        
    # Search parameters
    initial_intervals = {"CNNaffinity": [3, 10], "MolWt": [200, 700], "SAS": [0, 7.0]}
    search_params = {"s": 10, "n": 10, "molecules_per_scaffold": 20, "final_k": final_k, "context": context}
    
    interleaved_LMLFStar_ZincGPT2(protein=protein,
                                  labelled_data=labelled_data,
                                  unlabelled_data=unlabelled_data,
                                  initial_intervals=initial_intervals,
                                  gnina_path=gnina_path,
                                  config_path=config_path,
                                  temp_dir=temp_dir,
                                  output_dir=output_dir,
                                  s=search_params["s"],
                                  n=search_params["n"],
                                  molecules_per_scaffold=search_params["molecules_per_scaffold"],
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
    print("DONE [GenMolMF with ZincGPT2]")


# ================================
# Main: Parsing arguments and run
# ================================
def main():
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    
    print("="*70)
    print(f"   TARGET-SPECIFIC LEAD DISCOVERY USING ZincGPT2 [{date_time}]")
    print("="*70)
    
    parser = argparse.ArgumentParser(
        description="TARGET-SPECIFIC LEAD DISCOVERY USING ZincGPT2"
    )
    parser.add_argument("--choice", type=str, required=True,
                        help="Choice of pipeline: '1' (or '1f') for GenMol1F; '2' (or '1fplus') for GenMol1F with plus mode; '3' (or 'mf') for GenMolMF; '0' to abort")
    parser.add_argument("--protein", type=str, default="DBH", help="Target protein")
    parser.add_argument("--target_size", type=int, default=5, help="Target size for molecule generation")
    parser.add_argument("--context", type=str, default="False", help="Use context (True/False)")
    parser.add_argument("--model", type=str, default="gpt2_zinc_87m", help="Model engine to use")
    parser.add_argument("--final_k", type=int, default=20, help="Number of molecules to generate in the final step")
    args = parser.parse_args()
    
    context = args.context.lower() in ("true", "1", "yes")
    
    choice = args.choice.lower()
    print(args)

    if choice in ["1", "1f"]:
        print("Calling GenMol1F with ZincGPT2...")
        GenMol1F(seed=0, 
                 protein=args.protein, 
                 target_size=args.target_size, 
                 final_k=args.final_k, 
                 context=context, 
                 model_engine=args.model)
    elif choice in ["2", "1fplus"]:        
        print("Calling GenMol1F with plus mode and ZincGPT2...")
        GenMol1F(seed=0, 
                 protein=args.protein, 
                 target_size=args.target_size, 
                 final_k=args.final_k, 
                 context=context, 
                 model_engine=args.model,
                 plus_mode=True)
    elif choice in ["3", "mf"]:
        print("Calling GenMolMF with ZincGPT2...")
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