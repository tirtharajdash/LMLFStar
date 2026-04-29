#!/usr/bin/env python3
"""
GenMol_claude.py: Claude-based variant of GenMolMF.

Note: only --context True is supported here; the no-context Claude generator
(`generate_molecules_for_protein_multifactors_claude`) is not implemented in
LMLFStar.py. GenMol1F / 1Fplus are not implemented either; use --choice mf.

Example run:
    python GenMol_claude.py --protein DBH --target_size 5 --choice mf --context True \\
        --model claude-3-5-sonnet-20241022 --final_k 100
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

from env_utils import load_anthropic_api_key
from search import Hypothesis, compute_Q, construct_file_paths
from LMLFStar import (
    generate_molecules_for_protein_multifactors_with_context_claude
)


# =========================
# Helper: Environment Setup
# =========================
def setup_environment(protein, results_subdir, data_path="data",
                      model_engine="claude-3-5-sonnet-20241022"):
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    labelled_file, unlabelled_file = construct_file_paths(data_path, protein)
    labelled_data = pd.read_csv(labelled_file).to_dict(orient="records")
    unlabelled_data = pd.read_csv(unlabelled_file).to_dict(orient="records")
    api_key = load_anthropic_api_key()
    gnina_path = "./docking"
    config_path = f"./docking/{protein}/{protein}_config.txt"
    temp_dir = "/tmp/molecule_generation"
    output_dir = f"results_claude/{results_subdir}/{protein}/{model_engine}/{date_time}"
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
# Pipeline 1: GenMol1F (deprecated)
# ====================================
def GenMol1F(seed=0, protein="DBH", target_size=5, final_k=20,
             context=False, model_engine="claude-3-5-sonnet-20241022", plus_mode=False):
    raise NotImplementedError(
        "GenMol1F (and 1Fplus) are not implemented for the Claude pipeline. "
        "Use GenMolMF (--choice mf)."
    )


# ====================================
# Pipeline 2: GenMolMF (Multi-Factor)
# ====================================
def GenMolMF(seed=0, protein="DBH", target_size=5, final_k=20, context=False,
             model_engine="claude-3-5-sonnet-20241022"):
    """
    Multi-factor search (Claude). Only context=True is supported.
    """
    if not context:
        raise NotImplementedError(
            "Non-context Claude generation is not implemented "
            "(no `generate_molecules_for_protein_multifactors_claude` in LMLFStar.py). "
            "Pass --context True."
        )

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
                             output_dir, s=4, n=10, max_samples=5, final_k=100,
                             target_size=5, context=False):

        param_names = list(initial_intervals.keys())
        factors = [lambda x, p=param: x.get(p) for param in param_names]
        e_0 = [list(initial_intervals[param]) for param in param_names]
        h_0 = Hypothesis(factors, e_0)

        theta_ext_h_default = len(unlabelled_data) / (len(labelled_data) + len(unlabelled_data))
        w_0 = compute_Q(h_0, "Background Knowledge", labelled_data,
                        epsilon=0.1, theta_ext_h_approx=theta_ext_h_default)
        best_w = w_0

        patience = 3
        patience_counter = 0
        iteration_numbers = []
        current_Q_history = []
        best_Q_history = []
        interval_history = [e_0]
        Q_values = [w_0]
        w_values = [w_0]
        search_tree = []
        intermediate_csv = os.path.join(output_dir, "intermediate.csv")
        intermediate_data = []

        k = 1
        while k <= n:
            # LHS seeding: per-iteration seed; previously identical samples every round.
            lhs_samples = scipy.stats.qmc.LatinHypercube(
                d=len(param_names), seed=seed + k
            ).random(n=s)

            E_k = []
            for sample in lhs_samples:
                new_intervals = []
                # adaptive narrowing: build candidates from current best e_0,
                # not the original initial_intervals.
                for i, param in enumerate(param_names):
                    lo, hi = e_0[i][0], e_0[i][1]
                    quantiles = np.linspace(lo, hi, s + 1)
                    index = min(max(int(sample[i] * s), 0), s - 1)
                    if param == "CNNaffinity":
                        new_intervals.append([float(quantiles[index]), float(hi)])
                    elif param in ["MolWt", "SAS"]:
                        new_intervals.append([float(lo), float(quantiles[index])])
                    else:
                        new_intervals.append([float(lo), float(hi)])
                E_k.append(new_intervals)

            S = []
            for e in E_k:
                h_k = Hypothesis(factors, e)
                Q_k = compute_Q(h_k, "Background Knowledge", labelled_data,
                                epsilon=0.1, theta_ext_h_approx=theta_ext_h_default)
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
            prev_best_w = best_w 
            feasible_node_found = False
            w_k = 0

            for (Q_k, e_k) in sorted_S:
                if Q_k < best_w * 0.8:
                    continue

                print(f"Evaluating node with interval {e_k} and Q-score {Q_k:.4f}")
                parameter_ranges = {param: e_k[i] for i, param in enumerate(param_names)}

                generate_molecules_for_protein_multifactors_with_context_claude(
                    protein=protein,
                    input_csv=f"data/{protein}.txt",
                    output_dir=output_dir,
                    api_key=api_key,
                    model_name=model_engine,
                    gnina_path=gnina_path,
                    config_path=config_path,
                    temp_dir=temp_dir,
                    parameter_ranges=parameter_ranges,
                    target_size=target_size, 
                    max_iterations=1,
                    max_samples=max_samples
                )

                gen_csv = f"{output_dir}/generated.csv"
                if os.path.exists(gen_csv) and os.path.getsize(gen_csv) > 0:
                    properties_df = pd.read_csv(gen_csv)
                    for param, bounds in parameter_ranges.items():
                        properties_df = properties_df[
                            (properties_df[param] >= bounds[0]) &
                            (properties_df[param] <= bounds[1])
                        ]

                    if len(properties_df) > 0:
                        print(f"  Feasible molecules found in interval {e_k} with Q-score {Q_k:.4f}.")
                        best_w = max(best_w, Q_k)
                        w_k = Q_k
                        w_0 = Q_k
                        e_0 = e_k
                        feasible_node_found = True

                        new_data = properties_df.to_dict(orient="records")
                        intermediate_data.extend(new_data)
                        interm_df = pd.DataFrame(intermediate_data).drop_duplicates()
                        for param, bounds in zip(param_names, e_0):
                            interm_df = interm_df[
                                (interm_df[param] >= bounds[0]) &
                                (interm_df[param] <= bounds[1])
                            ]
                        intermediate_data = interm_df.to_dict(orient="records")
                        break
                    else:
                        print(f"  Generated molecules but none in interval {e_k}.")
                else:
                    print(f"  No molecules generated for interval {e_k}.")

            if not feasible_node_found:
                print("No feasible candidate nodes found that meet the threshold. Ending search.")
                break

            # patience: track at iteration level
            if best_w > prev_best_w + 1e-9:
                patience_counter = 0
            else:
                patience_counter += 1

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
        final_parameter_ranges = {param: interval_history[-1][i] for i, param in enumerate(param_names)}

        generate_molecules_for_protein_multifactors_with_context_claude(
            protein=protein,
            input_csv=f"data/{protein}.txt",
            output_dir=output_dir,
            api_key=api_key,
            model_name=model_engine,
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

        log_lines = ["Search Tree:"]
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

    initial_intervals = {"CNNaffinity": [3, 10], "MolWt": [200, 700], "SAS": [0, 7.0]}
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
    print("=" * 63)
    print(f"   TARGET-SPECIFIC LEAD DISCOVERY USING AN LLM (Claude) [{date_time}]")
    print("=" * 63)
    parser = argparse.ArgumentParser(
        description="TARGET-SPECIFIC LEAD DISCOVERY USING AN LLM (Claude)"
    )
    parser.add_argument("--choice", type=str, required=True,
                        help="Pipeline: '3' or 'mf' for GenMolMF; '0' to abort. "
                             "('1'/'1f' and '2'/'1fplus' are not implemented for Claude.)")
    parser.add_argument("--protein", type=str, default="DBH", help="Target protein")
    parser.add_argument("--target_size", type=int, default=5, help="Target size for molecule generation")
    parser.add_argument("--context", type=str, default="False", help="Use context (True/False); only True is supported")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022", help="Claude model to use")
    parser.add_argument("--final_k", type=int, default=20, help="Number of molecules to generate in the final step")
    args = parser.parse_args()

    context = args.context.lower() in ("true", "1", "yes")
    choice = args.choice.lower()
    print(args)

    if choice in ["1", "1f", "2", "1fplus"]:
        print(f"Choice '{args.choice}' is not implemented for the Claude pipeline. Use --choice mf.")
        return 1
    elif choice in ["3", "mf"]:
        print("Calling GenMolMF (Claude) ...")
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
