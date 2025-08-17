import pandas as pd
import argparse
from datetime import datetime
import random
import math
import numpy as np

# Seeding for reproducibility
random.seed(0)
np.random.seed(0)


def construct_file_paths(data_path, protein):
    labelled_file = f"{data_path}/{protein}_with_properties_{protein}binding.txt"
    #unlabelled_file = f"{data_path}/unlabelled_with_properties_{protein}binding_noNaN.txt"
    unlabelled_file = f"{data_path}/chembl1K_with_properties_{protein}binding.txt"
    return labelled_file, unlabelled_file


class Hypothesis:
    def __init__(self, factors, experiment):
        self.factors = factors
        self.experiment = experiment

    def is_feasible(self, x):
        for i, f in enumerate(self.factors):
            value = f(x)
            lower_bound, upper_bound = self.experiment[i]
            #print(f"Factor {i}, Value = {value}, Range = {self.experiment[i]}")
            if value is None:
                return False
            if lower_bound is not None and value < lower_bound:
                return False
            if upper_bound is not None and value > upper_bound:
                return False
        return True

    def __call__(self, x):
        return self.is_feasible(x)


def compute_Q_terms(TP_count, TN_count, FPN_count, theta_ext_h, epsilon):
    """
    Compute Q‐score components:
        - TP_count * log((1 - epsilon)/theta + epsilon)
        - TN_count * log((1 - epsilon)/(1 - theta) + epsilon)
        - FPN_count * log(epsilon)  (only if epsilon > 0)
        - log(epsilon) prior term  (only if epsilon > 0)
    """
    tp_term = TP_count * math.log((1 - epsilon) / theta_ext_h     + epsilon)
    tn_term = TN_count * math.log((1 - epsilon) / (1 - theta_ext_h) + epsilon)

    if epsilon > 0:
        print(f"[search.py] Noisy Q calculation (epsilon = {epsilon}).")
        prior_term = math.log(epsilon)
        fpn_term   = FPN_count * math.log(epsilon)
    else:
        print(f"[search.py] Noise-free Q calculation (epsilon = {epsilon}).")
        prior_term = 0.0
        fpn_term   = 0.0

    return prior_term + tp_term + tn_term + fpn_term


def compute_Q(h, B, D, epsilon=0.1, theta_ext_h_approx=0.5):
    """
    Fixed-theta variant of Q:
      - theta_ext_h is taken as theta_ext_h_approx (user passes; or default)
      - constant math.log(epsilon) offsets every hypothesis equally
    """
    if not D:
        return float('-inf')

    E_plus  = [e for e in D if e['Label'] == 1]
    E_minus = [e for e in D if e['Label'] == 0]

    TP = sum(1 for e in E_plus  if h.is_feasible(e))
    TN = sum(1 for e in E_minus if not h.is_feasible(e))
    FPN = len(D) - (TP + TN)

    theta_ext_h = theta_ext_h_approx

    return compute_Q_terms(TP, TN, FPN, theta_ext_h, epsilon)


def compute_Q_data(h, B, D, epsilon=0.1, theta_ext_h_approx=None):
    """
    Data-driven theta variant of Q:
      - theta_ext_h = (|ext n D| + 1) / (|D| + 2)  # Laplace smoothing
      - uses the same TP, TN, FPN counts
      
    Note. This function is not used in the paper.
    """
    if not D:
        return float('-inf')

    E_plus  = [e for e in D if e['Label'] == 1]
    E_minus = [e for e in D if e['Label'] == 0]

    TP = sum(1 for e in E_plus  if h.is_feasible(e))
    TN = sum(1 for e in E_minus if not h.is_feasible(e))
    FPN = len(D) - (TP + TN)

    # empirical coverage
    neg_cov = len(E_minus) - TN
    ext_cov = TP + neg_cov

    # Laplace smoothing to avoid θ=0 or 1
    theta_ext_h = (ext_cov + 1) / (len(D) + 2)

    return compute_Q_terms(TP, TN, FPN, theta_ext_h, epsilon)


def SearchHypothesis(B, D, factors, init_intervals, s, n, theta_ext_h_default=0.5):
    e_0 = init_intervals
    h_0 = Hypothesis(factors, e_0)
    w_0 = compute_Q(h_0, B, D)
    k = 1
    Q_values = [w_0]
    interval_history = [e_0]

    while k <= n:
        print(f"\nIteration {k}: Current Interval {e_0} | Q-score {w_0:.4f}\n")

        quantiles = list(map(float, np.linspace(e_0[0][0], e_0[0][1], 5)))
        E_k = [
            [[random.uniform(quantiles[q], quantiles[q + 1]), e_0[0][1]]]
            for q in range(len(quantiles) - 1) for _ in range(s // 4)
        ]
        S = [(compute_Q(Hypothesis(factors, e), B, D, theta_ext_h_approx=theta_ext_h_default), e) for e in E_k]
        
        print(f"Quantile\t\t: {quantiles}")
        print(f"Sampled invervals\t: {E_k}")
        print(f"Q-scores\t\t: {[q for q, e in S]}\n")

        (w_k, e_k) = max(S, key=lambda x: x[0]) 
        Q_values.append(w_k)
        interval_history.append(e_k)

        #if w_k <= w_0 - 1e2: #check for absolute change
        if (w_0 - w_k) / w_0 >= 0.10: #check for percentage change
            print(f"\nNo significant improvement in Q-score: {w_k:.4f} <= {w_0:.4f}")
            break

        w_0 = w_k
        e_0 = e_k
        k += 1

    final_interval = [float(e_0[0][0]), float(e_0[0][1])]  
    return Hypothesis(factors, [final_interval]), Q_values, interval_history


def main(labelled_file, unlabelled_file):
    labelled_data = pd.read_csv(labelled_file).to_dict(orient="records")
    
    try:
        unlabelled_data = pd.read_csv(unlabelled_file).to_dict(orient="records")
        theta_ext_h_default = len(unlabelled_data) / (len(labelled_data) + len(unlabelled_data))
        print(f"\nUsing computed theta(ext(h)) value: {theta_ext_h_default}")
    except FileNotFoundError:
        theta_ext_h_default = 0.1
        print(f"\nUsing default theta(ext(h)) value: {theta_ext_h_default}")

    B = "Background Knowledge"  #This is used just to stay consistent with the pseudocode (mainly, property computing codes)
    factor = lambda x: x['CNNaffinity']
    init_interval = [[3, 10]]
    s = 4
    n = 10
    
    final_hypothesis, Q_values, interval_history = SearchHypothesis(B, labelled_data, [factor], init_interval, s, n, theta_ext_h_default)

    print("\nSearch History:")
    print("-" * 20)
    print("Iteration    Q-score")
    print("-" * 20)
    for i, q in enumerate(Q_values):
        print(f"{i+1:<11}{q:.4f}")
    print("-" * 20)

    print("\nFinal Hypothesis Interval:", interval_history[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search Hypothesis Program: Pass the protein name to process associated files.")

    parser.add_argument(
        "--protein",
        type=str,
        help="The protein name (e.g., JAK2, DRD2, DBH).",
        required=False
    )
    
    args = parser.parse_args()
    
    data_path = "data"
    protein = args.protein if args.protein else None

    if not protein:
        print("No protein specified! Here's how to use the script:")
        print("\nUsage: python search.py --protein <PROTEIN_NAME>")
        print("Example: python search.py --protein JAK2")
        print("Available proteins: JAK2, DRD2, DBH\n")
        exit(1) 

    labelled_file, unlabelled_file = construct_file_paths(data_path, protein)

    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print("*" * 61)
    print(f" PROGRAM: SEARCH_HYPOTHESIS (TIMESTAMP: {now})")
    print(f" PROTEIN: {protein}")
    print("*" * 61)

    print(f"Labelled data   : {labelled_file}")
    print(f"Unlabelled data : {unlabelled_file}")

    main(labelled_file, unlabelled_file)
