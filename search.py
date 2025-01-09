import pandas as pd
import random
import math
import numpy as np

# Seeding for reproducibility
random.seed(0)
np.random.seed(0)


class Hypothesis:
    def __init__(self, factors, experiment):
        self.factors = factors
        self.experiment = experiment

    def is_feasible(self, x):
        return all(self.experiment[i][0] <= f(x) <= self.experiment[i][1] for i, f in enumerate(self.factors))

    def __call__(self, x):
        return self.is_feasible(x)

def compute_Q(h, B, D, epsilon=0.1, ext_h_approx=0.5):
    E_plus = [e for e in D if e['Label'] == 1]
    E_minus = [e for e in D if e['Label'] == 0]
    TP_count = len([e for e in E_plus if h.is_feasible(e)])
    TN_count = len([e for e in E_minus if not h.is_feasible(e)])
    FPN_count = len(D) - (TP_count + TN_count)
    theta_ext_h = ext_h_approx
    if len(D) == 0:
        return float('-inf')
    Q = (
        math.log(epsilon) +
        TP_count * math.log((1 - epsilon) / theta_ext_h + epsilon) +
        TN_count * math.log((1 - epsilon) / (1 - theta_ext_h) + epsilon) +
        FPN_count * math.log(epsilon)
    )
    return Q


def SearchHypothesis(B, D, factors, init_intervals, s, n, ext_h_default=0.5):
    e_0 = init_intervals
    h_0 = Hypothesis(factors, e_0)
    w_0 = compute_Q(h_0, B, D)
    k = 1
    Q_values = [w_0]
    interval_history = [e_0]

    while k <= n:
        print(f"Iteration {k}: Current Interval {e_0} | Q-score {w_0:.4f}")

        quantiles = np.linspace(e_0[0][0], e_0[0][1], 5) 
        E_k = [
            [[random.uniform(quantiles[q], quantiles[q + 1]), e_0[0][1]]]
            for q in range(len(quantiles) - 1) for _ in range(s // 4)
        ]
        print(quantiles)
        print(E_k)

        S = [(compute_Q(Hypothesis(factors, e), B, D, ext_h_approx=ext_h_default), e) for e in E_k]
        #print(f"Sampled Intervals and Q-scores:\n{[(e, q) for q, e in S]}\n")

        (w_k, e_k) = max(S, key=lambda x: x[0])  # Pick the interval with the highest Q-score
        Q_values.append(w_k)
        interval_history.append(e_k)

        if w_k <= w_0 + 1e-2:
            print(f"\nNo significant improvement in Q-score: {w_k:.4f} <= {w_0:.4f}")
            break

        w_0 = w_k
        e_0 = e_k  # Update the interval for the next iteration
        k += 1

    final_interval = [float(e_0[0][0]), float(e_0[0][1])]  # Ensure final interval is a single range
    return Hypothesis(factors, [final_interval]), Q_values, interval_history


def SearchHypothesis1(B, D, factors, init_intervals, s, n, ext_h_default=0.5):    
    e_0 = init_intervals
    h_0 = Hypothesis(factors, e_0)
    w_0 = compute_Q(h_0, B, D)
    k = 1
    Q_values = [w_0]
    interval_history = [e_0]
    
    while k <= n:
        print(f"Iteration {k}: Current Interval {e_0} | Q-score {w_0:.4f}")

        quantiles = [np.linspace(e[0], e[1], 5) for e in e_0]
        E_k = [
                [
                    [random.uniform(quantile[q], quantile[q + 1]), e[1]]
                    for e, quantile in zip(e_0, quantiles)
                    for q in range(len(quantile) - 1)
                    ]
                for _ in range(s)
                ]
        
        # Old sampling approach
        #E_k = [
        #        [[e[0] + random.uniform(0, (e[1] - e[0]) / 2), e[1]] for e in e_0] for _ in range(s)
        #]

        # Quantile based sampling
       

        S = [(compute_Q(Hypothesis(factors, e), B, D, ext_h_approx=ext_h_default), e) for e in E_k]
        #print(f"Sampled Intervals and Q-scores:\n{[(e, q) for q, e in S]}\n")
        
        (w_k, e_k) = max(S, key=lambda x: x[0])
        Q_values.append(w_k)
        interval_history.append(e_k)
        if w_k <= w_0 + 1e-2:  # Allow for small tolerance
            print(f"\nNo significant improvement in Q-score: {w_k:.4f} <= {w_0:.4f}")
            break
        w_0 = w_k
        e_0 = e_k
        k += 1

    return Hypothesis(factors, interval_history[-1]), Q_values, interval_history

def main():
    labelled_data = pd.read_csv("data/JAK2_with_properties.txt").to_dict(orient="records")
    
    try:
        unlabelled_data = pd.read_csv("data/unlabelled_with_properties_JAK2binding_noNaN.txt").to_dict(orient="records")
        ext_h_default = len(unlabelled_data) / (len(labelled_data) + len(unlabelled_data))
        print(f"ext(h) value: {ext_h_default}")
    except FileNotFoundError:
        ext_h_default = 0.1

    B = "Background Knowledge"
    factor = lambda x: x['CNNaffinity']
    init_interval = [[3, 10]]
    s = 4 
    n = 10
    final_hypothesis, Q_values, interval_history = SearchHypothesis(B, labelled_data, [factor], init_interval, s, n, ext_h_default)

    print("\nSearch History:")
    print("-"*20)
    print("Iteration    Q-score")
    print("-"*20)
    for i, q in enumerate(Q_values):
        print(f"{i+1:<11}{q:.4f}")
    print("-"*20)

    print("\nFinal Hypothesis Interval:", interval_history[-1])

if __name__ == "__main__":
    main()

