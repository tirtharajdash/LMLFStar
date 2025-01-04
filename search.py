import random
import math

# Define the Hypothesis class
class Hypothesis:
    def __init__(self, factors, experiment):
        """
        Initialize with factors and experiments.
        """
        self.factors = factors  # List of factor functions (f1, ..., fn)
        self.experiment = experiment  # List of intervals (i1, ..., in)

    def is_feasible(self, x):
        """
        Check if x satisfies the feasibility constraint.
        """
        # Constraint(phi(x)) checks all factors against their intervals
        constraint = all(f(x) in i for f, i in zip(self.factors, self.experiment))
        return constraint

    def __call__(self, x):
        """
        Callable method to evaluate feasibility of instance x.
        """
        return self.is_feasible(x)

# Define the Q function
def compute_Q(h, B, D, epsilon=0.1):
    """
    Compute the Q-score for a hypothesis h given B (background knowledge) and D (dataset).
    
    Parameters:
        h: Hypothesis (callable object with is_feasible method)
        B: Background knowledge (unused in this implementation, can be a predicate)
        D: Dataset (D = E+ U E−, where E+ is positive examples and E− is negative examples)
        epsilon: Probability of random assignment to positive or negative examples.
        
    Returns:
        Q-score as a float.
    """
    E_plus = [e for e in D if e['label'] == 1]
    E_minus = [e for e in D if e['label'] == 0]

    # True Positives
    TP = [e for e in E_plus if h.is_feasible(e['data'])]
    TP_count = len(TP)

    # True Negatives
    TN = [e for e in E_minus if not h.is_feasible(e['data'])]
    TN_count = len(TN)

    # False Positives and Negatives
    FPN_count = len(D) - (TP_count + TN_count)

    # Extent of hypothesis
    X = [e['data'] for e in D]
    ext_h = [x for x in X if h.is_feasible(x)]
    theta_ext_h = len(ext_h) / len(X) if len(X) > 0 else 0

    # Compute Q score
    if len(D) == 0:
        return float('-inf')

    Q = (
        math.log(epsilon) +
        TP_count * math.log((1 - epsilon) / theta_ext_h + epsilon) +
        TN_count * math.log((1 - epsilon) / (1 - theta_ext_h) + epsilon) +
        FPN_count * math.log(epsilon)
    )
    return Q

# Define the SearchHypothesis function
def SearchHypothesis(B, D, F, Theta, s, n):
    """
    Implements the SearchHypothesis procedure with updated definitions.
    """
    e_0 = Theta  # Initial factor intervals
    h_0 = Hypothesis(F, e_0)  # Initial hypothesis
    w_0 = compute_Q(h_0, B, D)  # Initial Q-score
    k = 1
    Done = (k > n)

    while not Done:
        # Generate a random sample of s intervals subsumed by e_{k-1}
        E_k = [random.sample(e_0, len(e_0)//2) for _ in range(s)]

        # Generate hypotheses and their Q-scores
        S = [(compute_Q(Hypothesis(F, e), B, D), e) for e in E_k]

        # Find the hypothesis with the highest score
        (w_k, e_k) = max(S, key=lambda x: x[0])

        # Check termination conditions
        Done = ((w_k <= w_0) or (k == n))
        k += 1
    
    return Hypothesis(F, e_k)

# Example usage
if __name__ == "__main__":
    # Background knowledge (can be predicates or ignored for now)
    B = "Background Knowledge"

    # Dataset: list of instances with labels
    D = [
        {'data': [6, 5], 'label': 1},  # Positive example
        {'data': [4, 7], 'label': 0},  # Negative example
        {'data': [7, 6], 'label': 1},  # Positive example
        {'data': [3, 8], 'label': 0},  # Negative example
    ]

    # Factors (controllable features)
    F = [
        lambda x: x[0],  # Affinity
        lambda x: x[1],  # SynthesisSteps
    ]

    # Initial intervals for factors
    Theta = [[5.5, 8.0], [4, 8]]  # Example: intervals for Affinity and SynthesisSteps

    # Number of samples and steps
    s = 5  # Number of interval-vectors to sample
    n = 100  # Upper bound on steps

    # Run search
    final_hypothesis = SearchHypothesis(B, D, F, Theta, s, n)
    print("Final Hypothesis Feasibility Check:", final_hypothesis([6, 5]))

