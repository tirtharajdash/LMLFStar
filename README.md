### LMLFStar: Generation of Target-specific Novel Lead Molecules using an LLM

LMLFStar is a molecular generation and optimization framework that uses a large language model (LLM) to discover potential lead compounds for target proteins. It employs an iterative search strategy using a Q-heuristic to explore the chemical property space efficiently.

#### Preprint and Publication

Submitted to bioRxiv. URL will be updated shortly.

#### Features
`GenMol.py` is an interleaved implementation of `LMLFStar.py`. The results in the paper are based on this.

- *Multiple Search Pipelines*:
  - `GenMol1F`: Single-factor optimization based on CNNaffinity.
  - `GenMol1Fplus`: Extended feasibility testing, including molecular weight and SAS constraints.
  - `GenMolMF`: Multi-factor search considering multiple molecular properties.
- *Dynamic Search Process*:
  - Interval-based hypothesis exploration with adaptive refinement.
  - Feasibility testing based on predefined constraints.
  - Iterative selection based on Q-score calculations.
- *Result Visualization*:
  - Detailed logging of each iteration.
  - Automated plotting of search progress (Q-score vs. iterations).
- *Customizable Parameters*:
  - Adjustable target size, model engine, feasibility thresholds, and search iterations.

#### Repository Structure
```
LMLFStar/
├── data/             # Contains input molecule datasets
├── docking/          # Configuration files for molecular docking
├── env_utils.py      # Utility functions for environment setup
├── env.yml           # Conda environment file
├── GenMol.py         # Main script for running molecule generation pipelines
├── get_mol_prop.py   # Script for computing molecular properties
├── legacy/           # Older scripts and references
├── LICENSE           # License information
├── LMLFStar.py       # Core functions for molecule generation
├── mol_utils.py      # Utilities for molecular processing
├── README.md         # Documentation for the repository
├── results/          # Stores generated molecules and search results
├── run.sh            # Shell script to execute the pipeline
├── safe/             # Backup or checkpointed files
├── search.py         # Implements the hypothesis-driven search strategy
├── unit_test.ipynb   # Jupyter Notebook for testing components
├── Tree.ipynb        # Analysis of the generated molecule search tree
└── nohup.out         # Log output from nohup execution
```

#### Installation

```bash
conda env create -f env.yml
conda activate chem
```

Additionally, you will need the `gnina` software for docking. The current implementation uses the official version v1.3: [GNINA v1.3](https://github.com/gnina/gnina/releases/tag/v1.3).

#### Usage

```bash
python GenMol.py --protein DBH --target_size 5 --choice mf --context True --model gpt-4o --final_k 100
```
##### Arguments:
| Argument       | Description |
|---------------|-------------|
| `--protein`   | Target protein name (e.g., `DBH`) |
| `--target_size` | Number of target molecules to reveal to LLM per iteration |
| `--choice` | Selects the pipeline (1: `1f`, 2: `1fplus`, 3: `mf`) |
| `--context` | Enables context-based molecule generation (True/False) |
| `--model` | LLM model used (e.g., `gpt-4o`) |
| `--final_k` | Number of molecules in the final generation step after search is complete |

see `run.sh` for a batch run.

#### Example Runs

##### Run GenMol1F (Single-factor search):
```bash
python GenMol.py --protein DBH --target_size 5 --choice 1 --context False --model gpt-4o --final_k 10
```
##### Run GenMolMF (Multi-factor search):
```bash
python GenMol.py --protein DBH --target_size 5 --choice 3 --context True --model gpt-4o --final_k 10
```

#### Function Call Structure

Below is a simplified structure of how the functions call each other:
```
GenMol.py
├── main()
    ├── Parses command-line arguments
    ├── Calls the appropriate pipeline:
        ├── GenMol1F()
        │   ├── setup_environment()
        │   ├── interleaved_LMLFStar()
        │       ├── Hypothesis()
        │       ├── compute_Q()
        │       ├── generate_molecules_for_protein()
        │       ├── generate_molecules_for_protein_with_context()
        │       ├── Logging and results handling
        ├── GenMolMF()
        │   ├── setup_environment()
        │   ├── interleaved_LMLFStar()
        │       ├── Hypothesis()
        │       ├── compute_Q()
        │       ├── generate_molecules_for_protein_multifactors()
        │       ├── generate_molecules_for_protein_multifactors_with_context()
        │       ├── Logging and results handling
```

### Contact
For any questions or contributions, feel free to raise an issue or submit a pull request.

### References
Some relevant sources and references for LMLF:
1. LMLF codebase: [LMLF](https://github.com/Shreyas-Bhat/LMLF)
2. LMLF paper: [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/27751)
3. McCreath and Sharma, LIME: [ALT 1998](https://doi.org/10.1007/3-540-49730-7_25)

