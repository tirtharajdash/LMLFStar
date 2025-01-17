from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
import pandas as pd

def calculate_similarity(target_smiles_list, input_smiles_list):
    """
    Calculate Jaccard (Tanimoto) similarities between input SMILES and a target database of SMILES,
    returning a DataFrame with similarity scores for each pair and average similarity.

    Args:
        target_smiles_list (list of str): A list of target SMILES strings to compare against (the database).
        input_smiles_list (list of str): A list of input SMILES strings (or a single SMILES) to calculate similarities for.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - "Mol": The SMILES from input_smiles_list.
            - Columns for each SMILES in target_smiles_list, with similarity scores.
            - "Avg. Similarity": The average similarity score for each input SMILES across all target SMILES.
        None: If either argument is missing or empty.
    """
    if not target_smiles_list or not input_smiles_list:
        print("Error: Both 'target_smiles_list' and 'input_smiles_list' must be provided and non-empty.")
        return None

    def smiles_to_fingerprint(smiles):
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return generator.GetFingerprint(mol)
        return None

    def validate_smiles(smiles_list):
        return [smiles for smiles in smiles_list if Chem.MolFromSmiles(smiles)]

    target_smiles_list = validate_smiles(target_smiles_list)
    input_smiles_list = validate_smiles(input_smiles_list)

    target_fps = {smiles: smiles_to_fingerprint(smiles) for smiles in target_smiles_list}
    target_fps = {k: v for k, v in target_fps.items() if v}

    input_fps = {smiles: smiles_to_fingerprint(smiles) for smiles in input_smiles_list}
    input_fps = {k: v for k, v in input_fps.items() if v}

    results = []
    for input_smiles, input_fp in input_fps.items():
        row = {"Mol": input_smiles}
        similarities = []
        for target_smiles, target_fp in target_fps.items():
            similarity = DataStructs.TanimotoSimilarity(input_fp, target_fp)
            row[target_smiles] = similarity
            similarities.append(similarity)
        row["Avg. Similarity"] = sum(similarities) / len(similarities) if similarities else 0
        results.append(row)

    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    target_smiles_list = ["CCO", "CCN", "CCCC", "c1=cc=cc=c1"]
    input_smiles_list = ["CCO", "ccn"]
    df = calculate_similarity(target_smiles_list, input_smiles_list)
    print(df)
    if df is not None:
        mol = "CCO"
        avg_similarity = df.loc[df["Mol"] == mol, "Avg. Similarity"].values
        if avg_similarity.size > 0:
            avg_similarity = avg_similarity[0]
            print(f"Average Similarity for {mol}: {avg_similarity}")

