from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs, SanitizeMol, SanitizeFlags
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64
import os
import pandas as pd


def sanitize_smiles(smiles):
    """
    Create and sanitize a molecule from a SMILES string.

    Args:
        smiles (str): The SMILES string of the molecule.

    Returns:
        tuple: A tuple (mol, flag)
            - mol: RDKit Mol object if successful, None otherwise.
            - flag: A string describing the result or error.
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None, "Invalid SMILES: Parsing failed."
        Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL)
        #print(f"Molecule {smiles} sanitized successfully.")
        return mol, True
    except Exception as e:
        print(f"Sanitization failed: {e}")
        return None, False


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


def generate_structure(csv_file):
    """
    Generates an HTML file with molecular structures from a given CSV file containing SMILES.
    
    Args:
        csv_file (str): Path to the input CSV file.
    
    Returns:
        str: Path to the generated HTML file.
    """
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return None

    df = pd.read_csv(csv_file)

    if "SMILES" not in df.columns:
        print("Error: The CSV file must contain a 'SMILES' column.")
        return None

    def smiles_to_base64(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(400, 400)) 
                img_io = BytesIO()
                img.save(img_io, format="PNG")
                img_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")  
                return f'<img src="data:image/png;base64,{img_base64}" width="100">'
            return "Invalid SMILES"
        except Exception as e:
            print(f"Error with SMILES {smiles}: {e}")
            return "Error"

    df["Structure"] = df["SMILES"].apply(smiles_to_base64)

    output_dir = os.path.dirname(csv_file)  
    output_filename = os.path.splitext(os.path.basename(csv_file))[0] + ".html" 
    html_file_path = os.path.join(output_dir, output_filename)
    html_content = df.to_html(escape=False, index=False) 

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Molecular Structures</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h2>Molecular Structures Report</h2>
        {html_content}
    </body>
    </html>
    """

    with open(html_file_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"HTML file saved as {html_file_path}")
    return html_file_path


def show_results(result_dir):
    """
    Shows the summary statistics of the generated molecules.
    Creates two dataframe from the files in: results_path.
    df_gen: Generated molecules (in `all.csv` within the results directory)
    df_tanimoto: Tanimoto coefficients for the molecules
    """
    df_gen = pd.read_csv(f"{result_dir}/all.csv")
    df_tanimoto = pd.read_csv(f"{result_dir}/tanimoto.csv")
    print("CNNaffinity statistics:")
    print(f" Mean: {df_gen['CNNaffinity'].mean():.2f}")
    print(f" Median: {df_gen['CNNaffinity'].median():.2f}")
    print(f" Std: {df_gen['CNNaffinity'].std():.2f}")
    print(f" Min: {df_gen['CNNaffinity'].min():.2f}")
    print(f" Max: {df_gen['CNNaffinity'].max():.2f}")
    print("Tanimoto statistics:")
    print(f" Mean: {df_tanimoto['Avg. Similarity'].mean():.2f}")
    print(f" Median: {df_tanimoto['Avg. Similarity'].median():.2f}")
    print(f" Std: {df_tanimoto['Avg. Similarity'].std():.3f}")
    print(f" Min: {df_tanimoto['Avg. Similarity'].min():.2f}")
    print(f" Max: {df_tanimoto['Avg. Similarity'].max():.2f}")


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

