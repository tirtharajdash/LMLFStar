import os

def load_api_key(file_path="safe/api.key"):
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise Exception(f"API key file '{file_path}' not found. Please make sure it exists.")

if __name__ == "__main__":
    print(f"My API Key: {load_pei_key()}")

