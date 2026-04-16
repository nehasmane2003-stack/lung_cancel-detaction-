import os

def check_model_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")