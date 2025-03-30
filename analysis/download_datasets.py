import kagglehub
import os

if __name__ == "__main__":
    path = kagglehub.dataset_download("dataclusterlabs/cardboard-object-detection") # cardboard dataset
    script_dir = os.path.dirname(__file__)
    os.rename(path, os.path.join(script_dir, "dataset")) # Move to dataset directory relative to script

    print("Path to dataset files:", path) # DEBUG

