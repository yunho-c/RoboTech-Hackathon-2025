# NOTE: currently doesn't work if `dataset` directory exists already. TODO: fix

import kagglehub
import os


def move_dataset(path):
    script_dir = os.path.dirname(__file__)
    os.rename(
        path, os.path.join(script_dir, "dataset")
    )  # Move to dataset directory relative to script


if __name__ == "__main__":
    path = kagglehub.dataset_download(
        "dataclusterlabs/cardboard-object-detection"
    )  # cardboard dataset
    move_dataset(path)

    # print("Path to dataset files:", path) # DEBUG
