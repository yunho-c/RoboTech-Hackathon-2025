# NOTE: currently doesn't work if `dataset` directory exists already. TODO: fix

import os
from roboflow import Roboflow


def move_dataset(path):
    script_dir = os.path.dirname(__file__)
    os.rename(
        path, os.path.join(script_dir, "dataset")
    )  # Move to dataset directory relative to script


if __name__ == "__main__":
    # source: https://universe.roboflow.com/test2-acixn/senior-project-zzhg3
    rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
    project = rf.workspace("yunhoflow").project("senior-project-zzhg3-iqrs6")
    version = project.version(1)
    dataset = version.download("yolov12", location="./dataset/Senior-Project")

    # print("Path to dataset files:", path) # DEBUG
