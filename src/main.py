import s3fs
import json
import random
import pandas as pd
import numpy as np
import os 
from omegaconf import OmegaConf
from util import setup
from LLM import create_model, classify


def main():
    config = setup("config/config.yaml")
    if config.seed is not None:
        random.seed(config.seed)

    print("-" * 33, "LOADING CONFIG", "-" * 33)
    print("Config:")
    print(OmegaConf.to_yaml(config))

    # Read text data
    print("-" * 33, "LOADING IN DATA", "-" * 33)
    # TODO lazy reading to deal with large dataset 
    df = pd.read_parquet(os.path.join(config.input.input_dir, config.input.input_file))

    print("df:", df)

    # Loading in (pre-trained) model(s)
    # TODO generalise embedding/prompting code to serve as generic interface 
    print("-" * 33, "CREATING/LOADING MODELS", "-" * 33)
    models = {}
    if config.llm.method == "embedding":
        models["embedding"] = create_model(config, "embedder")
        models["classification"] = create_model(config, "classifier")

    # print("models:", models)
    print("-" * 33, "CLASSIFICATION (WITH EMBEDDING)", "-" * 33)
    # Classification (with embedding in this case)
    print("Start embedding and classification...")

    df["OJA_label"] = classify(config, df["text"], models=models)
    print("-" * 33, "SAVE RESULTS", "-" * 33)
    # Save results in parquet
    print("df:", df)
    df.to_parquet(os.path.join(config.output.output_dir, "output.parquet"))


if __name__ == "__main__":
    main()
