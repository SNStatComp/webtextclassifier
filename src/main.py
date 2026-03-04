import s3fs
import json
import random
import pandas as pd
import numpy as np
import datetime
import os 
import time
import re
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

    def has_strange_chars(s):
        if pd.isna(s):  # skip NaN
            return False
        try:
            # Try to decode the string (assuming it's a string)
            s.encode('utf-8').decode('utf-8')
            return False
        except UnicodeDecodeError:
            return True

    def has_no_strange_chars_regex(s):
        if pd.isna(s):
            return 0  # or handle NaN as needed
        # Match non-printable characters (control chars, etc.)
        strange_chars = re.findall(r'[\x00-\x08\x0B\x0E-\x1F\x7F]', str(s))
        return not (len(strange_chars) / len(s)) > 0.1 

    # Filter rows
    # TODO this should happenn in focused scrape
    df = df[df['content'].apply(has_no_strange_chars_regex)]
    df = df.iloc[:200]

    # Loading in (pre-trained) model(s)
    # TODO generalise embedding/prompting code to serve as generic interface 
    print("-" * 33, "CREATING/LOADING MODELS", "-" * 33)
    models = {}
    if config.llm.method == "embedding":
        models["embedding"] = create_model(config, "embedder")
        models["classification"] = create_model(config, "classifier")

    print("-" * 33, f"CLASSIFICATION ({config.llm.method})", "-" * 33)
    
    # Classification
    # We do this batch-wise due to compute constraints, see config for batch size
    # Create empty column and fill it up iteratively
    df["OJA_label"] = "0"
    start_row = 0
    start_time = time.time()
    while start_row < len(df.index):
        end_row = start_row + config.output.batchsize
        print(f"Working on rows: {start_row}:{min(end_row, len(df.index))} out of {len(df.index)}, iter/s={start_row / (time.time() - start_time)}", end="\r")
        if (start_row + config.output.batchsize) > len(df.index):
            selected_text = df.loc[df.index[start_row:], "content"].reset_index(drop=True)
            labels = classify(config, selected_text, models=models)
            df.loc[df.index[start_row:], "OJA_label"] = labels
        else:
            selected_text = df.loc[df.index[start_row:end_row], "content"].reset_index(drop=True)  
            labels = classify(config,  selected_text, models=models)
            df.loc[df.index[start_row:end_row], "OJA_label"] = labels
        start_row += config.output.batchsize

    print("-" * 33, "SAVE RESULTS", "-" * 33)
    # Save results in parquet
    print("df:", df.head(5))
    print("Total rows in df:", len(df.index))

    df.to_parquet(os.path.join(config.output.output_dir, f"output_{datetime.datetime.now()}.parquet"))


if __name__ == "__main__":
    main()
