import s3fs
import json
import random
import pandas as pd
import numpy as np
import datetime
import os 
import time
import re
import torch
from omegaconf import OmegaConf
from util import setup
from LLM import create_model, classify


def main():
    # Turn off torch grad to save resources
    torch.set_grad_enabled(False)
    
    # Load config
    print("-" * 33, "LOADING CONFIG", "-" * 33)
    config = setup("config/config.yaml")
    if config.seed is not None:
        random.seed(config.seed)

    print("Config:")
    print(OmegaConf.to_yaml(config))

    # Read text data
    # TODO lazy reading to deal with large dataset 
    print("-" * 33, "LOADING IN DATA", "-" * 33)
    df = pd.read_parquet(os.path.join(config.input.input_dir, config.input.input_file))
    df = df.iloc[:1]
    print("Total rows in df after read:", len(df.index))

    # Clean up input data
    def has_no_strange_chars_regex(s):
        if pd.isna(s) or len(s) == 0:
            return False  # or handle NaN as needed
        # Match non-printable characters (control chars, etc.)
        strange_chars = re.findall(r'[\x00-\x08\x0B\x0E-\x1F\x7F]', str(s))
        return not (len(strange_chars) / len(s)) > 0.1 

    # Filter rows
    print("Shape before filtering strange characters:", df.shape)
    df = df[df['content'].apply(has_no_strange_chars_regex)]
    print("Shape after filtering strange characters:", df.shape)

    # Deduplicate texts
    print("Shape before de-duplicate:", df.shape)
    df = df.drop_duplicates(subset="content")
    print("Shape after de-duplicate:", df.shape)

    # Loading in (pre-trained) model(s)
    models = {}
    if config.llm.method == "embedding":
        models["embedding"] = create_model(config, "embedder")
        models["classification"] = create_model(config, "classifier")

    print("-" * 33, f"CLASSIFICATION ({config.llm.method})", "-" * 33)
    # Classification
    # batch-wise due to compute constraints, see config for batch size
    # Create empty column and fill it up iteratively
    df["OJA_label"] = "-1"
    start_row = 0
    start_time = time.time()

    # We cannot embed texts that are too long due to GPU memory limitations
    def is_too_long(s):
        if pd.isna(s):
            return True
        return len(str(s)) > config.llm.embedding.max_nchar

    # Mark long rows and assign default label 0
    df["too_long"] = df["content"].apply(is_too_long)
    df.loc[df["too_long"], "OJA_label"] = "0"
    while start_row < len(df.index):
        end_row = min(start_row + config.batchsize, len(df.index))
        batch_idx = df.index[start_row:end_row]

        print(
            f"""
            Working on rows: {start_row}:{min(end_row, len(df.index))} out of {len(df.index)},
             iter/s={start_row / (time.time() - start_time)},
             mean nchar of text: {np.mean(df.loc[df.index[start_row:end_row], "content"].apply(len))}
            """, end="\r"
        )

        # Filter out long texts from this batch
        batch_df = df.loc[batch_idx]
        short_mask = ~batch_df["too_long"]
        short_idx = batch_df[short_mask].index

        # Classify remaining texts in batch
        if len(short_idx) > 0:
            selected_text = batch_df.loc[short_idx, "content"].reset_index(drop=True)
            # classify only short texts
            labels = classify(config, selected_text, models=models)

            # Write labels back to the right rows
            df.loc[short_idx, "OJA_label"] = labels

        # too_long rows in this batch already have label 0 set above
        start_row += config.batchsize

        torch.cuda.empty_cache()

    print("-" * 33, "SAVE RESULTS", "-" * 33)

    # Final logging
    print("df:", df.head(5))
    print("Total rows in df:", len(df.index))
    print("# Rows too long:", df["too_long"].value_counts())
    print("Value counts labels:", df["OJA_label"].value_counts())

    # Save output
    print(f"Writing output to: output_{datetime.datetime.now()}.parquet")
    df.to_parquet(os.path.join(config.output.output_dir, f"output_{datetime.datetime.now()}.parquet"))


if __name__ == "__main__":
    main()
