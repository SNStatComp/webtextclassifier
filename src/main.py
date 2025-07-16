import s3fs
import json
import random
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from util import setup, extract_text
from LLM import classify, create_model
from sklearn.linear_model import LogisticRegression
from datetime import datetime


def main():
    config = setup("config/config.yaml")
    if config.seed is not None:
        random.seed(config.seed)

    print("Config:")
    print(OmegaConf.to_yaml(config))

    # TODO this assumes the use of Onyxia, change later to something more generic to read input file and variables
    # Create filesystem object
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': config.filesystem.endpoint},
        key=config.aws.access_key,
        secret=config.aws.secret_access_key,
        token=config.aws.session_token
    )
    BUCKET = config.filesystem.bucket
    FILE_KEY_S3 = config.input.input_file
    FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3

    # Read the URL dataframe containing URL and country code 
    with fs.open(FILE_PATH_S3, mode="rb") as file_in:
        url_data = pd.read_csv(file_in, sep=",")

    # Read the variables for which we want to see if this text contains the content
    with open(config.input.input_dir + "/" + config.input.input_variables, "r", encoding="utf-8") as file:
        variables = file.readlines()

    url_country = list(zip(url_data['url'], url_data['country']))
    random.shuffle(url_country)

    # For now, only use dutch websites
    url_country = [(url, country) for (url, country) in url_country if country == "NL"]

    # Only used for testing on a small sample to speed things up
    N = 100  # Sample size of list
    url_country = url_country[:N]

    models = None
    if config.llm.method == "embedding":
        model_embed = create_model(config)

        # TODO replace with code to read model config/load in pre-trained model
        model_class = LogisticRegression(max_iter=0)
        model_class.fit(np.random.rand(2, config.llm.embedding.dim), [0, 1])
        models = {
            "embed": model_embed,
            "class": model_class
        }

    for var in variables:
        output = {}
        print("Variable:", var)
        urls = []
        texts = []
        labels = []
        for i, uc_tuple in enumerate(url_country):
            url, country = uc_tuple

            urls.append(url)
            print(f"url {i + 1}/{len(url_country)}: {url}")
            try:
                extracted_text = extract_text(config, url, country)
            except Exception as e:
                texts.append("")
                labels.append("-1")
                print(e)
                continue

            if len(extracted_text) < 1:
                # print(f"Issue extracting text for url: {url}, language: {language}")
                texts.append("")
                labels.append("-1")
                continue

            texts.append(extracted_text)
            response = classify(config, extracted_text, var, models)
            labels.append(response)
            output[url] = response

        df_dict = {
            "url": urls,
            "text": texts,
            "label": labels
        }

        df = pd.DataFrame(data=df_dict)
        df = df[df["label"] != "-1"]
        df.to_parquet(f"output_trial_{var}_{config.llm.method}.parquet", index=False)
        print("# of 1s:", sum((df["label"]).astype(int)))
        # print("# times LLM output exceeded specified maximum length")
        # print(len([v for v in output.values() if len(v) > 3]))
        print(f"Total items: {len(output)}/{len(url_country)}")

        print(output)

        with open(f"{config.output.output_dir}/output_{var.replace(" ", "_").lower()}_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.json", "w") as fp:
            json.dump(output, fp)

    print("Done!")


if __name__ == "__main__":
    main()
