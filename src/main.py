import requests
import justext
import os
import s3fs
import json
import random
import pandas as pd
from util import setup, LLM_API
from datetime import datetime

config = setup("config/config.yaml")

# Create filesystem object
fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': config.filesystem.endpoint},
    key = config.aws.access_key,
    secret = config.aws.secret_access_key,
    token = config.aws.session_token
)
BUCKET = config.filesystem.bucket
FILE_KEY_S3 = config.filesystem.file_key
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3

# Read the URL dataframe containing URL and country code 
with fs.open(FILE_PATH_S3, mode="rb") as file_in:
    url_data = pd.read_csv(file_in, sep=",")

# Read the variables for which we want to see if this text contains the content
with open("input/variables.txt", "r", encoding="utf-8") as file:
    variables = file.readlines()

url_country = list(zip(url_data['url'], url_data['country']))
random.shuffle(url_country)

N = 10 # Sample size of list
url_country = url_country[:N]

for var in variables:
    output = {}
    print("Variable:", var)
    for url, country in url_country:
        try:
            response = requests.get(url)
        except:
            print("Could not request:", url)
            continue

        # TODO determine language by text content rather than domain
        language = "English"
        if country == "DE":
            language = "German"
        elif country == "NL":
            language = "Dutch"
        elif country == "PL":
            language = "Polish"
        elif country == "AT":
            language = "German"

        if not (response.status_code >= 200 and response.status_code < 400):
            # print(f"Non-viable response status for url: {url}, response: {response.status_code}")
            continue

        # TODO right now we extract text for every variable, but only needs to be done once
        extracted_paragraphs = justext.justext(response.content, justext.get_stoplist(language))
        extracted_text = "".join([paragraph.text for paragraph in extracted_paragraphs if (paragraph.cf_class in ["good", "neargood"] or paragraph.class_type in ["good", "neargood"])])

        if len(extracted_text) < 1:
            # print(f"Issue extracting text for url: {url}, language: {language}")
            continue
        
        response = LLM_API(config, var, extracted_text)
        output[url] = response

    print("# times LLM output exceeded specified maximum length")
    print(len([v for v in output.values() if len(v) > 3]))
    print(f"Total items: {len(output)}/{N}")

    with open(f"{config.output.output_dir}/output_{var.replace(" ", "_").lower()}_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.json", "w") as fp:
        json.dump(output, fp)

print("Done!")