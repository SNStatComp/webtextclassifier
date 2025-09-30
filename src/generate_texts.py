import openai
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


config = setup("config/config.yaml")


client = openai.OpenAI(
    api_key=config.llm.prompt.api_key,
    base_url=config.llm.prompt.api_url
)

prompt_content = f"""
Generate a text that you would find on a random web page. It should be a job vacancy post. Make sure to have be as creative as you can in the range of jobs, companies, industry, job, expertise-level. As if your text truly was a random sample out of ALL job vacancies you would find online. The only additional criteria is that it is not a small company (<10 employees) posting the job vacancy.
  I do not mean that your response should contain all of these texts, but rather that these are examples of texts I would expect from your response, 
  your response being one of them. 
Make sure to be creative in what instance the website belongs to. The length of the text should thus be what you 
could find on a web page. The web-text should be in Dutch. Make sure every part of the web page is diverse, and unique. For example, a previous attempt had every job be at senior level, and nearly all related to tech industries.
"""


N = 100
for i in range(N):
    print(f"{i+1}/{N}")
    response = client.chat.completions.create(
        model = config.llm.prompt.model,
        messages = [
            {"role":"system", "content":"You are someone that creates online Job Vacancy web-texts. Your prompt answer should just be the text, no additional text needed. Do NOT respond in markdown, but just plain text as if you've extracted the web page text."},
            {"role":"user", "content": prompt_content},
        ],
        temperature = 2.0,
        max_completion_tokens = 4096,
        max_tokens = 4096
    )

    with open(f"output/JV_text_{i+1}.txt", "w") as file:
        file.write(response.choices[0].message.content)
