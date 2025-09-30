import os
import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from util import setup

config = setup("config/config.yaml")
textfiles = os.listdir(os.path.join(config.input.input_dir, "texts"))
print("# Bestanden:", len(textfiles))

embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
print("Embedder device:", embedder.device)

textfiles = textfiles[:2]

data = []
for i, textfile in enumerate(textfiles):
    print(f"{i+1}/{len(textfiles)}")
    if textfile.endswith(".txt"):
        label = 1
        if "non_JV" in textfile:
            label = 0

        with open(os.path.join(config.input.input_dir, "texts", textfile), "r") as textfile_data:
            output = embedder.encode(textfile_data.read()) 
            print("Output shape:", output.shape)
            data_entry = (output, label)
            data.append(data_entry)

with open("output/data.pickle", "wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('output/data.pickle', 'rb') as handle:
    data2 = pickle.load(handle)

print("Eerste data 2 entry:", data2[0])

model = DecisionTreeClassifier()

train_x = [x[0] for x in data2]
train_y = [x[1] for x in data2]
model_trained = model.fit(train_x, train_y)

print("Model fitted on data")