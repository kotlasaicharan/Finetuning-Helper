from sklearn.model_selection import train_test_split
train,val = train_test_split(df, test_size=0.1, shuffle=False, stratify = None )

train.to_json("train.json", orient="records", lines=True)
val.to_json("val.json", orient="records", lines=True)

import json
from datasets import load_dataset
dataset = load_dataset("json", data_files={"train": "train.json", "validation": "val.json"})

dataset
