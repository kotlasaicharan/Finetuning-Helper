from sklearn.model_selection import train_test_split
train,val = train_test_split(df, test_size=0.1, shuffle=False, stratify = None )

train.to_json("train.json", orient="records", lines=True)
val.to_json("val.json", orient="records", lines=True)

import json
from datasets import load_dataset
dataset = load_dataset("json", data_files={"train": "train.json", "validation": "val.json"})
dataset

------
def format_example(row : dict):
    prompt = dedent(
        f"""
        "Translate the given urdu poem to english, Provide only the translation, without any additional comments or explanations :"
        {row["Urdu"]}
        """
    )
    messages = [
        {
            "role": "system",
            "content": "You are a great translator",
            },
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": row["orginal_eng"],
        }
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False)

df["text"] = df.apply(format_example, axis=1)
df.head()
