import requests
import os
import json
import pandas as pd
import joblib

def get_embedding(text_list):
    response = requests.post("http://localhost:11434/api/embed", json={"model": "bge-m3", "input": text_list})
    return response.json()["embeddings"]

jsons = os.listdir("new_jsons")
my_dicts = []
chunk_id = 0
for json_file in jsons:
    with open(f"new_jsons/{json_file}", "r", encoding="utf-8") as f:
        content = json.load(f)
    print(f"Creating embeddings for {json_file}")
    embedding = get_embedding([c['text'] for c in content['chunks']])

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk_id += 1
        chunk['embedding'] = embedding[i]
        my_dicts.append(chunk)
df = pd.DataFrame.from_records(my_dicts)

# Save this dataframe using joblib
joblib.dump(df, 'chunk_embeddings.joblib')
