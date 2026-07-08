import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import json

def get_embedding(text_list):
    response = requests.post("http://localhost:11434/api/embed", json={"model": "bge-m3", "input": text_list})
    return response.json()["embeddings"]

def inference(prompt):
    response = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "prompt": prompt, "stream": False})
    return response.json()

# Load the saved dataframe
df = joblib.load('chunk_embeddings.joblib')


#Taking user question input and generating embedding for it
question = input("Enter your question: ")
question_embedding = get_embedding([question])[0] 

#Find similarities of the question embedding with the chunk embeddings
similarities = cosine_similarity([question_embedding], df['embedding'].tolist())[0]
# print(similarities)
top_results = 100
top_n_indices = similarities.argsort()[::-1][:top_results]
top_n_results = df.iloc[top_n_indices]
# print(top_n_results[["audio", "chunk_id", "text"]])

prompt = f''' I've done some lab tests for on very famous items in indian food industry. Here are video subtitle chunks containing video title(audio), chunk_id, text, start and end time of the chunk. Here are the chunks:

{top_n_results[["audio", "chunk_id", "start", "end", "text"]].to_json(orient="records")}
--------------------
"{question}"
User asked this question related to video chunks, I want you to answer user's questions based on these chunks. If you don't find any relevant information in the chunks, please say "Please ask question from the provided context".
'''

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)
# for index, item in top_n_results.iterrows():
#     print(index, item["audio"], item["chunk_id"], item["text"], item["start"], item["end"])