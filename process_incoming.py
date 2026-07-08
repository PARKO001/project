import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def get_embedding(text_list):
    response = requests.post("http://localhost:11434/api/embed", json={"model": "bge-m3", "input": text_list})
    return response.json()["embeddings"]

# Load the saved dataframe
df = joblib.load('chunk_embeddings.joblib')


#Taking user question input and generating embedding for it
question = input("Enter your question: ")
question_embedding = get_embedding([question])[0] 

#Find similarities of the question embedding with the chunk embeddings
similarities = cosine_similarity([question_embedding], df['embedding'].tolist())[0]
# print(similarities)
top_results = 30
top_n_indices = similarities.argsort()[::-1][:top_results]
top_n_results = df.iloc[top_n_indices]
# print(top_n_results[["audio", "chunk_id", "text"]])

for index, item in top_n_results.iterrows():
    print(index, item["audio"], item["chunk_id"], item["text"], item["start"], item["end"])