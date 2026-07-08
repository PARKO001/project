import os
import math
import json

n = 5

for filename in os.listdir("jsons"):
    if filename.endswith(".json"):
        filepath = os.path.join("jsons", filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            new_chunks = []
            num_chunks = len(data["chunks"])
            num_groups = math.ceil(num_chunks / n)

            for i in range(num_groups):
                start_index = i * n
                end_index = min(start_index + n, num_chunks)
                chunk_group = data["chunks"][start_index:end_index]

                new_chunks.append({
                    "title": chunk_group[0]["audio"],
                    "start": chunk_group[0]["start"],
                    "end": chunk_group[-1]["end"],
                    "text": " ".join([chunk["text"] for chunk in chunk_group]),
                })

            # Save file withoiut double json
            os.makedirs("new_jsons", exist_ok=True)
            with open(os.path.join("new_jsons", filename), "w", encoding="utf-8") as json_file:
                json.dump({"chunks": new_chunks, "text": data["text"]}, json_file, ensure_ascii=False, indent=4)