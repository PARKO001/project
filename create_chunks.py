import whisper
import json
import os

model = whisper.load_model("large-v3") 

audios = os.listdir("audios")

for audio in audios:
    result = model.transcribe(audio = f"audios/{audio}", language = "hi", task = "translate", word_timestamps = False, fp16=False)

    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "audio": audio[:-4],
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })
    
    chunk_with_metadata = {
        "chunks": chunks,
        "text": result["text"]
    }

    with open(f"jsons/{audio[:-4]}.json", "w", encoding="utf-8") as f:
        json.dump(chunk_with_metadata, f, ensure_ascii=False, indent=4)