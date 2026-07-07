import whisper
import json

model = whisper.load_model("large-v3")

result = model.transcribe(audio = "audios/MN creatine.mp3", language = "hi", task = "translate", word_timestamps = False)
chunks = []
for segment in result["segments"]:
    chunks.append({
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"]
    })
chunk_with_metadata = {
    "chunks": chunks,
    "text": result["text"]
}
print(chunk_with_metadata)
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(chunk_with_metadata, f, ensure_ascii=False, indent=4)