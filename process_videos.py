# convert videos to mp3
from pathlib import Path
import subprocess

videos_dir = Path('videos')
audios_dir = Path('audios')
audios_dir.mkdir(exist_ok=True)

for video_path in videos_dir.glob('*.mp4'):
    output_path = audios_dir / f'{video_path.stem}.mp3'
    subprocess.run([
        'ffmpeg',
        '-i', str(video_path),
        '-q:a', '0',
        '-map', 'a',
        str(output_path),
    ], check=True)