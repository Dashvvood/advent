import requests
import time
import os

resp = requests.post(
    url="http://127.0.0.1:17700/path/",
    json={
        "args": {"language": "English"},
        "audio_path": os.path.abspath("../data/poi.mp3"),
        "tic": time.time(),
        "output_format": ["json", "srt"],
    }
)

print(resp.json())
