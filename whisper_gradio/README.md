### Whisper

Official Repository: https://github.com/openai/whisper/tree/main

**Requirement :**

```shell
# ffmpeg for audio processing
sudo apt install ffmpeg

# Build Conda Env using python3.11
conda create -n whisper python=3.11 -y
conda activate whisper

# pip install -r requirements.txt
pip3 install openai-whisper gradio
```

**Example cli**

```shell
whisper example.wav \
	--model small \ # small, medium, large, largev2 ...
	--model_dir . \ # default: ~/.cache/whisper
	--task transcribe \
	--language French
```