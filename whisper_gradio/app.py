import gradio as gr
import whisper
import yaml

config = yaml.safe_load(open("config.yaml", 'r'))

model = whisper.load_model(
    name=config["model"]["model_name"], 
    device=config["model"]["device"], 
    download_root=config["model"]["model_dir"]
)


def transcribe(audio_file, language=None):
    if language is None:
        result = model.transcribe(audio_file)
    else:
        result = model.transcribe(audio_file, language=language)
    return result["text"]


def main():
    audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath")
    lan_input = gr.Dropdown(
        choices=["English", "French", "Chinese", None],
        value=[None]
    )
    text_output = gr.Textbox()
    
    interface = gr.Interface(fn=transcribe, inputs=[audio_input, lan_input], 
                         outputs=text_output, title="Whisper Transcription",
                         description="Upload audio")
    interface.launch(
        server_name=config["app"]["server_name"], 
        server_port=config["app"]["server_port"]
    )


if __name__ == '__main__':
    main()
    