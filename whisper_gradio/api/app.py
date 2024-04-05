import gradio as gr
import whisper


model = whisper.load_model(name="small", device="cuda:0", download_root="../ckpt/")


def transcribe(audio_file, language):
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
                         outputs=[text_output], title="Whisper Transcription",
                         description="Upload audio")
    interface.launch(server_name="127.0.0.1", server_port=7861)


if __name__ == '__main__':
    main()
    