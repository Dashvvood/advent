from fastapi import FastAPI
import gradio as gr
from interface import demo
import os

app = FastAPI()
@app.get("/")
def read_main():
    return {"pid": f"{os.getpid()}"}

app = gr.mount_gradio_app(app, demo, path="/gradio")