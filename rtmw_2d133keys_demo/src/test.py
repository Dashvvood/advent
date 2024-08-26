import gradio as gr
from args import get_default_args
import random
import string


def generate_random_number():
    return random.randint(1, 100)


def n2s(n):
    return n
    # return str(n)

with gr.Blocks() as demo:
    x = gr.State(0)
    text = gr.Textbox()
    btn = gr.Button("Print Args")
    
    btn.click(fn=generate_random_number, inputs=None, outputs=x)
    x.change(fn=n2s, inputs=x, outputs=text)
    
demo.launch()
