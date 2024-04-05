# flask
from flask import (
    Flask, 
    request, 
    make_response,
    jsonify,
)

import whisper
import whisper.utils

import os
import logging
import time

import motti
from motti.gpu import set_process_gpu
import torch
from settings import *

logging.basicConfig(filename=LOG_PATH, encoding='utf-8', level=logging.DEBUG)

app = Flask(__name__)
gpu_id = set_process_gpu()

device = torch.device(gpu_id)
model = whisper.load_model(name=MODEL_NAME, device=device, download_root=MODEL_ROOT, in_memory=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
worker_id = int(os.environ.get('APP_WORKER_ID', 1))

logging.info(f"{worker_id = }, {MODEL_NAME = }, {device = };")

@app.route("/", methods=['GET'])
def index():
    logging.info(f"{request.remote_addr} = ")
    return make_response(f"{worker_id = }", 200)

@app.route("/path/", methods=['GET', 'POST'])
def task():
    try:
        logging.info(f"{request.remote_addr = }, {worker_id = }, pid = {os.getpid()}")
        if request.method == "GET":
            client = {
                "args": {"language": "English"},
                "audio_path": "str",
                "tic": time.time(),
                "output_format": ["json", "srt"],
            }
            return jsonify(client)
        
        elif request.method =="POST":
            D = request.get_json()
            args = D["args"]
            # 提取文件名
            audio_path = D["audio_path"]
            output_format = D["output_format"]
            tic = float(D["tic"])

            if not os.path.exists(audio_path):
                raise FileNotFoundError

            logging.info(f"{audio_path = }")

            res = model.transcribe(audio_path, **args)

            for format in output_format:
                writer = whisper.utils.get_writer(output_format=format, output_dir=OUTPUT_DIR)
                writer(result=res, audio_path=audio_path)

            toc = time.time()
            time_used = toc - tic
            logging.info(f"{time_used = }")
            return jsonify(time_used)

        else:
            pass
    except Exception as e:
        logging.error(e)
        return make_response(f"{e}")
    
if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port="17700")
    