import os
import sys
sys.path.append("..")
sys.path.append("../src")

from src.main import *

# flask
from flask import (
    Flask, 
    jsonify, 
    request, 
    send_file,
    make_response,
)

import logging
import torch
import argparse

import motti
from motti.gpu import set_process_gpu

logging.basicConfig(
    filename='./api.log',
    level=logging.INFO, 
    format=f'%(asctime)s %(levelname)s {os.getpid()} %(filename)s:%(lineno)d : %(message)s'
)


app = Flask(__name__)
config = motti.load_yaml("./config.yaml")
config['output_video_root'] = os.path.join(config['output_root'], "video")
config['output_pred_root'] = os.path.join(config['output_root'], "pred")

app.logger.info(f"Default config: {config}")

gpu_id = set_process_gpu()
device = torch.device(gpu_id)

# detector
detector = init_detector(
    config['det_config'], 
    config['det_checkpoint'], 
    device=device
)
detector.cfg = adapt_mmdet_pipeline(detector.cfg)

# build pose estimator
pose_estimator = init_pose_estimator(
    config['pose_config'],
    config['pose_checkpoint'],
    device=device,
    cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=config['draw_heatmap']))) 
)

# build visualizer
pose_estimator.cfg.visualizer.radius = config['radius']
pose_estimator.cfg.visualizer.alpha = config['alpha']
pose_estimator.cfg.visualizer.line_width = config['thickness']
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_pose_estimator
visualizer.set_dataset_meta(
    pose_estimator.dataset_meta, 
    skeleton_style=config['skeleton_style']
)


@app.route("/", methods=['GET'])
def index():
    logging.info(f"{request.remote_addr = }")
    worker_id = int(os.environ.get('APP_WORKER_ID', 1))
    logging.info(f"Worker ID: {worker_id}, pid: {os.getpid()}")

    response = make_response("", 200)
    return response


@app.route("/videopath/", methods=['GET', 'POST'])
def process_by_video_path():
    try:
        worker_id = int(os.environ.get('APP_WORKER_ID', 1))
        logging.info(f"{request.remote_addr = }; Worker ID: {worker_id}, pid: {os.getpid()}")

        if request.method == "GET":
            return make_response(f"Worker ID: {worker_id}", 200)

        elif request.method == 'POST':
            D = request.get_json()
            filelist = D["filelist"]
            args = D["args"]

            logging.info(f"{len(filelist) = }, {filelist[:2] = }, {filelist[-2:] = }")
            
            args = argparse.Namespace(**args)
            args.output_video_root = os.path.join(args.output_root, "video")
            args.output_pred_root = os.path.join(args.output_root, "pred")

            mmengine.mkdir_or_exist(args.output_root)
            mmengine.mkdir_or_exist(args.output_video_root)
            mmengine.mkdir_or_exist(args.output_pred_root)

            finished = []

            for filename in filelist:
                if args.visualize is True:
                    res = process_one_video(
                        args, filename=filename, detector=detector, 
                        pose_estimator=pose_estimator,
                        visualizer=visualizer
                    )

                else:
                    res = process_one_video_wo_visualizer(
                        args, filename=filename, detector=detector, 
                        pose_estimator=pose_estimator,
                        visualizer=visualizer
                    )
                logging.info(f"Finish {filename}")
                finished.append(filename)
            
            return jsonify({"res": finished})
        
        else:
            return make_response("not support", 404)

    except Exception as e:
        logging.error(e)
        return make_response(e, 500)
    
    return make_response("done", 200)



@app.route("/test/", methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        f = request.files['file']
        print(f"args: {request.args}")
        print(f"filename: {f.filename}")
        print(f"content_type: {request.headers.get('content_type')}")
        print(f"Headers: {request.headers}")
        f.save("./tmp.png")
        return "file uploaded"
    elif request.method == 'GET':
        pass
    else:
        raise NotImplementedError


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="17700")
    