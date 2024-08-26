# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from topdown_demo_with_mmdet import process_one_image

import gradio as gr
import os
from constant import (
    POSE_CONFIGS,
    POSE_CONFIG_DEFAULT,
    DET_CONFIGS,
    DET_CONFIG_DEFAULT,
    CONFIG_TO_CKPT,
    POSE_CONFIG_ROOT,
    DET_CONFIG_ROOT,
    POSE_CKPT_ROOT,
    DET_CKPT_ROOT,
)

from args import get_default_args
from functools import partial
import motti
import tempfile
import uuid


def init_pose_estimator_by_name(args, name):
    args.pose_config = POSE_CONFIGS[name]
    args.pose_ckpt = CONFIG_TO_CKPT[name]
    args.pose_config_path = os.path.join(POSE_CONFIG_ROOT, args.pose_config)
    args.pose_ckpt_path = os.path.join(POSE_CKPT_ROOT, args.pose_ckpt)
    
    try:
        pose_estimator = init_pose_estimator(args.pose_config_path, args.pose_ckpt_path, device=args.device)
        pose_estimator.cfg.visualizer.radius = args.radius
        pose_estimator.cfg.visualizer.alpha = args.alpha
        pose_estimator.cfg.visualizer.line_width = args.thickness
        
        args.visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        args.visualizer.set_dataset_meta(
            pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
    except Exception as e:
        print(e)
        return None
    
    return pose_estimator

def init_detector_by_name(args, name):
    args.det_config = DET_CONFIGS[name]
    args.det_ckpt = CONFIG_TO_CKPT[name]
    args.det_config_path = os.path.join(DET_CONFIG_ROOT, args.det_config)
    args.det_ckpt_path = os.path.join(DET_CKPT_ROOT, args.det_ckpt)

    try:
        detector = init_detector(args.det_config_path, args.det_ckpt_path, device=args.device)
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    except Exception as e:
        print(e)
        return None
    
    return detector

def reload_args(args, pose_estimator, checkbox_group, bbox_thr, nms_thr, kpt_thr, skeleton_style, radius, thickness, alpha, output_fps, det_cat_id):
    for x in checkbox_group:
        setattr(args, x, True)
    args.bbox_thr = bbox_thr
    args.nms_thr = nms_thr
    args.kpt_thr = kpt_thr
    args.skeleton_style = skeleton_style
    args.radius = radius
    args.thickness = thickness
    args.alpha = alpha
    args.output_fps = output_fps
    
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    
    args.visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    args.visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
    
    return args

def process_one_video(
    args,
    video_path,
    detector, 
    pose_estimator, 
    visualizer=None
):
    video_name_without_ext = os.path.splitext(os.path.basename(video_path))[0]
    
    # output_pred_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    # output_video_path = output_pred_file.name
    # output_pred_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
    # output_pred_path = output_pred_file.name
    
    output_name = uuid.uuid4().hex
    output_video_path = os.path.join(args.output_video_root, output_name + '.mp4')
    output_pred_path = os.path.join(args.output_pred_root, output_name + '.json')
    
    cap = cv2.VideoCapture(video_path)
    video_writer = None
    pred_instances_list = []
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1
        if not success:
            break
        pred_instances = process_one_image(
            args, frame, detector,
            pose_estimator, visualizer,0.001
        )

        pred_instances_list.append(
            dict(
                frame_id=frame_idx,
                instances=split_instances(pred_instances)
            )
        )

        frame_vis = visualizer.get_image()

        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # the size of the image with visualization may vary
            # depending on the presence of heatmaps
            video_writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                args.output_fps,  # saved fps
                (frame_vis.shape[1], frame_vis.shape[0]))

        video_writer.write(mmcv.rgb2bgr(frame_vis))

    video_writer.release()
    cap.release()
    
    with open(output_pred_path, 'w') as f:
        json.dump(
            dict(
                meta_info=pose_estimator.dataset_meta,
                instance_info=pred_instances_list
            ), f, indent='\t'
        )
    
    return output_pred_path, output_video_path

def _object_to_textbox(model):
    return str(model)

def api_image(args, img, detector, pose_estimator):
    pred_instances = process_one_image(args, img, detector, pose_estimator, args.visualizer, 0)
    frame_vis = args.visualizer.get_image()
    pred_instances_list = []
    pred_instances_list.append(
        dict(
            frame_id=-1,
            instances=split_instances(pred_instances)
        )
    )
    return pred_instances_list, mmcv.bgr2rgb(frame_vis)

def api_video(args, video, detector, pose_estimator):
    return process_one_video(args, video, detector, pose_estimator, args.visualizer)


with gr.Blocks() as demo:
    
    # states input
    args = gr.State(get_default_args())
    pose_estimator = gr.State()
    detector = gr.State()
    
    with gr.Row():
        with gr.Column():
            # dropdown
            pose_conf = gr.Dropdown(
                choices=list(POSE_CONFIGS.keys()), 
                label="Select Pose Configuration", 
                value=POSE_CONFIG_DEFAULT,
            )
            
            pose_output = gr.Textbox()
            pose_btn = gr.Button("Load Pose Model")
            
        with gr.Column():
            det_conf = gr.Dropdown(
                choices=list(DET_CONFIGS.keys()), 
                label="Select Detector Configuration", 
                value=DET_CONFIG_DEFAULT
            )
            det_output = gr.Textbox()
            det_btn = gr.Button("Load Detector Model")

        # action
        pose_btn.click(fn=init_pose_estimator_by_name, inputs=[args, pose_conf,], outputs=pose_estimator)
        det_btn.click(fn=init_detector_by_name, inputs=[args, det_conf,], outputs=detector)

        def _model_to_textbox(model):
            return type(model)
        pose_estimator.change(fn=_model_to_textbox, inputs=pose_estimator, outputs=pose_output)
        detector.change(fn=_model_to_textbox, inputs=detector, outputs=det_output)
        
    with gr.Accordion("More Settings", open=False):
        with gr.Row():
            checkbox_group = gr.CheckboxGroup(
                choices=["show", "draw_heatmap", "show_kpt_idx", "draw_bbox", "visualize"],
                value=[],
                label="Visualization Options"
            )
        with gr.Row():
            bbox_thr = gr.Number(label="Bounding Box Threshold", value=0.3)
            nms_thr = gr.Number(label="NMS Threshold", value=0.3)
            kpt_thr = gr.Number(label="Keypoint Threshold", value=0.5)
            radius = gr.Number(label="Radius", value=3)
            
        with gr.Row():
            thickness = gr.Number(label="Thickness", value=1)
            alpha = gr.Number(label="Alpha", value=0.8)
            output_fps = gr.Number(label="Output FPS", value=25)
            det_cat_id = gr.Number(label="Category ID for Bounding Box Detection Model", value=0)
            skeleton_style = gr.Dropdown(choices=['mmpose', 'openpose'], label="Select Skeleton Style", value='mmpose')
            more_btn = gr.Button("Update Options")
            
        with gr.Row():
            more_output = gr.Textbox(value=str(args))
            more_btn.click(
                fn=reload_args,
                inputs=[args, pose_estimator, checkbox_group, bbox_thr, nms_thr, kpt_thr, skeleton_style, radius, thickness, alpha, output_fps, det_cat_id],
                outputs=args
            )
            args.change(fn=_object_to_textbox, inputs=args, outputs=more_output)
            
    with gr.Row():
        img1 = gr.Image()
        img2 = gr.Image()
    
    with gr.Row():
        process_img_btn = gr.Button("Process Image", scale=2)
        
    with gr.Accordion("Prediction Data", open=False):
        json1 = gr.JSON()
        
    process_img_btn.click(fn=api_image, inputs=[img1], outputs=[json1, img2])

    with gr.Row():
        video1 = gr.Video()
        video2 = gr.Video()
    
    with gr.Row():
        # json2 = gr.JSON()
        file1 = gr.File(label="Prediction Data")
        
    with gr.Row():
        process_video_btn = gr.Button("Process Video", scale=2)
        
    process_video_btn.click(fn=api_video, inputs=[video1], outputs=[file1, video2])
    

if __name__ == '__main__':
    demo.launch(show_error=True)
