# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import mmcv

from mmpose.apis import init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False



from topdown_demo_with_mmdet import process_one_image

import gradio as gr
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



pose_config = None
det_config = None
pose_ckpt = None
det_ckpt = None
visualizer = None

pose_estimator = None
detector = None   

device = "cuda:0"
args = get_default_args()

def load_pose_by_ID(k):
    global pose_config, pose_ckpt, pose_estimator, visualizer
    
    pose_config = POSE_CONFIGS[k]
    pose_ckpt = CONFIG_TO_CKPT[k]
    pose_config_path = os.path.join(POSE_CONFIG_ROOT, pose_config)
    pose_ckpt_path = os.path.join(POSE_CKPT_ROOT, pose_ckpt)
    
    if not os.path.exists(pose_config_path):
        return f"Pose Config {pose_config} not found"
    if not os.path.exists(pose_ckpt_path):
        return f"Pose Checkpoint {pose_ckpt} not found"
    
    pose_estimator = init_pose_estimator(pose_config_path, pose_ckpt_path, device=device)
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
    

    return "Pose Model Loaded"
    
def load_detector_by_ID(k):
    global det_config, det_ckpt, detector
    det_config = DET_CONFIGS[k]
    det_ckpt = CONFIG_TO_CKPT[k]
    
    det_config_path = os.path.join(DET_CONFIG_ROOT, det_config)
    det_ckpt_path = os.path.join(DET_CKPT_ROOT, det_ckpt)
    
    if not os.path.exists(det_config_path):
        return f"Detector Config {det_config} not found"
    if not os.path.exists(det_ckpt_path):
        return f"Detector Checkpoint {det_ckpt} not found"

    detector = init_detector(det_config_path, det_ckpt_path, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    return "Detector Loaded"

def load_args(args, checkbox_group, bbox_thr, nms_thr, kpt_thr, skeleton_style, radius, thickness, alpha, output_fps, det_cat_id, visualize):
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
    args.det_cat_id = det_cat_id
    args.visualize = visualize
    
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    
    return str(args)
    


def process_one_video(
    args,
    video_path,
    detector, 
    pose_estimator, 
    visualizer=None
):
    video_name_without_ext = os.path.splitext( os.path.basename(video_path))[0]
    
    output_pred_file = tempfile.NamedTemporaryFile(suffix='.mp4')
    output_video_path = output_pred_file.name
    output_pred_file = tempfile.NamedTemporaryFile(suffix='.json')
    output_pred_path = output_pred_file.name
    # output_video_path = os.path.join(args.output_video_root, video_name_without_ext + '.mp4')
    # output_pred_path = os.path.join(args.output_pred_root, video_name_without_ext + '.json')
    
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
    
    # with open(output_pred_path, 'w') as f:
    #     json.dump(
    #         dict(
    #             meta_info=pose_estimator.dataset_meta,
    #             instance_info=pred_instances_list
    #         ), f, indent='\t'
    #     )
    
    
    return output_video_path, dict(meta_info=pose_estimator.dataset_meta,instance_info=pred_instances_list)
    
    
with gr.Blocks() as demo:
    args = gr.State(get_default_args())
    
    pose_config = gr.State()
    det_config = None
    pose_ckpt = None
    det_ckpt = None
    visualizer = None

    pose_estimator = None
    detector = None   
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
        pose_btn.click(fn=load_pose_by_ID, inputs=[pose_conf,], outputs=pose_output)
        det_btn.click(fn=load_detector_by_ID, inputs=[det_conf,], outputs=det_output)

        
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
                fn=partial(load_args, args), 
                inputs=[checkbox_group, bbox_thr, nms_thr, kpt_thr, skeleton_style, radius, thickness, alpha, output_fps, det_cat_id],
                outputs=more_output
            )
            
    with gr.Row():
        img1 = gr.Image()
        img2 = gr.Image()
    
    with gr.Row():
        process_img_btn = gr.Button("Process Image", scale=2)
        
    with gr.Accordion("Prediction Data", open=False):
        json1 = gr.JSON()
        
    def tmp_fn(img):
        pred_instances = process_one_image(args, img, detector, pose_estimator, visualizer, 0)
        frame_vis = visualizer.get_image()
        pred_instances_list = []
        pred_instances_list.append(
            dict(
                frame_id=-1,
                instances=split_instances(pred_instances)
            )
        )
        return pred_instances_list, mmcv.bgr2rgb(frame_vis)
    process_img_btn.click(fn=tmp_fn, inputs=[img1], outputs=[json1, img2])

    with gr.Row():
        video1 = gr.Video()
        video2 = gr.Video()
    
    with gr.Row():
        process_video_btn = gr.Button("Process Video", scale=2)
        
    with gr.Row():
        json2 = gr.JSON()
        # file1 = gr.File(label="Prediction Data")
        
    def tmp_process_video(video):
        output_video_path, output_pred_path = process_one_video(args, video, detector, pose_estimator, visualizer)
        return output_video_path, output_pred_path
        
    process_video_btn.click(fn=tmp_process_video, inputs=[video1], outputs=[video2, json2])

if __name__ == '__main__':
    demo.launch(show_error=True)
