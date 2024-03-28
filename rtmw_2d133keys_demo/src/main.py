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

from utils import get_known_args
from topdown_demo_with_mmdet import process_one_image
from tqdm import tqdm

def process_one_video(
    args,
    filename,
    detector, 
    pose_estimator, 
    visualizer=None
):
    input_file = os.path.join(args.input_root, filename)
    output_file = os.path.join(args.output_video_root, filename)
    pred_save_path = os.path.join(args.output_pred_root, os.path.splitext(filename)[0]+".json")

    cap = cv2.VideoCapture(input_file)
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
                output_file,
                fourcc,
                args.output_fps,  # saved fps
                (frame_vis.shape[1], frame_vis.shape[0]))

        video_writer.write(mmcv.rgb2bgr(frame_vis))

    video_writer.release()
    cap.release()
    
    with open(pred_save_path, 'w') as f:
        json.dump(
            dict(
                meta_info=pose_estimator.dataset_meta,
                instance_info=pred_instances_list
            ), f, indent='\t'
        )
    print(f"predictions have been saved at {pred_save_path}")
    print(f"the output has been saved at {output_file}")

    return True


def process_one_video_wo_visualizer(
    args,
    filename,
    detector, 
    pose_estimator, 
    visualizer=None
):
    input_file = os.path.join(args.input_root, filename)
    # output_file = os.path.join(args.output_video_root, filename)
    pred_save_path = os.path.join(args.output_pred_root, os.path.splitext(filename)[0]+".json")

    cap = cv2.VideoCapture(input_file)
    # video_writer = None
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

        # frame_vis = visualizer.get_image()

    #     if video_writer is None:
    #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #         # the size of the image with visualization may vary
    #         # depending on the presence of heatmaps
    #         video_writer = cv2.VideoWriter(
    #             output_file,
    #             fourcc,
    #             24,  # saved fps
    #             (frame_vis.shape[1], frame_vis.shape[0]))

    #     video_writer.write(mmcv.rgb2bgr(frame_vis))

    # video_writer.release()
    cap.release()
    
    with open(pred_save_path, 'w') as f:
        json.dump(
            dict(
                meta_info=pose_estimator.dataset_meta,
                instance_info=pred_instances_list
            ), f, indent='\t'
        )
    print(f"predictions have been saved at {pred_save_path}")
    # print(f"the output has been saved at {output_file}")

    return True


def main():
    args, dismatch_args = get_known_args()
    print(f"{dismatch_args = }")
    args.output_video_root = os.path.join(args.output_root, "video")
    args.output_pred_root = os.path.join(args.output_root, "pred")

    mmengine.mkdir_or_exist(args.output_root)
    mmengine.mkdir_or_exist(args.output_video_root)
    mmengine.mkdir_or_exist(args.output_pred_root)

    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap)))
    )

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    for filename in tqdm(os.listdir(args.input_root)):
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
        print(f"Finish {filename}")

if __name__ == '__main__':
    main()
    