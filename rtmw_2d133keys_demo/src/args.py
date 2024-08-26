import argparse

def get_known_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_config', help='Config file for detection')
    parser.add_argument('--det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('--pose_config', help='Config file for pose')
    parser.add_argument('--pose_checkpoint', help='Checkpoint file for pose')
    
    
    parser.add_argument('--show', action='store_true', default=False, help='whether to show img')
    parser.add_argument('--draw_heatmap', action='store_true', default=False, help='Draw heatmap predicted by the model')
    parser.add_argument('--show_kpt_idx', action='store_true', default=False, help='Whether to show the index of keypoints')
    parser.add_argument('--draw_bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualizer or not')

    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--det_cat_id', type=int, default=0, help='Category id for bounding box detection model')
    parser.add_argument('--bbox_thr', type=float, default=0.3, help='Bounding box score threshold')
    parser.add_argument('--nms_thr', type=float, default=0.3, help='IoU threshold for bounding box NMS')
    parser.add_argument('--kpt_thr', type=float, default=0.3, help='Visualizing keypoint thresholds')
    parser.add_argument('--skeleton_style', default='mmpose', type=str, choices=['mmpose', 'openpose'], help='Skeleton style selection')
    parser.add_argument('--radius', type=int, default=3, help='Keypoint radius for visualization')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness for visualization')
    parser.add_argument('--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument('--output_fps', type=int, default=25, help='FPS of output visualizer')
    return parser.parse_known_args()


from types import SimpleNamespace


def get_default_args():
    DEFAULT_ARGS = {
        'show': False,
        'draw_heatmap': False,
        'show_kpt_idx': False,
        'draw_bbox': False,
        'visualize': True,
        'device': 'cuda:0',
        'bbox_thr': 0.3,
        'nms_thr': 0.3,
        'kpt_thr': 0.3,
        'radius': 3,
        'thickness': 1,
        'alpha': 0.8,
        'output_fps': 25,
        'det_cat_id': 0,
        'skeleton_style': 'mmpose',
        'output_video_root': '../output/video/',
        'output_pred_root': '../output/pred/',
    }
    return SimpleNamespace(**DEFAULT_ARGS)
