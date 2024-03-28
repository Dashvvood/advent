import argparse

def get_known_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_config', help='Config file for detection')
    parser.add_argument('--det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('--pose_config', help='Config file for pose')
    parser.add_argument('--pose_checkpoint', help='Checkpoint file for pose')

    # File or Folder ?
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input_file", type=str, help='Image/Video file')
    group.add_argument("--input_root", type=str)

    
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output_root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    # parser.add_argument(
    #     '--save-predictions',
    #     action='store_true',
    #     default=False,
    #     help='whether to save predicted results')[]
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det_cat_id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox_thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms_thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt_thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw_heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show_kpt_idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton_style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    # parser.add_argument(
    #     '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw_bbox', action='store_true', help='Draw bboxes of instances')

    parser.add_argument(
        '--output_fps', type=int, default=25, help='FPS of output visualizer')
    
    parser.add_argument(
        '--visualize', action="store_true", default=False, help='visualizer or not')
    
    # assert has_mmdet, 'Please install mmdet to run the demo.'

    return parser.parse_known_args()

if __name__ == '__main__':
    args = get_known_args()
