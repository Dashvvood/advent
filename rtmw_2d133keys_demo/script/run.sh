#!/bin/bash
MAIN="../src/main.py"
INPUT="../data/focus/"
OUTPUT="../output/"

DET_CONFIG="../config/rtmdet_m_640-8xb32_coco-person.py"
DET_CKPT="../ckpt/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
POSE_CONFIG="../config/rtmw-x_8xb320-270e_cocktail14-384x288.py"
POSE_CKPT="../ckpt/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

python ${MAIN} \
    --det_config ${DET_CONFIG} \
    --det_checkpoint ${DET_CKPT} \
    --pose_config ${POSE_CONFIG} \
    --pose_checkpoint ${POSE_CKPT} \
    --input_root ${INPUT} \
    --output_root ${OUTPUT} \
    --device cuda:0 \
    --output_fps 25 \
    $@