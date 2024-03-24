python ../src/topdown_demo_with_mmdet.py \
    ../config/rtmdet_m_640-8xb32_coco-person.py \
    ../ckpt/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    ../config/rtmw-x_8xb320-270e_cocktail14-384x288.py \
    ../ckpt/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth \
    --input ../data/000000000785.jpg \
    --output-root . \
    --draw-heatmap