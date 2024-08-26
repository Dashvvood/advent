# 配置文件列表
POSE_CONFIGS = {
    "rtmpose-l_384x288": "rtmpose-l_8xb32-270e_coco-wholebody-384x288.py",
    "rtmpose-l_256x192": "rtmpose-l_8xb64-270e_coco-wholebody-256x192.py",
    "rtmpose-m_256x192": "rtmpose-m_8xb64-270e_coco-wholebody-256x192.py",
    "rtmpose-s_256x192": "rtmpose-s_8xb64-270e_coco-wholebody-256x192.py",
    "rtmpose-t_256x192": "rtmpose-t_8xb64-270e_coco-wholebody-256x192.py",
    "rtmpose-x_384x288": "rtmpose-x_8xb32-270e_coco-wholebody-384x288.py",
    "rtmw-l_256x192": "rtmw-l_8xb1024-270e_cocktail14-256x192.py",
    "rtmw-l_384x288": "rtmw-l_8xb320-270e_cocktail14-384x288.py",
    "rtmw-m_256x192": "rtmw-m_8xb1024-270e_cocktail14-256x192.py",
    "rtmw-x_384x288": "rtmw-x_8xb320-270e_cocktail14-384x288.py",
    "rtmw-x_256x192": "rtmw-x_8xb704-270e_cocktail14-256x192.py"
}
POSE_CONFIG_DEFAULT = "rtmw-x_384x288"


DET_CONFIGS = {
    "humanart_detection": "humanart_detection.py",
    "rtmdet_l 8xb32-300e humanart": "rtmdet_l_8xb32-300e_humanart.py",
    "rtmdet_m_640-8xb32_coco-person": "rtmdet_m_640-8xb32_coco-person.py",
    "rtmdet_m 8xb32-300e humanart": "rtmdet_m_8xb32-300e_humanart.py",
    "rtmdet_nano 320-8xb32 coco-person": "rtmdet_nano_320-8xb32_coco-person.py",
    "rtmdet_s 8xb32-300e humanart": "rtmdet_s_8xb32-300e_humanart.py",
    "rtmdet_tiny 8xb32-300e humanart": "rtmdet_tiny_8xb32-300e_humanart.py",
    "rtmdet_x 8xb32-300e humanart": "rtmdet_x_8xb32-300e_humanart.py"
}
DET_CONFIG_DEFAULT = "rtmdet_m_640-8xb32_coco-person"


CONFIG_TO_CKPT = {
    "rtmw-x_384x288": "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth",
    "rtmdet_m_640-8xb32_coco-person": "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",   
}

POSE_CONFIG_ROOT = "../config/"
DET_CONFIG_ROOT = "../config/"
POSE_CKPT_ROOT = "../ckpt/"
DET_CKPT_ROOT = "../ckpt/"
