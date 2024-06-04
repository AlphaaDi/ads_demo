import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


config = '/home/davinci/work/unipopcorn/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
checkpoint = '/home/davinci/work/unipopcorn/unipop/pipe/weights/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth'


def get_colors():
    np.random.seed(997)
    colors = np.stack(np.meshgrid(np.arange(0,255),np.arange(0,255),np.arange(0,255)))
    colors = colors.reshape(3,-1).T
    permute_idxs = np.random.permutation(np.arange(colors.shape[0]))
    colors = colors[permute_idxs]
    return colors.astype(np.uint8)

class Mask2Former(torch.nn.Module):
    def __init__(self, weights, config_path, device='cuda:1', threshold = 0.8):
        super().__init__()
        config = mmcv.Config.fromfile(config_path)
        model = build_detector(config.model)
        checkpoint = load_checkpoint(model, weights, map_location=device)
        self.classes = checkpoint['meta']['CLASSES']
        self.threshold = threshold
        model.to(device)
        model.cfg = config
        self.model = model.eval()
        self.colors = get_colors()

    @torch.no_grad()
    def forward(self, image):
        H,W,_ = image.shape
        image_res = cv2.resize(image, (512,512))
        print(image_res.shape)
        class_bboxs, masks = inference_detector(self.model, image_res)
        objects_mask = np.zeros_like(image_res).astype(np.uint8)
        obj_couint = 0
        for idx, class_name in enumerate(self.classes):
            for bbox, mask in zip(class_bboxs[idx], masks[idx]):
                if bbox[-1] < self.threshold:
                    continue
                objects_mask[mask] = [1,1,1]
                obj_couint += 1
                
        del class_bboxs, masks
        torch.cuda.empty_cache()

        objects_mask = cv2.resize(objects_mask, (W,H))
        return objects_mask