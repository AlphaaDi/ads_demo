from pathlib import Path
import numpy as np
import cv2

import torch
import torch.nn.functional as F

from flowformer.configs.submission import get_cfg as get_submission_cfg
from flowformer.configs.things_eval import get_cfg as get_things_cfg
from flowformer.configs.small_things_eval import get_cfg as get_small_things_cfg
from flowformer.core.utils.misc import process_cfg
from flowformer.core.FlowFormer import build_flowformer
from flowformer.core.utils.utils import InputPadder, forward_interpolate
from flowformer.core.utils import flow_viz


TRAIN_SIZE = [432, 960]

def format_img(img):
    img = np.array(img).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]

def read_frames(folder_path, glob='*.jpg'):
    frames_pathes = list(Path(folder_path).glob(glob))
    frames_pathes.sort(key=lambda x: int(x.stem))
    frames = [cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB) for frame_path in frames_pathes]
    return frames

class OpticalFlowFormer(torch.nn.Module):
    def __init__(self, model_path='path', device='cuda'):
        super().__init__()
        self.device = device
        cfg = get_things_cfg()

        self.model = torch.nn.DataParallel(build_flowformer(cfg))
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.eval().to(self.device)

    def forward(self, frames):
        with torch.no_grad():
            flows = []
            for img1_path, img2_path in zip(frames[:-1], frames[1:]):
                image1 = format_img(img1_path).to(self.device)
                image2 = format_img(img2_path).to(self.device)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow = self.model(image1, image2)
                flow_unpad = padder.unpad(flow[0]).cpu()[0]
                flow_unpad = flow_unpad.numpy().transpose((1,2,0))
                flows.append(flow_unpad)
    
        return flows

if __name__ == "__main__":
    flowFormer = OpticalFlowFormer('/home/davinci/work/unipopcorn/unipop/flowf/weights/things.pth')
    vc = cv2.VideoCapture('/home/davinci/work/unipopcorn/unipop/flowf/flowformer/woman-63241.mp4')
    frames = []
    _, img = vc.read()
    while _:
        frames.append(img)
        cv2.imshow('img', img)
        cv2.waitKey(5)
        _, img = vc.read()

    flows = flowFormer.forward(frames[:5]) 
    for flow in flows:
        flow_img = flow_viz(flow)
        cv2.imshow('flow', flow_img[:,:,[2,1,0]])
        cv2.waitKey(5)