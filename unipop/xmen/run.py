
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.set_grad_enabled(False)

from unipop.xmen.inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from unipop.xmen.inference.data.mask_mapper import MaskMapper
from unipop.xmen.model.network import XMem
from unipop.xmen.inference.inference_core import InferenceCore
from unipop.xmen.inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis


XMEN_WEIGHTS = '/home/davinci/work/unipopcorn/unipop/pipe/weights/XMem-s012.pth'

def read_frames(folder_path, glob='*.jpg'):
    frames_pathes = list(Path(folder_path).glob(glob))
    frames_pathes.sort(key=lambda x: int(x.stem))
    frames = [
        cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB) 
        for frame_path in frames_pathes
    ]
    return frames

def frame_extraction(video_path):
    frames = []
    vid = cv2.VideoCapture(video_path)
    flag,frame = vid.read()
    cnt = 0
    new_h, new_w= None, None
    while flag:
        frames.append(frame)
        flag, frame = vid.read()
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    return frames

def get_iou(mask1, mask2):
    classes = (set(np.unique(mask1)) | set(np.unique(mask2))) - {0}
    areas = 0
    intersection = 0
    for class_ in classes:
        mask1_area = np.count_nonzero (mask1 == class_)
        mask2_area = np.count_nonzero (mask2 == class_)
        intersection += np.count_nonzero(np.logical_and(mask1 == class_, mask2 == class_))
        areas += mask1_area + mask2_area
    return intersection / (areas - intersection + 1)


def get_mean_ious(masks1, masks2):
    iou_acc = 0
    count = 0
    for mask1, mask2 in zip(masks1, masks2):
        iou_acc += get_iou(mask1, mask2)
        count += 1
    return iou_acc / count


def prepare_mask(mask, colors= None) :
    msk= mask.astype(np.int64)
    mskk = msk[:,:,0] * 255 * 255 + msk[:,:,1] * 255 + msk[:,:,2]
    if colors is None:
        colors = np.unique(mskk)
    for idx, val in enumerate(colors) :
        mskk[mskk==val] = idx
    return mskk.astype(np.uint8), colors


class PropagationxMem(torch.nn.Module):
    def __init__(self,model_path, device='cuda:0'):
        super().__init__()
        self.device = device
        self.config = {
            'top_k': 30,
            'mem_every': 5,
            'deep_update_every': -1,
            'enable_long_term': True,
            'enable_long_term_count_usage': True,
            'num_prototypes': 128,
            'min_mid_term_frames': 5,
            'max_mid_term_frames': 10,
            'max_long_term_elements': 10_000,
        }
        self.network = XMem(self.config, model_path, map_location=self.device).eval().to(self.device)

    def forward(self, frames, first_mask):
        mask, colors = prepare_mask(first_mask)
        num_objects = len(np.unique(mask)) - 1
        processor = InferenceCore(self.network, config=self.config)
        processor.set_all_labels(range(1, num_objects+1))
        torch.cuda.empty_cache()
        current_frame_index = 0
        predictions = []
        with torch.cuda.amp.autocast(enabled=True):
            for frame in frames:
                frame_torch, _ = image_to_torch(frame, device=self.device)
                if current_frame_index == 0:
                    mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(self.device)
                    prediction = processor.step(frame_torch, mask_torch[1:])
                else:
                    prediction = processor.step(frame_torch)
                current_frame_index += 1
                prediction = torch_prob_to_numpy_mask(prediction)
                predictions.append(prediction)
        return predictions

if __name__ == "__main__":
    import cv2


    propogator = PropagationxMem('/home/davinci/work/unipopcorn/unipop/xmen/scripts/saves/XMem-s012.pth',device='cuda:1')

    frames = read_frames('/home/davinci/work/unipopcorn/data/imgs/kite-walk')
    masks = read_frames('/home/davinci/work/unipopcorn/data/masks/1080p/kite-walk',glob='*.png')

    prop_masks = propogator.forward(frames, masks[0])
    prop_masks = np.array(prop_masks)
    print(len(frames), len(prop_masks))
    print(prop_masks.dtype, np.min(prop_masks), np.max(prop_masks), prop_masks.shape)

    for i, msk in enumerate(prop_masks):
#       cv2.imwrite(f'{i}.png',msk * 255)
        cv2.imshow('mask', msk*255)
        cv2.waitKey(5)
#    img0 = cv2.imread()
#    img1 = cv2.imread()
#    mask0 = cv2.imread()
#    prop = PropagationXmen()
#    prop.init(img0, mask0)
#    mask1 = prop.forward(img1)
#    res = cv2.hconcat([img0, mask0, img1, mask1])
#    cv2.imshow('res', res)
    