import torch
import numpy as np
import argparse
import cv2
import sys
import os
from PIL import Image


#sys.path.insert(0, './thirdparty/RAFT/core')
from unipop.raft.core.utils.utils import InputPadder
from unipop.raft.core.raft import RAFT
from unipop.raft.core.utils import flow_viz
#sys.path.remove('./thirdparty/RAFT/core')


class RAFTWrapper():
    def __init__(self, model_path, device, max_long_edge=900):
        args = argparse.Namespace()
        args.small = False
        args.mixed_precision = True
        args.model = model_path
        args.max_long_edge = max_long_edge
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model))

        self.model = self.model.module
        self.model.to(device)
        self.device = device
        self.model.eval()

        self.args = args

    def load_image(self, fn):
        img = np.array(Image.open(fn)).astype(np.uint8)
        if img is None:
            print(f'Error reading file: {fn}')
            sys.exit(1)

        im_h = img.shape[0]
        im_w = img.shape[1]
        long_edge = max(im_w, im_h)
        factor = long_edge / self.args.max_long_edge
        if factor > 1:
            new_w = int(im_w // factor)
            new_h = int(im_h // factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img
    
    def load_image_cv(self, mat):
        im_h = mat.shape[0]
        im_w = mat.shape[1]
        long_edge = max(im_w, im_h)
        factor = long_edge / self.args.max_long_edge
        if factor > 1:
            new_w = int(im_w // factor)
            new_h = int(im_h // factor)
            mat = cv2.resize(mat, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(mat).permute(2,0,1).float()
        img = img.to(self.device)
        img_not_ask = torch.stack([img, img], dim=0)
        padder = InputPadder(img_not_ask.shape)
        img = padder.pad(img_not_ask)[0]
        img = img[0, None]
        return img

    def load_image_list(self, image_files):
        images = []
        for imfile in sorted(image_files):
            images.append(self.load_image(imfile))

        images = torch.stack(images, dim=0)
        images = images.to(self.device)

        padder = InputPadder(images.shape)
        return padder.pad(images)[0]

    def load_images(self, fn1, fn2):
        """ load and resize to multiple of 64 """
        images = [fn1, fn2]
        images = self.load_image_list(images)
        im1 = images[0, None]
        im2 = images[1, None]
        return im1, im2

    def compute_flow(self, im1, im2):
        padder = InputPadder(im1.shape)
        im1, im2 = padder.pad(im1, im2)
        _, flow12 = self.model(im1, im2, iters=20, test_mode=True)
        flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()

        return flow12
    
    def compute_flow_cv(self, im1, im2):

        im1 = self.load_image_cv(im1)
        im2 = self.load_image_cv(im2)
        flow12 = self.compute_flow(im1, im2)

        return flow12

    def viz(self, flo):

        # map flow to rgb image
        img = flow_viz.flow_to_image(flo)
        return img

