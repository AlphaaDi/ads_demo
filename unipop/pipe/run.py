import os
import sys
import pathlib
import numpy as np
from tqdm import tqdm
import pickle
import glob
import uuid


import cv2
import torch

from unipop.xmen.run import PropagationxMem
from unipop.mmdet.mask2former import Mask2Former
from unipop.atlas.train import main as trainAtlas
from unipop.atlas.only_edit import main as editAtlas
from unipop.raft.raft_wrapper import RAFTWrapper
from unipop.pipe.utils.video_utils import read_video
from unipop.pipe.utils.atlas_config import config as atlasConfig
from unipop.mmdet.mask2former import config as mask2FormerConfig
from unipop.mmdet.mask2former import checkpoint as mask2FormerWeights
from unipop.xmen.run import XMEN_WEIGHTS
from unipop.pipe.utils.overlay import mark_image_with_logo



class VideoEditer(object):

    def __init__(self, id, device, storage='/home/davinci/work/unipopcorn/meta') -> None:
        self.device = device
        self._id = id
        self.meta_path = pathlib.Path(storage) / str(id)
        #checl here for planes later
        self.magic_helper = {
            'woman' : [0,1, False, True],
            'cruise' : [0,1, True, False],
            'boat'  : [0,2, False, True],
        }
        self.planes_path = self.meta_path / 'plane'
        glob_regex = self.planes_path / '*pickle'
        pickle_path = glob.glob(str(glob_regex))[0]
        with open(pickle_path, 'rb') as f:
            self.planes = pickle.load(f)
#debug only
#       logo = cv2.imread(str('/home/davinci/work/unipopcorn/data/logo/lv.png'), cv2.IMREAD_UNCHANGED) 
#       logo = cv2.resize(logo, (448, 448))

#       i = self.magic_helper[self._id][0]
#       j = self.magic_helper[self._id][1]
#       self._i = i
#       masks = self.planes['frame_masks'][i]
#       norms = self.planes['new_parameters'][i]
#       img_path = self.meta_path / 'images' / (str(i).zfill(6) + '.jpg')
#       img = cv2.imread(str(img_path))
#       planed_logo, res = mark_image_with_logo(img, [masks[j]], [norms[j]], logo)
#       cv2.imshow('res', res)
#       cv2.waitKey()

    def _run_atlas(self, edit_frame : np.array, index_frame : int) -> pathlib.Path :
        atlas_fld = self.meta_path / 'atlasdir' 
        all_atlasses = os.listdir(atlas_fld)
        all_atlasses = sorted(all_atlasses)
        last_atlas = all_atlasses[-1]
        atlas_train_fld = atlas_fld / last_atlas
        edit_frame_path = self.meta_path / f'{str(uuid.uuid1())}_edit_to_prop.png'
        frames_fld = self.meta_path / 'images'
        masks_fld  = self.meta_path / 'maskdir'
        out_fld = self.meta_path / 'editdir'
        edit_frame = cv2.resize(edit_frame, (768, 432))
        cv2.imwrite(str(edit_frame_path), edit_frame)
        out_folder_fin = editAtlas(str(atlas_train_fld), True, str(frames_fld),
                  str(masks_fld), str(edit_frame_path), 0, 0,
                  self._i, str(out_fld), self._id,
                  self.magic_helper[self._id][2], self.magic_helper[self._id][3], self.device)
        video_edit_path_glob = pathlib.Path(out_folder_fin) / '*mp4'
        video_edit_path = glob.glob(str(video_edit_path_glob))[0]
        return pathlib.Path(video_edit_path)



    def edit(self, logo_path: pathlib.Path) -> pathlib.Path:
        logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED) 
        logo = cv2.resize(logo, (448, 448))

        i = self.magic_helper[self._id][0]
        j = self.magic_helper[self._id][1]
        self._i = i
        masks = self.planes['frame_masks'][i]
        norms = self.planes['new_parameters'][i]
        img_path = self.meta_path / 'images' / (str(i).zfill(6) + '.jpg')
        img = cv2.imread(str(img_path))
        planed_logo, res = mark_image_with_logo(img, [masks[j]], [norms[j]], logo)
        canv_rgba = planed_logo[0]
        res = self._run_atlas(canv_rgba, i)
        return res
        pass

class PreAtlasExtraction(object):
    def __init__(self, device: torch.device) -> None:

        self.flow = RAFTWrapper(
            model_path='/home/davinci/work/unipopcorn/unipop/raft/models/raft-things.pth',
            device=device,
            max_long_edge=1000
        ) 
        self.propagation = PropagationxMem(XMEN_WEIGHTS, device)
        self.salient = Mask2Former(mask2FormerWeights, mask2FormerConfig, device)
        self._inf_shape = (1024, 1024)
    
    def _preprocess(self, img : np.array) -> np.array:
        self._base_shape = (img.shape[1], img.shape[0])
        img = cv2.resize(img, self._inf_shape)
        return img

    def _postprocess(self, img : np.array) -> np.array:
        return cv2.resize(img, self._base_shape)
    
    def _mask_estimate(self, video: np.array) -> np.array:
        first_mask = self.salient(self._preprocess(video[0]))

        masks = self.propagation([self._preprocess(frame) for frame in video], first_mask)
        return np.array([self._postprocess(mask) for mask in masks])

    def _flow_estimate(self, video: np.array) -> np.array:
        flows_forward  = []
        flows_backward = []
        for i in range(len(video)-1):
            img1 = video[i]
            img2 = video[i+1]

            flow12 = self.flow.compute_flow_cv(img1, img2)
            flow21 = self.flow.compute_flow_cv(img2, img1)

            flows_forward.append(flow12)
            flows_backward.append(flow21)
        
        flows = np.array([flows_forward, flows_backward])
        return flows

    def forward(self, video: np.array) -> np.array: 
        flows = self._flow_estimate(video)
        masks = self._mask_estimate(video)
        del self.flow
        del self.propagation
        del self.salient
        return flows, masks


    def forward_path(self, video_path) -> np.array: 
        vc = cv2.VideoCapture(str(video_path))
        frames = read_video(vc)
        return self.forward(frames)

class AtlasWorker(object):
    def __init__(self, id, device, storage='/home/davinci/work/unipopcorn/meta') -> None:

        self.device = device 

        self.metaExtractor = PreAtlasExtraction(self.device)

        storage = pathlib.Path(storage)
        self.meta_path = storage / pathlib.Path(str(id))

        self.imgs_path = self.meta_path / pathlib.Path('images')
        self.flow_path = self.meta_path / pathlib.Path('flowdir')
        self.mask_path = self.meta_path / pathlib.Path('maskdir')
        self.atlas_path= self.meta_path / pathlib.Path('atlasdir')
        self.meta_flag = self.meta_path / pathlib.Path('done')

        print('Checking that id never been used before')
#       assert not os.path.isdir(self.meta_path)

        os.makedirs(self.meta_path, exist_ok=True)
        os.makedirs(self.flow_path, exist_ok=True)
        os.makedirs(self.mask_path, exist_ok=True)
        os.makedirs(self.atlas_path, exist_ok=True)
        os.makedirs(self.imgs_path, exist_ok=True)

        pass


    def _flow_save(self, video_flow: np.array) -> pathlib.Path:
        flow12 = video_flow[0]
        flow21 = video_flow[1]

        for i in range(len(flow12)):
            fn1 = str(i).zfill(6)
            fn2 = str(i+1).zfill(6)

            out_flow12_path = self.flow_path / f'{fn1}_{fn2}.npy' 
            out_flow21_path = self.flow_path / f'{fn2}_{fn1}.npy'

            np.save(out_flow12_path, flow12[i])
            np.save(out_flow21_path, flow21[i])
        return self.flow_path

    def _mask_save(self, video_masks: np.array) -> pathlib.Path:
        for i, mask in enumerate(video_masks):
            mask_name = str(i).zfill(6) + '.png'
            mask_path = self.mask_path / pathlib.Path(mask_name)
            cv2.imwrite(str(mask_path), mask)
        return self.mask_path
        pass    

    def _video_save(self, video: np.array) -> pathlib.Path:
        for i, img in enumerate(video):
            img_name = str(i).zfill(6) + '.jpg'
            img_path = self.imgs_path / pathlib.Path(img_name)
            cv2.imwrite(str(img_path), img)
        return self.imgs_path
    
    def _read_flow(self, flow_path: pathlib.Path) -> np.array:
        flow12 = []
        flow21 = []
        
        flows = glob.glob(str(flow_path / '*.npy'))
        l = len(flows) // 2
        for i in range(l):
            fn1 = str(i).zfill(6)
            fn2 = str(i+1).zfill(6)

            flow12_path = flow_path / f'{fn1}_{fn2}.npy'
            flow21_path = flow_path / f'{fn2}_{fn1}.npy'
            flow12.append(np.load(flow12_path))
            flow21.append(np.load(flow21_path))
        flows = [flow12, flow21]
        flows = np.array(flows)
        return flows

    def _read_mask(self, mask_path: pathlib.Path) -> np.array:
        out_mask = []
        
        video_mask = glob.glob(str(mask_path / '*png'))
        l = len(video_mask)
        for i in range(l):
            mask_name = str(i).zfill(6) + '.png'
            mask = mask_path /  mask_name
            mask = cv2.imread(str(mask_path), 0)
            out_mask.append(mask)
        out_mask = np.array(out_mask)
        return out_mask

    def _binarieze_mask(self, video_mask : np.array) -> np.array:
        out_mask = []
        for mask in video_mask:
            mask = (mask > 0).astype(np.uint8) * 255
            out_mask.append(mask)
        return np.array(out_mask)
    
    def _configure_atlas(self):
        self.atlasConfig = atlasConfig
        self.atlasConfig['results_folder_name'] = str(self.atlas_path)
        self.atlasConfig['data_folder'] = str(self.meta_path)

    def run_train(self, video_path: pathlib.Path) -> pathlib.Path:
        video_imgs = read_video(cv2.VideoCapture(str(video_path)))
        if os.path.exists(self.meta_flag):
            video_flow = self._read_flow(self.flow_path)
            video_mask = self._read_mask(self.mask_path)
        else:
            video_flow, video_mask = self.metaExtractor.forward(video_imgs)
            video_mask = self._binarieze_mask(video_mask)

            self._video_save(video_imgs)
            self._mask_save(video_mask)
            self._flow_save(video_flow)
            os.makedirs(self.meta_flag)
            del self.metaExtractor
        self._configure_atlas()
        with torch.enable_grad():
            trainAtlas(self.atlasConfig, self.device)
        return self.meta_path
            




if __name__ == "__main__":



#check atlas worker

#   device = torch.device('cuda:0')
#   worker = AtlasWorker('library', device)
#   path = worker.run_train(pathlib.Path('/home/davinci/work/unipopcorn/data/library-846.mp4'))
#   print(path)

#check for VideoEditor
    device = torch.device('cuda:1')
    editor = VideoEditer('woman', device)
    path = editor.edit(pathlib.Path('/home/davinci/work/unipopcorn/data/logo/lv.png'))
    print(path)
