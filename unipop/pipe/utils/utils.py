import cv2
import numpy as np
from pathlib import Path
import os
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt



def get_mask_center(mask):
    M = cv2.moments(mask)
    cX = int(M["m10"] / (M["m00"] + 0.001))
    cY = int(M["m01"] / (M["m00"] + 0.001))
    return (cX, cY)


def get_biggest_rectangle(mask):
    where = np.array(np.where(mask))

    x1, y1 = np.amin(where, axis=1)
    x2, y2 = np.amax(where, axis=1)

    return x1,y1,x2,y2


def add(prev, nxt, level=1.0):
    return (prev + level * nxt).clip(0, 255).astype(np.uint8)


def alpha_blend(img1, img2, mask_):
    assert img1.shape == img2.shape
    assert img1.shape[:2] == mask_.shape

    if mask_.dtype == np.uint8:
        mask = mask_.astype(np.float32) / 255.0
    else:
        mask = mask_

    if len(img1.shape) == 3:
        h, w, c = img1.shape
        tmp = np.empty_like(img1)
        for i in range(c):
            tmp[:, :, i] = img1[:, :, i] * (1-mask) + img2[:, :, i] * mask
        return tmp
    else:
        h, w = img1.shape
        return (img1 * (1-mask) + img2 * mask).astype(img1.dtype)


def get_center_rad_inscribed(mask):
    mask = mask.astype(np.uint8)
    mask_pad = np.pad(mask,1)
    dist_map = cv2.distanceTransform(mask_pad, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, r, _, center = cv2.minMaxLoc(dist_map)
    center = (center[0] , center[1])

    return ((center[0]-r, center[1]-r), (center[0] + r, center[1] + r)), r, center


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim)
    return resized
    

def get_basis(v_norm):
    v_norm = normalize(v_norm[None])[0]

    v_rnd = np.array([0,1,0])
    v_rnd = normalize(v_rnd[None])[0]
    proj_len = (v_norm * v_rnd).sum()

    v_first = v_rnd - v_norm * proj_len
    v_first = normalize(v_first[None])[0]

    v_second = -np.cross(v_norm, v_first)
    return v_norm, v_first, v_second


def get_center_matrix(w,h):
    return np.matrix([
                    [1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [-w/2,-h/2,0,1]])


def get_trans_rot_matrix(v_norm, v_first, v_second, z_deep = 500):
    mat = np.zeros((4,4))
    mat[0] = list(v_second) + [0,]
    mat[1] = list(v_first) + [0,]
    mat[2] = list(v_norm) + [0,]
    mat[3] = [0, 0, z_deep, 0]
    mat = np.matrix(mat)
    return mat


def get_perspective_matrix(img):
    fovy = np.deg2rad(90)
    ctg = 1/ np.tan(fovy / 2)
    aspect = img.shape[1] / img.shape[0]
    f = 1000
    n = 0.01

    perspective = np.array(
        [
            [ctg / aspect,0,0,0],
            [0,ctg,0,0],
            [0,0,(f+n)/(f-n),1],
            [0,0,-2*f*n/(f-n),0]
        ]
    )
    perspective = np.matrix(perspective)
    return perspective


def crop_logo(ret):
    ret_gray = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    bin_mask = (ret_gray > 0).astype(np.uint8) * 255
    x1,y1,x2,y2 = get_biggest_rectangle(bin_mask)
    crop = ret[x1:x2,y1:y2]
    return crop


def resize_respect_size(crop, w):
    wc, hc = crop.shape[1], crop.shape[0]
    if wc > hc:
        crop_r = image_resize(crop, width=w)
    else:
        crop_r = image_resize(crop, height=w)
    return crop_r

def mark_image_with_logo(img, masks, planes, logo):
    w, h, _ = logo.shape
    masks_info = []
    for i in range(len(masks)):

        mask = masks[i]
        plane_norm = planes[i]
#       mask_center = get_mask_center(mask)

        plane_norm[0], plane_norm[1], plane_norm[2] = plane_norm[0], -plane_norm[2], plane_norm[1]
        v_norm, v_first, v_second = get_basis(plane_norm)

        src_pts = [[0,0],
                   [0, logo.shape[0]],
                   [logo.shape[1], 0],
                   [logo.shape[1], logo.shape[0]]]

        mat_center = get_center_matrix(w, h)
        
        mat = get_trans_rot_matrix(v_norm, v_first, v_second)
        
        perspective = get_perspective_matrix(img)

        src_pts_01 = [src + [0,1] for src in src_pts]
        dst_pts = np.array(src_pts_01) @ mat_center @ mat @ perspective
        dst_pts = dst_pts[:,:2] / dst_pts[:,3]

        dst_pts += 1
        dst_pts[:,0] *= img.shape[1] / 2
        dst_pts[:,1] *= img.shape[0] / 2

        src_pts = np.array(src_pts).astype(np.float32)
        dst_pts = np.array(dst_pts).astype(np.float32)
        
        mat_transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
        logo_perspective = cv2.warpPerspective(logo, mat_transform, (img.shape[1], img.shape[0]))

        logo_cropped = crop_logo(logo_perspective) 

        biggest_rectangle, r, center = get_center_rad_inscribed(mask)
        w = int(r * (2 ** (1 / 2)))  # square side from outer circle (math, not magic, also on m)
        crop_r = resize_respect_size(logo_cropped, w)

        canva = np.zeros((img.shape[0], img.shape[1], 3)).astype(np.uint8)
        x1,y1 = center[0] - w//2, center[1] - w//2

        w_crop, h_crop, _ = crop_r.shape
        canva[y1 : y1 + w_crop, x1 : x1 + h_crop] = crop_r
        
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        xcoords = np.arange(canva.shape[1], dtype=np.float32)
        ycoords = np.arange(canva.shape[0], dtype=np.float32)
        mapX, mapY = np.meshgrid(xcoords, ycoords)
        
        canva = cv2.remap(canva, mapX , mapY + (img_l / 20).astype(np.float32), cv2.INTER_LINEAR)

        canva_l = cv2.cvtColor(canva, cv2.COLOR_BGR2GRAY)
        img[canva_l > 0] = canva[canva_l > 0]
        masks_info.append((x1,y1, crop_r))

    return img, masks_info

# base_meta = Path('/Users/dkamarouski/work/data/davis_meta/rollerblade')
# base_folder = base_meta
# img = cv2.imread('15_image_0.png')[:,:,::-1].astype(np.uint8)
# masks = np.load(base_folder / '15_plane_masks_0.npy')
# planes = np.load(base_folder / '15_plane_parameters_0.npy')
# logo = cv2.imread('logo.png')[:,:,::-1]
# image_with_marked_logo, masks_info = mark_image_with_logo(img, masks, planes, logo)
# 
# plt.imshow(image_with_marked_logo)
# plt.show()
