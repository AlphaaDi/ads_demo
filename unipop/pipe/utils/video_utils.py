import cv2
import numpy as np

def read_video(vc: cv2.VideoCapture) -> np.array:
    frames = []
    _, img = vc.read()
    while _:
        frames.append(img)
        _, img = vc.read()
    
    frames = np.array(frames)
    return frames
