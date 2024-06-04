import cv2
import pathlib

# from unipop.pipe.run import VideoEditer
# from unipop.pipe.run import AtlasWorker

ALLOWED_VIDEO_ID = ['woman']


class Editer:
    def __init__(self, video_id, gpu_id, storage):
        self.device = gpu_id
        self.video_id = video_id

        assert self.video_id in ALLOWED_VIDEO_ID
        self.editer = VideoEditer(self.video_id, gpu_id)
        self.storage = storage

    def place_logo(self, logo_id):
        # Replace this with the actual logo placement implementation
        print(f"Placing logo {logo_id} on video: {self.video_id} on {self.device} device")
        logo_path = pathlib.Path(self.storage) / (logo_id + '.png')
        output_video_path = self.editer.edit(logo_path)
        return output_video_path



class Processor:
    def __init__(self, video_id, gpu_id, storage):
        self.device = gpu_id
        self.video_id = video_id
        self.atlas_worker = AtlasWorker(video_id, gpu_id)
        self.storage = storage
        self.editer = None


    def process(self):
        # Replace this with the actual video processing implementation
        print(f"Processing video: {self.video_id} on {self.device} device")
        video_path = pathlib.Path(self.storage) / (self.video_id + '.mp4')
        meta_path = self.atlas_worker.run_train(video_path)
        return meta_path
