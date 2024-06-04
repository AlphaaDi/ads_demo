import os
import time
import concurrent.futures
from sql_handler import SQLHandler
from video_processor import Editer, Processor
import threading
import argparse
import yaml
import torch

parser = argparse.ArgumentParser(description='',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config_path', default='config.yaml', type=str)


class VideoProcessorManager:
    def __init__(self, media_files_db, blob_storage_path, processers_idxs=[1], inter_task_wait=10):
        self.media_files_db = media_files_db
        self.blob_storage_path = blob_storage_path
        
        self.processers_idxs = processers_idxs
        self.num_processers = len(processers_idxs)

        self.inter_task_wait = inter_task_wait

    def process_video(self, video_id, gpu_id):
        print('process_video(self, video_id, gpu_id)', video_id, gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        video_processor = Processor(video_id, device, self.blob_storage_path)
        video_processor.process()
        return video_id


    def start_manager_work(self):
        self.sql_handler = SQLHandler(self.media_files_db)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_processers) as process_executor:
            while True:
                time.sleep(self.inter_task_wait)
                videos_to_process = self.sql_handler.get_videos_to_process(self.num_processers)
                print(videos_to_process)
                if videos_to_process:
                    process_tasks = []
                else:
                    continue

                for idx, video_id in enumerate(videos_to_process):
                    gpu_id = self.processers_idxs[idx]
                    task = process_executor.submit(self.process_video, video_id[0], gpu_id)
                    process_tasks.append(task)

                video_ids = []
                for task in concurrent.futures.as_completed(process_tasks):
                    try:
                        video_id = task.result()
                        video_ids.append(video_id)
                    except Exception as exc:
                        print(f"An error occurred during video processing: {exc}")

                self.sql_handler.set_videos_status_ready(video_ids)


class VideoEditorManager:
    def __init__(self, media_files_db, blob_storage_path, editors_idxs=[0], inter_task_wait=1):
        self.media_files_db = media_files_db
        self.blob_storage_path = blob_storage_path
        
        self.editors_idxs = editors_idxs
        self.num_placers = len(editors_idxs)
        
        self.inter_task_wait = inter_task_wait


    def process_video_place_logo(self, video_id, logo_id, gpu_id):
        print('process_video_place_logo(self, video_id, logo_id, gpu_id)')
        device = torch.device(f'cuda:{gpu_id}')
        video_processor = Editer(video_id, device, self.blob_storage_path)
        output_video_path = video_processor.place_logo(logo_id)
        return output_video_path

    def start_manager_work(self):
        self.sql_handler = SQLHandler(self.media_files_db)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_placers) as place_logo_executor:
            while True:
                time.sleep(self.inter_task_wait)
                oldest_logo_placement_tasks = self.sql_handler.get_oldest_logo_placement_tasks(self.num_placers)
                print('oldest_logo_placement_tasks', oldest_logo_placement_tasks)
                if oldest_logo_placement_tasks:
                    place_logo_tasks = []
                else:
                    continue

                for idx, (video_id, logo_id, timestamp) in enumerate(oldest_logo_placement_tasks):
                    gpu_id = self.editors_idxs[idx]
                    task = place_logo_executor.submit(self.process_video_place_logo, video_id, logo_id, gpu_id)
                    place_logo_tasks.append(task)

                for (video_id, logo_id, timestamp), task in zip(oldest_logo_placement_tasks, concurrent.futures.as_completed(place_logo_tasks)):
                    try:
                        output_video_path = task.result()
                        self.sql_handler.remove_placement_task(video_id, logo_id)
                        self.sql_handler.insert_logo_placement_result(video_id, logo_id, timestamp, output_video_path)
                    except Exception as exc:
                        print(f"An error occurred during video editing: {exc}")


class GlobalManager:
    def __init__(self, config):
        self.config = config
        self.video_processor_manager = VideoProcessorManager(
            config['media_files_db'],
            config['blob_storage_path'],
            **config['video_processor_manager_kwargs']
        )
        self.video_editor_manager = VideoEditorManager(
            config['media_files_db'],
            config['blob_storage_path'],
            **config['video_editor_manager_kwargs']
        )

    def run(self):
        video_processing_thread = threading.Thread(target=self.video_processor_manager.start_manager_work)
        logo_placement_thread = threading.Thread(target=self.video_editor_manager.start_manager_work)

        video_processing_thread.start()
        logo_placement_thread.start()

        # video_processing_thread.run()
        # logo_placement_thread.run()

        video_processing_thread.join()
        logo_placement_thread.join()


if __name__ == "__main__":
    args = parser.parse_args()
    config = yaml.full_load(open(args.config_path, 'r'))

    blob_storage_path = config['blob_storage_path']
    
    video_processing_manager = GlobalManager(config)
    
    video_processing_manager.run()