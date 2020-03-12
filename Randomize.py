import numpy as np
import glob, os, shutil


class RandomizeTask:
    def __init__(self, task_dir: str = "/home/gal/White_Noise_fMRI/For_task"):
        self.task_dir = task_dir

    def init_subdirs(self, task_dir: str):
        for d in os.listdir(task_dir):
            if os.path.isdir(f"{task_dir}/{d}"):
                if "Videos" in d:
                    videos = f"{task_dir}/{d}"
                elif "Images" in d:
                    images = f"{task_dir}/{d}"
                elif "Audio" in d:
                    audio = f"{task_dir}/{d}"
        return videos, images, audio

    def get_vis(self, videos: str, images: str):
        vid_files = glob.glob(f"{videos}/*")
        img_files = glob.glob(f"{images}/*")
        vis_files = []
        for f in vid_files:
            vis_files.append(f)
        for f in img_files:
            vis_files.append(f)
        return vis_files

    def get_aud(self, audio: str):
        aud_files = glob.glob(f"{audio}/*")
        return aud_files

    def gen_rand_dir(self, task_dir: str):
        for d in ["Random_Vis", "Random_Aud"]:
            if not os.path.isdir(f"{task_dir}/{d}"):
                os.mkdir(f"{task_dir}/{d}")
                print(f'Created Random files directory at {f"{task_dir}/{d}"}')

