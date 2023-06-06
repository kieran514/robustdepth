import os
import json
import numpy as np
import PIL.Image as pil
import torch
from glob import glob

from .mono_dataset import MonoDataset


class CityscapesDataset(MonoDataset):
    """Cityscapes dataset - loading raw Cityscape and cropping needs 
    to be done inside get_color.
    """
    RAW_HEIGHT = 1024
    RAW_WIDTH = 2048

    def __init__(self, *args, **kwargs):
        super(CityscapesDataset, self).__init__(*args, **kwargs)

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits

        txt file is of format:
            aachen aachen_000000 4
        """
        city, frame_name = self.filenames[index].split()
        side = None
        return city, frame_name, side

    def check_depth(self):
        return False

    def load_intrinsics(self, city, frame_name):
        # adapted from sfmlearner
        split = "train" if self.is_train else "test"

        # print('city: ', city)
        # print('frame_name: ', frame_name)

        if split == 'train':
            camera_file = os.path.join(self.data_path, 'camera',
                                    split, city, frame_name[:-7] + '_*_camera.json')
            camera_file = glob(camera_file)[0]

        else:
            camera_file = os.path.join(self.data_path, 'camera',
                                   split, city, frame_name + '_camera.json')

        with open(camera_file, 'r') as f:
            camera = json.load(f)
        fx = camera['intrinsic']['fx']
        fy = camera['intrinsic']['fy']
        u0 = camera['intrinsic']['u0']
        v0 = camera['intrinsic']['v0']
        intrinsics = np.array([[fx, 0, u0, 0],
                               [0, fy, v0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]).astype(np.float32)
        intrinsics[0, :] /= self.RAW_WIDTH
        intrinsics[1, :] /= self.RAW_HEIGHT * 0.75
        return intrinsics

    def get_color(self, city, frame_name, side, spec, do_flip, is_sequence=True, foggy=False):
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides yet")

        color = self.loader(self.get_image_path(city, frame_name, spec, side, is_sequence, foggy))

        # crop down to cityscapes size
        w, h = color.size
        crop_h = h * 3 // 4
        color = color.crop((0, 0, w, crop_h))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_offset_framename(self, frame_name, offset=-2):
        city, seq, frame_num = frame_name.split('_')

        frame_num = int(frame_num) + offset
        frame_num = str(frame_num).zfill(6)
        return '{}_{}_{}'.format(city, seq, frame_num)

    def get_colors(self, city, frame_name, side, do_flip, spec, augs = False, foggy=False):
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides")

        if not self.is_train:

            color = self.get_color(city, frame_name, side, spec, do_flip, foggy=foggy)
            inputs = {}
            if augs:
                inputs[("weather", 0, -1)] = color
            else:
                inputs[("color", 0, -1)] = color

        else:
            color = self.get_color(city, frame_name, side, spec, do_flip, foggy=foggy)

            prev_name = self.get_offset_framename(frame_name, offset=-2)
            prev_color = self.get_color(city, prev_name, side, spec, do_flip, is_sequence=True, foggy=foggy)

            next_name = self.get_offset_framename(frame_name, offset=2)
            next_color = self.get_color(city, next_name, side, spec, do_flip, is_sequence=True, foggy=foggy)

            inputs = {}
            if augs:
                inputs[("weather", 0, -1)] = color
                inputs[("weather", -1, -1)] = prev_color
                inputs[("weather", 1, -1)] = next_color
            else:
                inputs[("color", 0, -1)] = color
                inputs[("color", -1, -1)] = prev_color
                inputs[("color", 1, -1)] = next_color

        return inputs

    def get_image_path(self, city, frame_name, spec, side, is_sequence=False, foggy=False):
        folder = "leftImg8bit" if not is_sequence else "leftImg8bit_sequence"
        split = "train" if self.is_train else "test"
        if spec == 'data':
            image_path = os.path.join(self.data_path, folder, split, city, frame_name + '_leftImg8bit' + self.img_ext)
        else:
            image_path = os.path.join(self.data_path, folder, split, city + '_' + spec, frame_name + '_leftImg8bit' + self.img_ext)

        if foggy and split == "test":
            image_path = os.path.join(self.data_path, folder, "fog_test", city, frame_name + '_leftImg8bit' + '_foggy_beta_0.02.png')

        return image_path
    
