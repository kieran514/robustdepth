import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class NUSCENESEVAL(MonoDataset):
    """Superclass for NuScenes dataset loader
    """
    def __init__(self, *args, **kwargs):
        super(NUSCENESEVAL, self).__init__(*args, **kwargs)

    def get_correct_index(self, index):
        index = self.filenames[index]
        return int(index)


    def get_color(self, index):
        sample = self.nusc.sample[index]
        cam = self.nusc.get('sample_data', token=sample['data']["CAM_FRONT"])['filename']
        color = self.loader(os.path.join(self.data_path, cam))

        return color

