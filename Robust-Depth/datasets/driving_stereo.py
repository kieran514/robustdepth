import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class DRIVINGSTEREO(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(DRIVINGSTEREO, self).__init__(*args, **kwargs)

    def get_color(self, line, stereo_split):
        color = self.loader(os.path.join(self.data_path, stereo_split, 'left-image-full-size', line))

        return color

