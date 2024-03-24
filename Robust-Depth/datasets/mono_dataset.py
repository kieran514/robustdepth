import os
import random
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
from PIL import Image  
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms
import pdb
import math
import torchvision.transforms as T

cv2.setNumThreads(0)

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 robust_val=False,
                 img_ext='.jpg',
                 mask_noise=False,
                 feat_warp=False,
                 vertical=False,
                 tiling=False,
                 do_rain_aug=False,
                 do_fog_aug=False,
                 do_night_aug=False,
                 do_scale_aug=False,
                 do_blur_aug=False,
                 do_erase_aug=False,
                 do_color_aug=False,
                 do_gauss_aug=False,
                 do_shot_aug=False,
                 do_impulse_aug=False,
                 do_defocus_aug=False,
                 do_glass_aug=False,
                 do_zoom_aug=False,
                 do_snow_aug=False,
                 do_frost_aug=False,
                 do_elastic_aug=False,
                 do_pixelate_aug=False,
                 do_jpeg_comp_aug=False,
                 do_flip_aug=False,
                 do_dawn_aug=False,
                 do_dusk_aug=False,
                 do_ground_snow_aug=False,
                 do_greyscale_aug=False,
                 is_robust_test=False,
                 do_Red_aug=False,
                 do_Green_aug=False,
                 do_Blue_aug=False,
                 robust_augment=False,
                 foggy=False,
                 stereo_split=False):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.nuscenes_data = "/media/kieran/SSDNEW/Base-Model/data/nuScenes_RAW"
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.sensor = 'CAM_FRONT'
        self.stereo_split = stereo_split

        self.interp = T.InterpolationMode.LANCZOS

        self.foggy = foggy

        self.frame_idxs = frame_idxs
        self.robust_val = robust_val

        self.is_train = is_train
        self.img_ext = img_ext
        self.is_robust_test = is_robust_test
        self.robust_augment = robust_augment

        self.vertical = vertical
        self.tiling = tiling
        self.do_rain_aug = do_rain_aug
        self.do_fog_aug = do_fog_aug
        self.do_night_aug = do_night_aug
        self.do_scale_aug = do_scale_aug
        self.do_blur_aug = do_blur_aug
        self.do_erase_aug = do_erase_aug
        self.do_color_aug = do_color_aug
        self.do_flip_aug = do_flip_aug
        self.do_greyscale_aug = do_greyscale_aug
        self.do_Red_aug = do_Red_aug
        self.do_Green_aug = do_Green_aug
        self.do_Blue_aug = do_Blue_aug

        self.do_gauss_aug = do_gauss_aug
        self.do_shot_aug = do_shot_aug
        self.do_impulse_aug = do_impulse_aug
        self.do_defocus_aug = do_defocus_aug
        self.do_zoom_aug = do_zoom_aug
        self.do_snow_aug = do_snow_aug
        self.do_glass_aug = do_glass_aug
        self.do_frost_aug = do_frost_aug
        self.do_elastic_aug = do_elastic_aug
        self.do_pixelate_aug = do_pixelate_aug
        self.do_jpeg_comp_aug = do_jpeg_comp_aug
        self.do_dawn_aug = do_dawn_aug
        self.do_dusk_aug = do_dusk_aug
        self.do_ground_snow_aug = do_ground_snow_aug

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.mask_noise = mask_noise
        self.feat_warp = feat_warp

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

    def tile_crop(self, color_aug_f, do_tiling, selection, order):
        if do_tiling:
            _, h, w = color_aug_f.shape
            height_selection = selection[0]
            width_selection = selection[1]
            height_split = h // height_selection 
            width_split = w // width_selection
            selection_prod = np.prod(selection)
            # has to be divisabel by 3 and 4
            tiles = [color_aug_f[:, x:x+height_split, y:y+width_split] for x in range(0, h, height_split) for y in range(0, w, width_split)]
            tiles = [tiles[i] for i in order]
            width_cat = [torch.cat(tiles[i:width_selection + i], dim=2) for i in range(0, selection_prod, width_selection)]
            final = torch.cat(width_cat, dim=1)
        else:
            final = color_aug_f
        return final

    def vertical_crop(self, color_aug_f, do_vertical, rand_w):
        '''Applies a vertical dependence augmentation
        '''
        if do_vertical and rand_w > 0:
            output_image = []
            in_h = color_aug_f.shape[1]
            cropped_y = [0, int(rand_w * in_h), in_h]
            cropped_image = [color_aug_f[:, cropped_y[n]:cropped_y[n+1], :] for n in range(2)]
            a = cropped_image[::-1]
            output_image = torch.cat(a, dim=1)
        else:
            output_image = color_aug_f
        return output_image

    def preprocess(self, inputs, color_aug, erase_aug, do_vertical, do_scale, small, height_re, width_re, box_HiS, do_flip, order, do_tiling, selection, rand_w, spec):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        self.scale_resize = {}
        if do_scale:
            if small:
                for i in range(self.num_scales):
                    s = 2 ** i
                    self.scale_resize[i] = transforms.Resize((height_re // s, width_re // s), interpolation=self.interp)
            else:
                for i in range(self.num_scales):
                    s = 2 ** i
                    self.scale_resize[i] = transforms.Resize((height_re // s, width_re // s), interpolation=self.interp)

        for k in list(inputs):
            if ("color" in k) or ("augmented" in k):
                n, im, i = k
                for i in range(self.num_scales):
                    if do_scale:
                        if n == "augmented": # augmented should always be in the inputs even if its just the original color data
                            if small: 
                                inputs[("augmented", im, i)] = self.scale_resize[i](inputs[(n, im, i - 1)])
                                inputs[("aug", im, i)] = inputs[("augmented", im, i)]
                            else:
                                inputs[("augmented", im, i)] = self.scale_resize[i](inputs[(n, im, i - 1)])
                                inputs[("aug", im, i)] = inputs[("augmented", im, i)].crop(box_HiS[i])
                        elif n == "color":
                            if small:
                                inputs[("color", im, i)] = self.scale_resize[i](inputs[(n, im, i - 1)]) # without augmented
                                inputs[("scale_aug", im, i)] = inputs[("color", im, i)] # without augmented augments
                            else:
                                inputs[("color", im, i)] = self.scale_resize[i](inputs[(n, im, i - 1)]).crop(box_HiS[i]) # without augmented
                                inputs[("scale_aug", im, i)] = inputs[("color", im, i)] # without augmented
                            
                            inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
                    else:
                        if n == "color":
                            inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)]) # n = color
                            inputs[("scale_aug", im, i)] = inputs[(n, im, i)] # n = color
                        elif n == "augmented":
                            inputs[("augmented", im, i)] = self.resize[i](inputs[(n, im, i - 1)])
                            inputs[("aug", im, i)] = inputs[("augmented", im, i)]

        for k in list(inputs):
            f = inputs[k]
            if ("color" in k) or ("aug" in k) or ("scale_aug" in k):
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                if n == "aug":
                    if not small: # this is the original method and will run if do scale is false and it would run if do scale is true + not small 
                        if inputs[(n, im, i)].sum() == 0:
                            inputs[("color_" + n, im, i)] = inputs[(n, im, i)]
                        else:
                            f = color_aug(f)
                            if spec in ['fog', 'fog+night', 'rain+fog', 'rain+fog+night', 'dusk+fog', 'dawn+fog', 'dawn+rain+fog', 'dusk+rain+fog']:
                                img_ = self.to_tensor(f)[[2,1,0],:,:]
                            else:
                                img_ = self.to_tensor(f)
                            inputs[("color_" + n, im, i)] = self.tile_crop(self.vertical_crop(erase_aug(img_), do_vertical, rand_w), do_tiling, selection, order)
                    elif small:
                        if i != -1:
                            f = color_aug(f)
                            LoS_part = self.to_tensor(f)
                            _, height_, width_ = inputs[("color", im, i)].size()
                            width_re, height_re = inputs[("resize", i)]
                            width_re, height_re = int(width_re.item()), int(height_re.item())
                            point1 = int(2 * width_re - width_)
                            point2 = int(2 * height_re - height_)
                            Tensor_LoS = torch.zeros(3, height_, width_)
                            Tensor_LoS[:, 0:height_re, 0:width_re] = LoS_part
                            Tensor_LoS[:, height_re:height_, 0:width_re] = LoS_part[:, point2:height_re, 0:width_re]
                            Tensor_LoS[:, 0:height_re, width_re:width_] = LoS_part[:, 0:height_re, point1:width_re]
                            Tensor_LoS[:, height_re:height_, width_re:width_] = LoS_part[:, point2:height_re, point1:width_re]
                            inputs[(n, im, i)] = Tensor_LoS
                            if inputs[(n, im, i)].sum() == 0:
                                inputs[("color_" + n, im, i)] = inputs[(n, im, i)]
                            else:
                                if spec in ['fog', 'fog+night', 'rain+fog', 'rain+fog+night', 'dusk+fog', 'dawn+fog', 'dawn+rain+fog', 'dusk+rain+fog']:
                                    inputs[(n, im, i)] = inputs[(n, im, i)][[2,1,0],:,:]
                                inputs[("color_" + n, im, i)] = self.tile_crop(self.vertical_crop(erase_aug(inputs[(n, im, i)]), do_vertical, rand_w), do_tiling, selection, order)
                        else:
                            inputs[("color_" + n, im, -1)] = 0
                elif n == "scale_aug":
                    if do_scale:
                        if small:
                            if i != -1:
                                LoS_part = self.to_tensor(f)
                                _, height_, width_ = inputs[("color", im, i)].size()
                                width_re, height_re = inputs[("resize", i)]
                                width_re, height_re = int(width_re.item()), int(height_re.item())
                                point1 = int(2 * width_re - width_)
                                point2 = int(2 * height_re - height_)
                                Tensor_LoS = torch.zeros(3, height_, width_)
                                Tensor_LoS[:, 0:height_re, 0:width_re] = LoS_part
                                Tensor_LoS[:, height_re:height_, 0:width_re] = LoS_part[:, point2:height_re, 0:width_re]
                                Tensor_LoS[:, 0:height_re, width_re:width_] = LoS_part[:, 0:height_re, point1:width_re]
                                Tensor_LoS[:, height_re:height_, width_re:width_] = LoS_part[:, point2:height_re, point1:width_re]
                                inputs[(n, im, i)] = Tensor_LoS
                            else:
                                inputs[(n, im, -1)] = 0
    def __len__(self):
        return len(self.filenames)

    def load_intrinsics_kitti(self, folder, frame_index):
        return self.K.copy()

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,

        <frame_id> is:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        if self.is_train or self.robust_val:
            dict_color_augs = {'blur': self.do_blur_aug, 'defocus_blur': self.do_defocus_aug, 'elastic_transform': self.do_elastic_aug, 'fog': self.do_fog_aug, 'fog+night': self.do_fog_aug and self.do_night_aug, 
            'frost': self.do_frost_aug, 'gaussian_noise': self.do_gauss_aug, 'glass_blur': self.do_glass_aug, 'impulse_noise': self.do_impulse_aug, 'jpeg_compression': self.do_jpeg_comp_aug, 'night': self.do_night_aug,
            'pixelate': self.do_pixelate_aug, 'rain': self.do_rain_aug, 'rain+fog': self.do_rain_aug and self.do_fog_aug, 'rain+fog+night': self.do_rain_aug and self.do_fog_aug and self.do_night_aug,
            'rain+night': self.do_rain_aug and self.do_night_aug, 'shot_noise': self.do_shot_aug, 'snow': self.do_snow_aug, 'zoom_blur': self.do_zoom_aug, 'color': self.do_color_aug, 'dusk':self.do_dusk_aug, 'dawn':self.do_dawn_aug,
            'ground_snow':self.do_ground_snow_aug, 'dawn+rain':self.do_dawn_aug and self.do_rain_aug, 'dusk+rain':self.do_dusk_aug and self.do_rain_aug, 'dusk+fog':self.do_dusk_aug and self.do_fog_aug,
            'dawn+fog':self.do_dawn_aug and self.do_fog_aug, 'dawn+rain+fog':self.do_dawn_aug and self.do_rain_aug and self.do_fog_aug, 'dusk+rain+fog':self.do_dusk_aug and self.do_rain_aug and self.do_fog_aug,
            'greyscale':self.do_greyscale_aug, 'R':self.do_Red_aug, 'G':self.do_Green_aug, 'B':self.do_Blue_aug,'clear': self.do_scale_aug or self.tiling or self.vertical or self.do_erase_aug or self.do_flip_aug}
            
            valid_items = [key for key, value in dict_color_augs.items() if value]

            if len(valid_items) == 1:
                spec = valid_items[0]
            elif len(valid_items) == 0:
                spec = 'data'
            else:
                spec = random.choice(valid_items)
            
            if spec == 'color':
                spec = 'data'
                do_color_aug = True
                self.brightness = (0.5, 3) # extreme
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            else:
                do_color_aug = False
                
            if spec == 'clear':
                spec = 'data'
                do_geomtric = True
            else:
                do_geomtric = False

        else:
            spec = 'data'
            do_color_aug = False
            
        do_flip = self.is_train and self.do_flip_aug and random.random() > 0.5

        if self.is_train:
            dict_geometry_augs = {'tiling': self.tiling, 'scale': self.do_scale_aug, 'erase': self.do_erase_aug, 'vertical': self.vertical}

            valid_items = [key for key, value in dict_geometry_augs.items() if value]

            if len(valid_items) == 1:
                geometric = valid_items[0]
            elif len(valid_items) == 0:
                geometric = ''
            else:
                geometric = random.choice(valid_items) 

            if do_geomtric:
                do_vertical = (geometric == 'vertical')
                do_tiling = (geometric == 'tiling')
                do_scale = (geometric == 'scale')
                small = do_scale and random.random() > 0.5
                rand_erase = (geometric == 'erase')
            else:
                do_vertical = (geometric == 'vertical' and random.random() > 0.5)
                do_tiling = (geometric == 'tiling' and random.random() > 0.5)
                do_scale = (geometric == 'scale' and random.random() > 0.5)
                small = do_scale and random.random() > 0.5
                rand_erase = (geometric == 'erase' and random.random() > 0.5)

            if do_vertical:
                geometric = 'vertical'
            elif do_tiling:
                geometric = 'tiling'
            elif do_scale:
                geometric = 'scale'
            elif rand_erase:
                geometric = 'erase'
            else:
                geometric = 'None'
            
        else:
            do_vertical = False
            do_tiling = False
            do_scale = False
            small = False
            rand_erase = False
            geometric = ''
        
        if do_scale:
            if small:
                ra = 0.7
                rb = 0.9
                resize_ratio = (rb - ra) * random.random() + ra
                height_re = int(self.height * resize_ratio)
                width_re = int(self.width * resize_ratio)
                dx = 0
                dy = 0
                for i in range(self.num_scales):
                    s = 2 ** i
                    dx_s = dx // s
                    dy_s = dy // s
                    inputs[("dxy", i)] = torch.Tensor((dx_s, dy_s))
                    inputs[("resize", i)] = torch.Tensor((width_re // s, height_re // s))
                box_HiS = 0
            else:
                box_HiS = []
                ra = 1.1
                rb = 2.0
                resize_ratio = (rb - ra) * random.random() + ra
                height_re = int(self.height * resize_ratio)
                width_re = int(self.width * resize_ratio)
                height_d = height_re - self.height
                width_d = width_re - self.width
                dx = int(width_d * random.random())
                dy = int(height_d*random.random())
                for i in range(self.num_scales):
                    s = 2 ** i
                    dx_s = dx // s
                    dy_s = dy // s
                    inputs[("dxy", i)] = torch.Tensor((dx_s, dy_s))
                    inputs[("resize", i)] = torch.Tensor((width_re // s, height_re // s))
                    box_HiS.append((dx_s, dy_s, dx_s + (self.width // s), dy_s + (self.height // s)))
        else:
            height_re=0
            width_re=0
            dx=0
            dy=0
            box_HiS = 0
            for i in range(self.num_scales):       
                inputs[("dxy", i)] = torch.Tensor((0, 0))
                inputs[("resize", i)] = torch.Tensor((0, 0))

        poses = {}
        if type(self).__name__ == "CityscapesDataset":
            folder, frame_index, side = self.index_to_folder_and_frame_idx(index)
            inputs.update(self.get_colors(folder, frame_index, side, do_flip, 'data', augs = False, foggy=self.foggy))
            if self.is_train or self.robust_val:
                inputs.update(self.get_colors(folder, frame_index, side, do_flip, spec, augs = True))
            inputs["dataset"] = 1

        elif type(self).__name__ == "KITTIRAWDataset" or type(self).__name__ == "KITTIOdomDataset":
            inputs["dataset"] = 0
            if self.is_robust_test:
                folder, frame_index, side, spec = self.index_to_folder_and_frame_idx(index)
                if self.robust_augment != None:
                    spec = self.robust_augment
            else:
                folder, frame_index, side, _ = self.index_to_folder_and_frame_idx(index)
            for i in self.frame_idxs:
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("color", i, -1)] = self.get_color(
                        folder, frame_index, other_side, "data", do_flip)
                else:
                    try:
                        if self.is_robust_test:
                            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, spec, do_flip)
                        else:
                            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, "data", do_flip)
                        if self.is_train or self.robust_val:
                            inputs[("augmented", i, -1)] = self.get_color(folder, frame_index + i, side, spec, do_flip)
                    except FileNotFoundError as e:
                        if i != 0:
                            # fill with dummy values
                            inputs[("color", i, -1)] = Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))
                            if self.is_train or self.robust_val:
                                inputs[("augmented", i, -1)] = Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))
                            poses[i] = None
                        else:
                            raise FileNotFoundError(f'Cannot find frame - make sure your '
                                                    f'--data_path is set correctly, or try adding'
                                                    f' the --png flag. {e}')
            for scale in range(self.num_scales):
                K = self.load_intrinsics_kitti(folder, frame_index)
                if do_scale:
                    K[0, :] *= width_re // (2 ** scale)
                    K[1, :] *= height_re // (2 ** scale)
                    inv_K = np.linalg.pinv(K)
                    inputs[("K", scale)] = torch.from_numpy(K)
                    inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
                else:
                    K[0, :] *= self.width // (2 ** scale)
                    K[1, :] *= self.height // (2 ** scale)
                    inv_K = np.linalg.pinv(K)
                    inputs[("K", scale)] = torch.from_numpy(K)
                    inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        elif type(self).__name__ == "NuScenesDataset":
            inputs["dataset"] = 2
            new_index = self.get_correct_index(index)
            sample = self.get_sample_data(new_index)
            for i in self.frame_idxs:
                if i == "s":
                    raise NotImplementedError('nuscenes dataset does not support stereo depth')
                else:
                    inputs[("color", i, -1)] = self.get_color_nuscenes(sample, i, do_flip)

            for scale in range(self.num_scales):
                K = self.load_intrinsics_nuscenes(sample)
                if do_scale:
                    K[0, :] *= width_re // (2 ** scale)
                    K[1, :] *= height_re // (2 ** scale)
                    inv_K = np.linalg.pinv(K)
                    inputs[("K", scale)] = torch.from_numpy(K)
                    inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
                else:
                    K[0, :] *= self.width // (2 ** scale)
                    K[1, :] *= self.height // (2 ** scale)
                    inv_K = np.linalg.pinv(K)
                    inputs[("K", scale)] = torch.from_numpy(K)
                    inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
            
        elif type(self).__name__ == "DRIVINGSTEREO":

            inputs[("color", 0, -1)] = self.get_color(self.filenames[index], self.stereo_split)

        elif type(self).__name__ == "NUSCENESEVAL":

            new_index = self.get_correct_index(index)

            inputs[("color", 0, -1)] = self.get_color(new_index)
        
        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        if rand_erase:
            erase_aug = transforms.RandomErasing(p=1, scale=(0.015, 0.1), ratio=(0.15, 1.5), value=0, inplace=False)
        else:
            erase_aug = (lambda x: x)

        if do_vertical:
            rand_w = random.randint(1, 4) / 5
        else:
            rand_w = 0
        
        if do_tiling:
            height_range = [2, 3]
            width_range = [2, 4]
            # factors of 32 except 3 
            height_selection = random.choice(height_range)
            width_selection = random.choice(width_range)
            selection = (height_selection, width_selection)  
            both = np.prod(selection) 
            order = random.sample(range(both), both)
            if len(order) < 12:
                order.extend([0] * (12 - len(order)))
            else:
                order = order
        else:
            selection = (0, 0)
            order = [0] * 12

    
        self.preprocess(inputs, color_aug, erase_aug, do_vertical, do_scale, small, height_re, width_re, box_HiS, do_flip, order, do_tiling, selection, rand_w, spec)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color", i, 1)]
            del inputs[("color", i, 2)]
            del inputs[("color", i, 3)]
            if self.is_train or self.robust_val:
                del inputs[("augmented", i, -1)]
                del inputs[("augmented", i, 0)]
                del inputs[("augmented", i, 1)]
                del inputs[("augmented", i, 2)]
                del inputs[("augmented", i, 3)]
        
        inputs["index"] = index
        inputs["rand_w"] = rand_w
        inputs["order"] = torch.tensor(order)
        inputs["do_tiling"] = do_tiling
        inputs["tile_selection"] = torch.tensor(selection)

        new_dict = {}
        if self.is_train:
            for key in dict_color_augs.keys():
                if key == "color" and do_color_aug:
                    new_dict[key] = True
                elif spec == key:
                    new_dict[key] = True
                elif key == "tiling" and do_tiling:
                    new_dict[key] = True
                elif key == "vertical" and do_tiling:
                    new_dict[key] = True
                elif key == "erase" and do_tiling:
                    new_dict[key] = True
                elif key == "scale" and do_tiling:
                    new_dict[key] = True
                else:
                    new_dict[key] = False
        inputs["distribution"] = new_dict

        inputs["do_scale"] = do_scale
        inputs["small"] = small

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    
    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
