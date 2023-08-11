from codecs import backslashreplace_errors
from mimetypes import init
import os
from re import S
from urllib.request import BaseHandler
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import pdb
import numpy as np
import time
import random
from collections import Counter
import progressbar
from tqdm import tqdm
import wandb
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from rigid_warp import forward_warp
import scipy
import torch.nn as nn

import json

from utils import readlines, sec_to_hm_str
from layers import SSIM, BackprojectDepth, Project3D, BackprojectDepthRA, Project3DRA, transformation_from_parameters, \
    disp_to_depth, get_smooth_loss, compute_depth_errors

import datasets, networks
import matplotlib.pyplot as plt


_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer:
    def __init__(self, options):
        self.opt = options
        wandb.init(project='ROBUSTDEPTH')
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.c = Counter()

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else f"cuda:{self.opt.cuda}")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        frames_to_load = self.opt.frame_ids.copy()

        print('Loading frames: {}'.format(frames_to_load))

        if self.opt.ViT:
            self.models["encoder"] = networks.mpvit_small() 
            self.models["encoder"].num_ch_enc = [64,128,216,288,288] 
            self.models["encoder"].to(self.device)

            self.models["depth"] = networks.DepthDecoder()
            self.models["depth"].to(self.device)
        else:
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)

            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)

            self.parameters_to_train += list(self.models["encoder"].parameters())

        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["pose_encoder"] = networks.ResnetEncoder(
                                        18, 
                                        self.opt.weights_init == "pretrained",
                                        num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)

        self.models["pose"] = networks.PoseDecoder(
                                    self.models["pose_encoder"].num_ch_enc,
                                    num_input_features=1,
                                    num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)

        self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.ViT:
            self.params = [ {
            "params":self.parameters_to_train, 
            "lr": 1e-4
            },
            {
            "params": list(self.models["encoder"].parameters()), 
           "lr": 5e-5
            } ]
            self.model_optimizer = optim.AdamW(self.params)
            self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.model_optimizer, 0.9)
        else:
            self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
            self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.model_optimizer, milestones=self.opt.scheduler_step_size, gamma=0.1)


        if self.opt.load_weights_folder != 'None':
            print("Loading weights")
            self.load_model()

        print("Training model named:  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to: ", self.opt.log_dir)
        print("Training is using:  ", self.device)

        datasets_dict = {"kitti": datasets.KITTIRAWDataset}

        img_ext = '.png' if self.opt.png else '.jpg'
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join("splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=True, img_ext=img_ext, vertical = self.opt.do_vertical, tiling = self.opt.do_tiling, 
            do_rain_aug = self.opt.do_rain, do_fog_aug = self.opt.do_fog, do_night_aug = self.opt.do_night, 
            do_scale_aug = self.opt.do_scale, do_blur_aug = self.opt.do_blur, do_erase_aug = self.opt.do_erase, do_color_aug = self.opt.do_color,
            do_gauss_aug = self.opt.do_gauss, do_shot_aug = self.opt.do_shot, do_impulse_aug = self.opt.do_impulse, do_defocus_aug = self.opt.do_defocus,
            do_glass_aug = self.opt.do_glass, do_zoom_aug = self.opt.do_zoom, do_snow_aug = self.opt.do_snow, do_frost_aug = self.opt.do_frost,
            do_elastic_aug = self.opt.do_elastic, do_pixelate_aug = self.opt.do_pixelate, do_jpeg_comp_aug = self.opt.do_jpeg_comp, do_flip_aug = self.opt.do_flip,
            do_dawn_aug = self.opt.do_dawn, do_dusk_aug = self.opt.do_dusk, do_ground_snow_aug = self.opt.do_ground_snow, do_greyscale_aug = self.opt.do_greyscale,
            do_Red_aug = self.opt.R, do_Green_aug = self.opt.G, do_Blue_aug = self.opt.B)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker)

        val_filenames = readlines(os.path.join('splits', self.opt.eval_split, "val_files.txt"))

        val_frames_load = frames_to_load

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            val_frames_load, 4, is_train=False, robust_val=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, 1, shuffle=False,
            num_workers=self.opt.val_num_workers, pin_memory=True, drop_last=False)

        robust_val_dataset = datasets_dict["kitti"](
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=False, robust_val=True, img_ext=img_ext, 
            do_rain_aug = self.opt.do_rain, do_fog_aug = self.opt.do_fog, do_night_aug = self.opt.do_night, 
            do_blur_aug = self.opt.do_blur, do_color_aug = self.opt.do_color,
            do_gauss_aug = self.opt.do_gauss, do_shot_aug = self.opt.do_shot, do_impulse_aug = self.opt.do_impulse, do_defocus_aug = self.opt.do_defocus,
            do_glass_aug = self.opt.do_glass, do_zoom_aug = self.opt.do_zoom, do_snow_aug = self.opt.do_snow, do_frost_aug = self.opt.do_frost,
            do_elastic_aug = self.opt.do_elastic, do_pixelate_aug = self.opt.do_pixelate, do_jpeg_comp_aug = self.opt.do_jpeg_comp, do_flip_aug = self.opt.do_flip,
            do_dawn_aug = self.opt.do_dawn, do_dusk_aug = self.opt.do_dusk, do_ground_snow_aug = self.opt.do_ground_snow, do_greyscale_aug = self.opt.do_greyscale,
            do_Red_aug = self.opt.R, do_Green_aug = self.opt.G, do_Blue_aug = self.opt.B)


        self.robust_val_loader = DataLoader(
            robust_val_dataset, 1, shuffle=False,
            num_workers=self.opt.val_num_workers, pin_memory=True, drop_last=False)

        self.train_filenames = train_filenames
        self.val_filenames = val_filenames

        gt_path = os.path.join('splits', self.opt.eval_split, "gt_depths.npz")
        self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        self.writers = {}
        for mode in ["train", "val", "robust_val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim == 'true':
            self.ssim = SSIM()
            self.ssim.to(self.device)

        # self.backproject_depth = {}
        self.backproject_depth_RA= {}
        # self.project_3d = {}
        self.project_3d_RA = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth_RA[scale] = BackprojectDepthRA(self.opt.batch_size, h, w, self.device)
            self.backproject_depth_RA[scale].to(self.device)

            self.project_3d_RA[scale] = Project3DRA(self.opt.batch_size, h, w)
            self.project_3d_RA[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        self.len_train_dataset = len(train_dataset)
        print("There are {:d} training items and {:d} validation items\n".format(
            self.len_train_dataset, len(val_dataset)))

        print(f"You have selected: \
        \nTeacher:{self.opt.teacher}, MotionBlur:{self.opt.do_blur} \
        \nGauss_N:{self.opt.do_gauss}, Shot_N:{self.opt.do_shot}, Impulse_N:{self.opt.do_impulse}, Defocus:{self.opt.do_defocus} \
        \nGlass:{self.opt.do_glass}, Zoom:{self.opt.do_zoom}, Snow:{self.opt.do_snow}, Frost:{self.opt.do_frost} \
        \nElastic:{self.opt.do_elastic}, Pixelate:{self.opt.do_pixelate}, JPEG:{self.opt.do_jpeg_comp}, Brightness:{self.opt.do_color} \
        \nRain:{self.opt.do_rain}, Fog:{self.opt.do_fog}, Night:{self.opt.do_night}, Flip:{self.opt.do_flip} \
        \nScale:{self.opt.do_scale}, Tiling:{self.opt.do_tiling}, Vertical:{self.opt.do_vertical}, Erase:{self.opt.do_erase} \
        \nGreyscale:{self.opt.do_greyscale}, GroundSnow:{self.opt.do_ground_snow}, Dusk:{self.opt.do_dusk}, Dawn:{self.opt.do_dawn} \
        \nRed:{self.opt.R}, Green:{self.opt.G}, Blue:{self.opt.B}")

        print(f"Are we using Augmented pose loss? {self.opt.use_augpose_loss}\nAre we using Augmented pose warping? {self.opt.use_augpose_warping}")

        print(f"Learning rate updates at epochs {self.opt.scheduler_step_size}")

        self.save_opts()
        wandb.config.update(self.opt)
        self.best = 10.0
        self.robust_best = 10.0

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        if self.opt.load_weights_folder != 'None':
            number = self.opt.load_weights_folder.split("_")[-1]
            print(number)
            if number == 'best':
                self.epoch = 15
                number = 15
            else:
                self.epoch = int(number) + 1 # start training from where you stopped
            start = int(number) + 1 # start training from where you stopped
            self.step = self.epoch * (self.len_train_dataset // self.opt.batch_size)
            for _ in range(start):
                self.model_lr_scheduler.step()
        else:
            self.epoch = 0
            self.step = 0
            start = 0

        if self.opt.ViT:
            depth_lr = self.model_optimizer.param_groups[1]['lr']
            pose_lr = self.model_optimizer.param_groups[0]['lr']
            print(f'\nStarting from epoch {self.epoch} and current learning rate for depth is {depth_lr} and pose lr is {pose_lr}')
        else:
            starting_lr = self.model_optimizer.param_groups[0]['lr']
            print(f'\nStarting from epoch {self.epoch} and current learning rate is {starting_lr}')
        self.start_time = time.time()
        for self.epoch in range(start, self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
        self.models = init
        wandb.join()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(tqdm(self.train_loader)):

            before_op_time = time.time()

            outputs, outputs_hint, losses = self.process_batch(inputs, is_train=True)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            early_phase = batch_idx % self.opt.log_frequency == 0

            if early_phase and self.step > 0:
                print(self.c)
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                current_learning_rate = self.model_optimizer.param_groups[0]['lr']
                print(f'Current learning rate: {current_learning_rate}')

                self.log("train", inputs, outputs, outputs_hint, losses)
                self.val()
                                
            self.step += 1
        self.model_lr_scheduler.step()

    def process_batch(self, inputs, is_train=True, robust_val=False):
        """Pass a minibatch through the network and generate images and losses
        """
        new_dict = {}
        for key, value in inputs["distribution"].items():
            new_dict[key] = sum(value).item()
        self.c.update(new_dict)

        del inputs["distribution"]
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs_hint = {}

        if self.opt.teacher and is_train:
            pose_pred = self.predict_poses(inputs, None, is_train or self.opt.export, do_aug=False)
            outputs_hint.update(pose_pred)

            feats = self.models["encoder"](inputs["color", 0, 0])
            outputs_hint.update(self.models['depth'](feats))

            # update depth map
            if any(inputs["do_scale"]):
                self.update_depth_scale(inputs, outputs_hint)

            self.generate_images_pred(inputs, outputs_hint, clear=True)

            outputs_hint["feats"] = feats

        if is_train:
            outputs = {}

            if self.opt.use_augpose_warping or self.opt.use_augpose_loss:
                outputs.update(self.predict_poses(inputs, None, is_train or self.opt.export, do_aug=True))

            feats = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs.update(self.models['depth'](feats))
            outputs["feats"] = feats

            if self.opt.do_vertical or self.opt.do_tiling:
                outputs.update(self.reverse_depth_vertical(inputs, self.reverse_tile_crop(inputs, outputs)))
            self.generate_images_pred(inputs, outputs, outputs_hint = outputs_hint, clear_warp = (not self.opt.use_augpose_warping), clear=False)
            losses = self.compute_losses(inputs, outputs, outputs_hint)
        else:
            outputs = {}
            if robust_val:
                feats = self.models["encoder"](inputs["color_aug", 0, 0])
            else:
                feats = self.models["encoder"](inputs["color", 0, 0])
            outputs.update(self.models['depth'](feats))
            _, outputs["depth", 0, 0] = disp_to_depth(outputs["disp", 0], self.opt.min_depth, self.opt.max_depth)
            losses = None

        return outputs, outputs_hint, losses 

    def predict_poses(self, inputs, features=None, is_train=True, do_aug=True):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if is_train:
            frameIDs = self.opt.frame_ids
        else:
            frameIDs = [0, -1]
        if self.num_pose_frames == 2:
            if do_aug:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in frameIDs}
            else:
                pose_feats = {f_i: inputs["color", f_i, 0] for f_i in frameIDs}
            for f_i in frameIDs[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    outputs[("pose_feats", 0, f_i)] = pose_inputs

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        else:
            raise NotImplementedError

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        losses = {}
        for metric in self.depth_metric_names:
            losses[metric] = 0.0
        total_batches = 0.0
        kitti_idx = 0
        city_idx = 0
        nu_idx = 0
        with torch.no_grad():
            for batch_idx, inputs in enumerate(tqdm(self.val_loader)):
                total_batches += 1.0
                outputs, outputs_hint, _ = self.process_batch(inputs, is_train=False)
                self.compute_depth_losses(inputs, outputs, losses, batch_idx, kitti_idx, city_idx, nu_idx, accumulate=True)

        print('Val result:')
        for metric in self.depth_metric_names:
            losses[metric] /= total_batches
            print(metric, ': ', losses[metric])
        self.log('val', inputs, outputs, outputs_hint, losses)
        if losses["de/abs_rel"] < self.best:
            self.best = losses["de/abs_rel"]
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Saving best result: ", self.best)
            self.save_model("best")
            absrel = round(losses["de/abs_rel"] * 1000)
            if absrel < 120:
                self.save_model('absrel{}'.format(absrel))
        del inputs, outputs, losses

        losses = {}
        for metric in self.depth_metric_names:
            losses[metric] = 0.0
        total_batches = 0.0
        kitti_idx = 0
        city_idx = 0
        nu_idx = 0
        with torch.no_grad():
            for batch_idx, inputs in enumerate(tqdm(self.robust_val_loader)):
                total_batches += 1.0
                outputs, outputs_hint, _ = self.process_batch(inputs, is_train=False, robust_val=True) #HERE
                self.compute_depth_losses(inputs, outputs, losses, batch_idx, kitti_idx, city_idx, nu_idx, accumulate=True)

        print('Robust Val result:')
        for metric in self.depth_metric_names:
            losses[metric] /= total_batches
            print(metric, ': ', losses[metric])
        self.log('val', inputs, outputs, outputs_hint, losses, robust=True) #HERE
        if losses["de/abs_rel"] < self.robust_best:
            self.robust_best = losses["de/abs_rel"]
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Saving best robust result: ", self.robust_best)
        self.set_train()

    def generate_images_pred(self, inputs, outputs, outputs_hint=None, clear_warp=False, clear=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if clear_warp and self.opt.teacher: # if we want to use clear pose for warping
                    T = outputs_hint[("cam_T_cam", 0, frame_id)]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth_RA[source_scale](
                    depth, inputs[("inv_K", source_scale)], inputs[("dxy", source_scale)])
                pix_coords = self.project_3d_RA[source_scale](
                    cam_points, inputs[("K", source_scale)], T, inputs[("dxy", source_scale)])
                outputs[("sample", frame_id, scale)] = pix_coords
                if (self.opt.teacher and self.opt.warp_clear) or clear:
                    outputs[("color", frame_id, scale)] = F.grid_sample(inputs[("scale_aug", frame_id, source_scale)], outputs[("sample", frame_id, scale)],padding_mode="border",align_corners = True) # clear image
                else:
                    outputs[("color", frame_id, scale)] = F.grid_sample(inputs[("aug", frame_id, source_scale)], outputs[("sample", frame_id, scale)],padding_mode="border",align_corners = True)

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim == 'true':
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs, outputs_hint):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            if self.opt.teacher and self.opt.warp_clear:
                color = inputs[("scale_aug", 0, scale)]
                target = inputs[("scale_aug", 0, source_scale)]
                target_aug = inputs[("scale_aug", 0, source_scale)]
                proxy_supervised = outputs_hint[("disp", scale)]
            elif self.opt.teacher:
                color = inputs[("scale_aug", 0, scale)]
                target_aug = inputs[("aug", 0, source_scale)]
                target = inputs[("scale_aug", 0, source_scale)]
                proxy_supervised = outputs_hint[("disp", scale)]
            else:
                color = inputs[("aug", 0, scale)]
                target = inputs[("aug", 0, source_scale)]
                

            reprojection_losses = [self.compute_reprojection_loss(outputs[("color", frame_id, scale)], target_aug) for frame_id in self.opt.frame_ids[1:]] # target here should be aug
            reprojection_hint_losses = [self.compute_reprojection_loss(outputs_hint[("color", frame_id, scale)], target) for frame_id in self.opt.frame_ids[1:] if self.opt.teacher]

            reprojection_losses = torch.cat(reprojection_losses, 1)
            if self.opt.teacher:
                reprojection_hint_loss = torch.cat(reprojection_hint_losses, 1)

            if not self.opt.disable_automasking: 
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    if self.opt.teacher:
                        pred = inputs[("scale_aug", frame_id, source_scale)]
                    else:
                        pred = inputs[("aug", frame_id, source_scale)]
                    ident_reproj_loss = self.compute_reprojection_loss(pred, target)
                    identity_reprojection_losses.append(ident_reproj_loss)

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_loss = identity_reprojection_losses
            else:
                identity_reprojection_loss = None

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(self.device) * 0.00001

            if self.opt.teacher:
                combined = torch.cat((identity_reprojection_loss, reprojection_loss, reprojection_hint_loss), dim=1)
                if self.opt.one_optim:
                    to_optimise, idxs = torch.min(combined, dim=1)
                else:
                    to_optimise_aug, _ = torch.min(torch.cat((identity_reprojection_loss, reprojection_loss), dim=1), dim=1)
                    to_optimise_clear, _ = torch.min(torch.cat((identity_reprojection_loss, reprojection_hint_loss), dim=1), dim=1)
                    idxs = torch.argmin(combined, dim=1)
                idxs = idxs.unsqueeze(1).detach()
                proxy_supervised_mask = (((idxs == 4) + (idxs == 5)).float()).detach()
                hint_supervision_mask = (((idxs == 2) + (idxs == 3)).float()).detach()
                if scale != 0:
                    proxy_supervised_mask = F.interpolate(proxy_supervised_mask, [int(self.opt.height // 2 ** scale), int(self.opt.width // 2 ** scale)], mode="nearest").detach()
                    hint_supervision_mask = F.interpolate(hint_supervision_mask, [int(self.opt.height // 2 ** scale), int(self.opt.width // 2 ** scale)], mode="nearest").detach()

                if self.opt.depth_loss:
                    proxy_supervised_loss_hint = (self.compute_proxy_supervised_loss(disp, proxy_supervised.detach().clone(), proxy_supervised_mask) / (2 ** scale)) * self.opt.weighter
                    proxy_supervised_loss_proxy = (self.compute_proxy_supervised_loss(disp.detach().clone(), proxy_supervised, hint_supervision_mask) / (2 ** scale)) * self.opt.weighter

                    losses["loss/proxy_supervised_loss_hint{}".format(scale)] = proxy_supervised_loss_hint
                    losses["loss/proxy_supervised_loss_proxy{}".format(scale)] = proxy_supervised_loss_proxy
                    loss += proxy_supervised_loss_hint
                    loss += proxy_supervised_loss_proxy
                    zero_depth1 = (1000000 * (proxy_supervised <= 0).sum())
                    zero_depth2 = (1000000 * (disp <= 0).sum())
                    loss += zero_depth1 + zero_depth2

                if scale == 0:
                    inputs["mask_hint"] = proxy_supervised_mask.detach()
                    inputs["mask_proxy"] = hint_supervision_mask.detach()
            else:
                to_optimise, idxs = torch.min(torch.cat((identity_reprojection_loss, reprojection_loss), dim=1), dim=1)
                        
            if self.opt.one_optim:
                loss += to_optimise.mean()
            else:
                loss += to_optimise_aug.mean()
                loss += to_optimise_clear.mean()

            smooth_loss = 0
            smooth_loss_proxy_supervised = 0
            for batchs in range(self.opt.batch_size):
                if not inputs["small"][batchs]:
                    mean_disp = (disp[batchs].unsqueeze(0)).mean(2, True).mean(3, True)
                    norm_disp = (disp[batchs].unsqueeze(0)) / (mean_disp + 1e-7)
                    smooth_loss += get_smooth_loss(norm_disp, color[batchs].unsqueeze(0))
                    if self.opt.teacher:
                        mean_proxy_supervised = (proxy_supervised[batchs].unsqueeze(0)).mean(2, True).mean(3, True)
                        norm_proxy_supervised = (proxy_supervised[batchs].unsqueeze(0)) / (mean_proxy_supervised + 1e-7)
                        smooth_loss_proxy_supervised += get_smooth_loss(norm_proxy_supervised, color[batchs].unsqueeze(0))

            losses["loss/disp_smooth{}".format(scale)] = (self.opt.disparity_smoothness * (smooth_loss / self.opt.batch_size) / (2 ** scale))
            loss += losses["loss/disp_smooth{}".format(scale)]
            losses["loss/hint_smooth{}".format(scale)] = (self.opt.disparity_smoothness * (smooth_loss_proxy_supervised / self.opt.batch_size) / (2 ** scale))
            loss += losses["loss/hint_smooth{}".format(scale)]

            total_loss += loss

            losses["loss/{}".format(scale)] = loss
        total_loss /= self.num_scales

        if self.opt.teacher and self.opt.use_augpose_loss:
            axisangle_loss = 0
            translation_loss = 0
            for frame_id in self.opt.frame_ids[1:]:
                axisangle_loss += (torch.abs(outputs_hint[("axisangle", 0, frame_id)].detach().clone() - outputs[("axisangle", 0, frame_id)]).mean() / 2) * self.opt.weighter
                translation_loss += (torch.abs(outputs_hint[("translation", 0, frame_id)].detach().clone() - outputs[("translation", 0, frame_id)]).mean() / 2) * self.opt.weighter
            losses["loss/axisangle"] = axisangle_loss
            losses["loss/translation"] = translation_loss
            total_loss += (losses["loss/axisangle"] + losses["loss/translation"])
        
        losses["loss"] = total_loss
        return losses


    def compute_proxy_supervised_loss(self, pred, target, loss_mask):        
        loss = torch.log(torch.abs(target - pred) + 1)
        loss = loss * loss_mask
        loss = loss.sum() / (loss_mask.sum() + 1e-7)

        return loss

    def gradient(self, D):
        D_dy = D[:, :, 1:] - D[:, :, :-1]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy

    def update_depth_scale(self, inputs, outputs):
        do_scale = inputs["do_scale"]
        small = inputs["small"]
        for k in list(outputs):
            if 'disp' in k:
                n, scale = k
                origional = outputs[(n, scale)] 
                individual = []
                for i in range(self.opt.batch_size):
                    if do_scale[i] == True:
                        resizer = inputs["resize", scale]
                        dxy = inputs["dxy", scale]
                        if small[i] == True:
                            width_re = int(resizer[i,0])
                            height_re = int(resizer[i,1])
                            _, height_, width_ = origional[i].size()
                            disp_un_MiS = origional[i].unsqueeze(0) # torch.Size([1, 1, 24, 80])
                            disp_un_MiS = F.interpolate(disp_un_MiS, [height_re, width_re], mode="bilinear", align_corners=False)# torch.Size([1, 1, 24, 80])
                            disp_un_MiS = disp_un_MiS.squeeze(0) # torch.Size([1, 24, 80])
                            point1 = int(2 * width_re - width_)
                            point2 = int(2 * height_re - height_)
                            Tensor_LoS = torch.zeros(1, height_, width_)
                            Tensor_LoS[:, 0:height_re, 0:width_re] = disp_un_MiS
                            Tensor_LoS[:, height_re:height_, 0:width_re] = disp_un_MiS[:, point2:height_re, 0:width_re]
                            Tensor_LoS[:, 0:height_re, width_re:width_] = disp_un_MiS[:, 0:height_re, point1:width_re]
                            Tensor_LoS[:, height_re:height_, width_re:width_] = disp_un_MiS[:, point2:height_re, point1:width_re]
                            individual.append(Tensor_LoS.unsqueeze(0).to(self.device))
                        else:
                            _, height_, width_ = origional[i].size()
                            x_start = round( width_ * int(dxy[i,0]) * 1.0 / int(resizer[i,0]) )
                            y_start = round( height_ * int(dxy[i,1]) * 1.0 / int(resizer[i,1]) )
                            width_union = round( width_ * width_ * 1.0 / int(resizer[i,0]) )
                            height_union = round( height_ * height_ * 1.0 / int(resizer[i,1]) )
                            disp_union_MiS = origional[i,:,y_start:y_start+height_union,x_start:x_start+width_union].unsqueeze(0)
                            disp_union_MiS = F.interpolate(disp_union_MiS, [height_, width_], mode="bilinear", align_corners=False) # torch.Size([1, 1, 24, 80])
                            individual.append(disp_union_MiS.to(self.device))
                    else:
                        individual.append(origional[i].unsqueeze(0))
                outputs[(n, scale)] = torch.cat(individual, dim=0)

    def reverse_depth_vertical(self, inputs, outputs):
        for k in list(outputs):
            if 'disp' in k:
                individual = []
                for batch in range(self.opt.batch_size):
                    rand_w = 1 - inputs["rand_w"][batch]
                    if rand_w < 1:
                        in_h = outputs[k].shape[2]
                        cropped_y = [0, int(rand_w * in_h), in_h]
                        cropped_disp = [outputs[k][batch][:, cropped_y[n]:cropped_y[n+1], :] for n in range(2)]
                        cropped_disp = cropped_disp[::-1]
                        individual.append(torch.cat(cropped_disp, dim=1).unsqueeze(0))
                    else:
                        individual.append(outputs[k][batch].unsqueeze(0))
                outputs[k] = torch.cat(individual, dim=0)
        return outputs

    def reverse_tile_crop(self, inputs, outputs):
        for k in list(outputs):
            if 'disp' in k:
                individual = []
                for batch in range(self.opt.batch_size):
                    if inputs["do_tiling"][batch]:
                        selection = inputs["tile_selection"][batch]
                        _, c, h, w = outputs[k].shape
                        height_slection = selection[0]
                        width_selection = selection[1]
                        prod_selecion = torch.prod(selection)
                        height_split = h // height_slection
                        width_split = w // width_selection
                        tiles = [outputs[k][batch][:, x:x+height_split, y:y+width_split] for x in range(0, h, height_split) for y in range(0, w, width_split)]
                        tiles = [x for _, x in sorted(zip(inputs["order"][batch][:prod_selecion], tiles), key=lambda x: x[0])]
                        width_cat = [torch.cat(tiles[i:width_selection + i], dim=2) for i in range(0, prod_selecion, width_selection)]
                        final = torch.cat(width_cat, dim=1).unsqueeze(0)
                        individual.append(final)
                    else:
                        individual.append(outputs[k][batch].unsqueeze(0))
                outputs[k] = torch.cat(individual, dim=0)
        return outputs

    def compute_depth_losses(self, inputs, outputs, losses, idx, kitti_idx, city_idx, nu_idx, accumulate=False):
        """Compute depth metrics, to allow monitoring during training

        Contrary to Monodepth2 we are only using a batch of 1 therefore we have a more
        accurate of validation 
        """
        min_depth = 1e-3
        max_depth = 80

        _, depth_pred = disp_to_depth(outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

        gt_depth = self.gt_depths[idx]
        gt_height, gt_width = gt_depth.shape[:2]

        depth_pred = torch.clamp(F.interpolate(depth_pred, [gt_height, gt_width], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach().squeeze()
        

        if self.opt.eval_split in ["eigen", "eigen_benchmark", "eigen_zhou"]:
            mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                            0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)                 

        gt_depth = torch.from_numpy(gt_depth).to(self.device)
        mask = torch.from_numpy(mask).to(self.device)

        depth_pred *= torch.median(gt_depth[mask]) / torch.median(depth_pred[mask])
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(gt_depth[mask], depth_pred[mask])

        for i, metric in enumerate(self.depth_metric_names):
            if accumulate:
                losses[metric] += np.array(depth_errors[i].cpu())
            else:
                losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, outputs_hint, losses, robust=False):
        """Write an event to the tensorboard events file
        """
        if mode == 'val':
            frameIDs = [0, -1]
            BS = 1
        else:
            frameIDs = self.opt.frame_ids
            BS = self.opt.batch_size
        
        if robust:
            mode = 'robust_val'


        writer = self.writers[mode]
        for l, v in losses.items():
            if l != "reprojection_losses":
                wandb.log({"{}_{}".format(mode, l): v}, step=self.step)
                writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, BS)):  # write a maximum of four images
            s = 0  # log only max scale
            for frame_id in frameIDs:
                if mode == 'robust_val':
                    writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    logimg = wandb.Image(inputs[("color_aug", frame_id, s)][j].data.permute(1,2,0).cpu().numpy())
                    wandb.log({"{}/color_aug_{}_{}/{}".format(mode, frame_id, s, j): logimg}, step=self.step)
                else:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    logimg = wandb.Image(inputs[("color", frame_id, s)][j].data.permute(1,2,0).cpu().numpy())
                    wandb.log({"{}/color_{}_{}/{}".format(mode, frame_id, s, j): logimg}, step=self.step)

                if mode == "train":
                    writer.add_image(
                        "color_aug_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color_aug", frame_id, s)][j].data, self.step)
                    logimg = wandb.Image(inputs[("color_aug", frame_id, s)][j].data.permute(1,2,0).cpu().numpy())
                    wandb.log({"{}/color_aug_{}_{}/{}".format(mode, frame_id, s, j): logimg}, step=self.step)
                if mode == "train" and self.opt.teacher:# and (self.epoch > 0):
                    logmask = wandb.Image((inputs["mask_hint"])[j].data.permute(1,2,0).cpu().numpy())
                    wandb.log({"{}/mask_hint_{}_{}/{}".format(mode, frame_id, s, j): logmask}, step=self.step)
                    logdmaskdisp = wandb.Image(colormap((inputs["mask_hint"] * outputs_hint[("disp", s)])[j, 0]).transpose(1,2,0))
                    wandb.log({"{}/mask_hint_depth_{}_{}/{}".format(mode, frame_id, s, j): logdmaskdisp}, step=self.step)
                    
                    logmaskproxy = wandb.Image((inputs["mask_proxy"])[j].data.permute(1,2,0).cpu().numpy())
                    wandb.log({"{}/mask_proxy_{}_{}/{}".format(mode, frame_id, s, j): logmaskproxy}, step=self.step)
                    logdmaskproxy = wandb.Image(colormap((inputs["mask_proxy"] * outputs[("disp", s)])[j, 0]).transpose(1,2,0))
                    wandb.log({"{}/mask_proxy_depth_{}_{}/{}".format(mode, frame_id, s, j): logdmaskproxy}, step=self.step)

                    logdisphint = wandb.Image(colormap(outputs_hint[("disp", s)][j, 0]).transpose(1,2,0))
                    wandb.log({"{}/hintdepth_{}_{}/{}".format(mode, frame_id, s, j): logdisphint}, step=self.step)

                if s == 0 and frame_id != 0 and mode == 'train':
                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data, self.step)
                    logimg = wandb.Image(outputs[("color", frame_id, s)][j].data.permute(1,2,0).cpu().numpy())
                    wandb.log({"{}/color_pred_{}_{}/{}".format(mode, frame_id, s, j): logimg}, step=self.step)

            disp = colormap(outputs[("disp", s)][j, 0])
            writer.add_image(
                "disp_multi_{}/{}".format(s, j),
                disp, self.step)
            logimg = wandb.Image(disp.transpose(1,2,0))
            wandb.log({"{}/disp_multi_{}/{}".format(mode, s, j): logimg}, step=self.step)
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, name=None, save_step=False):
        """Save model weights to disk
        """
        if name is not None:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(name))
        else:
            if save_step:
                save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch,
                                                                                        self.step))
            else:
                save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Can't load Adam - using random")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis
