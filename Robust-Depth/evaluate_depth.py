# python vddepth/evaluate_depth.py --eval_mono --load_weights_folder /media/kieran/ODD/logs/train_all/models/weights_24
# python vddepth/evaluate_depth.py --eval_mono --load_weights_folder /media/kieran/SSDNEW/logs/train_1/models/weights_absrel91 --robust_test 
# python vddepth/evaluate_depth.py --eval_mono --load_weights_folder /media/kieran/ODD/logs/train_all_fm_gone/models/weights_absrel89 --robust_test --robust_augment blur
# python vddepth/evaluate_depth.py --eval_mono --load_weights_folder /media/kieran/ODD/logs/train_all_fm_gone/models/weights_absrel89 --robust_test --eval_split eigen_benchmark
# python vddepth/evaluate_depth.py --eval_mono --load_weights_folder /media/kieran/ODD/logs/train_all/models/weights_24 --eval_split cityscape --data_path /media/kieran/SSDNEW/Base-Model/data/CS_RAW --foggy
# python vddepth/evaluate_depth.py --eval_mono --load_weights_folder /media/kieran/ODD/logs/train_all_fm_gone/models/weights_best --data_path /media/kieran/SSDNEW/Base-Model/data/DrivingStereo --eval_split foggy
# python vddepth/evaluate_depth.py --eval_mono --load_weights_folder /media/kieran/ODD/logs/train_all/models/weights_24 --data_path /media/kieran/SSDNEW/Base-Model/data/nuScenes_RAW --eval_split nuScenes_test_night

from __future__ import absolute_import, division, print_function
from torchvision.utils import save_image

import os
import cv2
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks, networksvit
import pdb

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join("splits")

STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))
        if opt.eval_split == "nuScenes_test_night":
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "night_lidar.txt"))
        elif opt.eval_split == "eigen_zhou":
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "val_files.txt"))
        else:
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path, map_location='cuda:0')

        if opt.eval_split == 'cityscape':
            print("Using CITYSCAPE")
            dataset = datasets.CityscapesDataset(opt.data_path, filenames,
                                                     encoder_dict['height'], encoder_dict['width'],
                                                     [0], 4, is_train=False, foggy=opt.foggy)
        elif opt.eval_split == "sunny" or opt.eval_split == "foggy" or opt.eval_split == "rainy" or opt.eval_split == "cloudy":
            dataset = datasets.DRIVINGSTEREO(opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'],
                                                     [0], 4, is_train=False, stereo_split=opt.eval_split)
        elif opt.eval_split == "nuScenes_test_night":
            from datasets.nuscenes import NuScenes
            nusc = NuScenes(version='v1.0-trainval', dataroot='/media/kieran/SSDNEW/Base-Model/data/nuScenes_RAW', verbose=False)
            dataset = datasets.NUSCENESEVAL(opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'],
                                                     [0], 4, is_train=False, nusc=nusc)
        else:
            dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                               encoder_dict['height'], encoder_dict['width'],
                                               [0], 4, is_train=False, is_robust_test=opt.robust_test, 
                                           robust_augment=opt.robust_augment)

        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=6,
                            pin_memory=True, drop_last=False)

        # encoder = networks.ResnetEncoder(opt.num_layers, False)
        # depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        # model_dict = encoder.state_dict()
        # encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        # depth_decoder.load_state_dict(torch.load(decoder_path, map_location='cuda:0'))

        if not opt.ViT:

            encoder = networks.ResnetEncoder(18, False)
            loaded_dict_enc = torch.load(encoder_path, map_location="cuda:0")

            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
            encoder.load_state_dict(filtered_dict_enc)

            print("   Loading pretrained decoder")
            depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

            loaded_dict = torch.load(decoder_path, map_location="cuda:0")
            depth_decoder.load_state_dict(loaded_dict)

        else:

            encoder_dict = torch.load(encoder_path, map_location='cuda:0')

            encoder = networksvit.mpvit_small() 
            encoder.num_ch_enc = [64,128,216,288,288]
            depth_decoder = networksvit.DepthDecoder()

            model_dict = encoder.state_dict()
            encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            depth_decoder.load_state_dict(torch.load(decoder_path, map_location='cuda:0'))

        encoder.to("cuda:0")
        encoder.eval()
        depth_decoder.to("cuda:0")
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            n=0
            for data in tqdm.tqdm(dataloader):
                input_color = data[("color", 0, 0)].to("cuda:0")
                # pdb.set_trace()
                input_color.shape

                save_image(input_color, f'/media/kieran/SSDNEW/results/{n}.jpg')
                n+=1

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
        print(len(pred_disps))
        

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    if opt.eval_split == 'cityscape':
        print('loading cityscapes gt depths individually due to their combined size!')
        gt_depths = os.path.join(splits_dir, opt.eval_split, "gt_depths")
    else:
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]


    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in tqdm.tqdm(range(pred_disps.shape[0])):

        if opt.eval_split == 'cityscape':
            gt_depth = np.load(os.path.join(gt_depths, str(i).zfill(3) + '_depth.npy'))
            gt_height, gt_width = gt_depth.shape[:2]
            # crop ground truth to remove ego car -> this has happened in the dataloader for input
            # images
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]

        else:
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == 'cityscape':
            # when evaluating cityscapes, we centre crop to the middle 50% of the image.
            # Bottom 25% has already been removed - so crop the sides and the top here
            gt_depth = gt_depth[256:, 192:1856]
            pred_depth = pred_depth[256:, 192:1856]

        if opt.eval_split == "eigen" or opt.eval_split == "eigen_zhou":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        elif opt.eval_split == 'cityscape':
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
