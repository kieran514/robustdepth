import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="vddepth options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default='data/KITTI_RAW')
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default='log')

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="Robust-Depth")
        self.parser.add_argument("--use_augpose_loss",
                                 help="use pose loss for augmented images",
                                 action="store_true")
        self.parser.add_argument("--use_augpose_warping",
                                 help="Warping with augmented pose",
                                 action="store_true")
        
        self.parser.add_argument("--do_color",
                                 help="ColorJitter",
                                 action="store_true")
        self.parser.add_argument("--do_dawn",
                                 help="Dawn augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_dusk",
                                 help="Dusk augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_ground_snow",
                                 help="Ground snow augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_greyscale",
                                 help="Greyscale augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_gauss",
                                 help="Guassian Noise augmentation",
                                 action="store_true")

        self.parser.add_argument("--do_shot",
                                 help="Shot Noise augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_impulse",
                                 help="Impulse Noise augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_defocus",
                                 help="Defocus blur augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_glass",
                                 help="Glass blur augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_zoom",
                                 help="Zoom augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_snow",
                                 help="Snow augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_frost",
                                 help="Frost augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_elastic",
                                 help="Elastic transform augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_pixelate",
                                 help="Pixelate augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_jpeg_comp",
                                 help="Jpeg compression augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_rain",
                                 help="Rain augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_fog",
                                 help="Fog augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_night",
                                 help="Night augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_blur",
                                 help="Motion Blur augmentation",
                                 action="store_true")
        self.parser.add_argument("--R",
                                 help="Red augmentation",
                                 action="store_true")
        self.parser.add_argument("--G",
                                 help="Green augmentation",
                                 action="store_true")
        self.parser.add_argument("--B",
                                 help="Blue augmentation",
                                 action="store_true")

        self.parser.add_argument("--do_vertical",
                                 help="to use vertical augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_tiling",
                                 help="to use tile augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_erase",
                                 help="to use erase augmentation",
                                 action="store_true")
        self.parser.add_argument("--do_flip",
                                 help="Horizontal Flip",
                                 action="store_true")
        self.parser.add_argument("--do_scale",
                                 help="if set use modified RADepth scales",
                                 action="store_true")

        self.parser.add_argument("--one_optim",
                                 help="if set, we use naive optimisation",
                                 action="store_true")
        self.parser.add_argument("--warp_clear",
                                 help="if set, we use semi-augmented warping",
                                 action="store_true")
        self.parser.add_argument("--depth_loss",
                                 help="if set, we use pseudo supervised depth loss",
                                 action="store_true")
        self.parser.add_argument("--weighter",
                                 type=float,
                                 help="weight for proxy supervision",
                                 default=0.001)
        self.parser.add_argument("--teacher",
                                 help="if set, we use augmeted images during learning",
                                 action="store_true")
        self.parser.add_argument("--ViT",
                                 help="if set, use monovit depth network",
                                 action="store_true")

        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--cuda",
                                 type=int,
                                 help="number of cuda",
                                 default=0,
                                 choices=[0,1,2,3,4,5,6,7,8,9,10])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=512)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0.002)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=80.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=1)
        self.parser.add_argument("--scheduler_step_size",
                                 nargs="+",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=[15, 25, 29])


        self.parser.add_argument("--pytorch_random_seed",
                                 default=42,
                                 type=int)

        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 type=str,
                                 default="false",
                                 choices=["true", "false"])
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=4)
        self.parser.add_argument("--val_num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=4)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load",
                                 default='logs/')
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)
        self.parser.add_argument("--save_intermediate_models",
                                 help="if set, save the model each time we log to tensorboard",
                                 action='store_true')


        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--robust_test",
                                 help="if set evaluates using robust images",
                                 action="store_true")

        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen_zhou",
                                 choices=["eigen", "eigen_benchmark", "eigen_zhou", "benchmark", "odom_9",
                                          "odom_10", "cityscapes", "nuScenes_val"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        # Visualize while evaluation
        self.parser.add_argument("--visualize",
                                 help="visualize the evaluation results",
                                 action="store_true")
        self.parser.add_argument("--error_range",
                                 type=float,
                                 help="the range of the error to visualize, from 0 to error_range, in meters",
                                 default=200)
        self.parser.add_argument("--vis_name",
                                 help="saved error figure name",
                                 type=str,
                                 default='diff')             
        self.parser.add_argument("--robust_augment",
                                 help="The augmented scene variant",
                                 type=str,
                                 choices=['blur', 'defocus_blur', 'elastic_transform', 'fog', 'fog+night', 
                                 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 
                                 'night', 'pixelate', 'rain', 'rain+fog', 'rain+fog+night', 'rain+night', 'shot_noise', 
                                 'snow', 'zoom_blur', 'color', 'dusk', 'dawn', 'ground_snow', 'dawn+rain', 'dusk+rain', 
                                 'dusk+fog', 'dawn+fog', 'dawn+rain+fog', 'dusk+rain+fog', 'greyscale', 'R', 'G', 'B'])
        self.parser.add_argument("--foggy",
                                 help="use foggy for cityscape ",
                                 action="store_true")

        # Export warped img
        self.parser.add_argument('--export',
                                 action='store_true',
                                 help='If set, will export warped img based on the multi frame depth and pose prediction')
        
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
