# Self-supervised Monocular Depth Estimation: Let's Talk About The Weather


 >**Self-supervised Monocular Depth Estimation: Let's Talk About The Weather
 >
 >[[Arxiv](https://arxiv.org/pdf/2307.08357.pdf)] [[Project Page](https://kieran514.github.io/Robust-Depth-Project/)]



https://github.com/kieran514/Robust-Depth/assets/51883968/1946b0c7-2166-4c7a-95e1-0038eb0acee8

If you find our work useful in your research, kindly consider citing our paper:

```
@misc{saunders2023selfsupervised,
      title={Self-supervised Monocular Depth Estimation: Let's Talk About The Weather}, 
      author={Kieran Saunders and George Vogiatzis and Luis Manso},
      year={2023},
      eprint={2307.08357},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Installation Setup

The models were trained using CUDA 11.1, Python 3.7.4 (conda environment), and PyTorch 1.8.0.

Create a conda environment with the PyTorch library:

```bash
conda env create --file environment.yml
conda activate robustdepth
```

## Datasets

### Training
We use the [KITTI dataset](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) and follow the downloading/preprocessing instructions set out by [Monodepth2](https://github.com/nianticlabs/monodepth2).
Download from scripts;
```
wget -i scripts/kitti_archives_to_download.txt -P data/KITTI_RAW/
```
Then unzip the downloaded files;
```
cd data/KITTI_RAW
unzip "*.zip"
cd ..
cd ..
```
Then convert all images to jpg;
```
find data/KITTI_RAW/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

## Creating Augmentations For Any Dataset
Here, we have the flexibility to create any augmentations we desire before commencing the training process. Once we have generated the augmented data using the steps outlined below, we can proceed to train using only those augmented images.

### Motion Blur & Snow
We first download the repository from [AutoMold](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library) into the main branch. First, rename the Automold--Road-Augmentation-Library-master to Automold. Then execute the snow_motion.py script provided in the scripts folder to create motion blur and snow augmentations. 

Firstly, we need to download the repository from [AutoMold](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library) into the main branch. After downloading, rename the folder "Automold--Road-Augmentation-Library-master" to simply "Automold." Next, proceed to execute the "snow_motion.py" script located in the scripts folder. This will generate motion blur and snow augmentations.
```
python scripts/snow_motion.py 
```
Please direct over to the [AutoMold](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library) GitHub page for more information.

### Corruptions & RGB-Grey
We execute the corruption.py script provided in the scripts folder to create image degradation augmentations, including red, green, blue and grey images. The code used is a modified version from [robustness](https://github.com/hendrycks/robustness).
```
python scripts/corruption.py 
```
Please direct over to the [robustness](https://github.com/hendrycks/robustness) GitHub page for more information.

### Rain
First, we create a rainy version of the KITTI dataset using a GAN. We download CycleGAN from the repository [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Here we trained a CycleGAN model to convert clear to rainy using the NuScenes dataset. We have provided the pretrained model for ease of use [RainGAN](https://drive.google.com/drive/folders/1Yb67rvfTyBfwpcoRx98Ubw_KlGPLV3jc?usp=drive_link) which needs to be placed inside pytorch-CycleGAN-and-pix2pix-master/checkpoints/rain_cyclegan/. Before we continue, please locate pytorch-CycleGAN-and-pix2pix-master/util/visualizer.py and add the following if statement on line 41 (indent until line 50). 
```
if label != 'real':
```
And replace line 43 with the below code:
```
image_name = '%s.png' % (name)
```
Using this model and the script provided, we create a rainy version of the KITTI dataset.
```
bash scripts/run_rain_sim.sh 
```
Next, we must create a depth version of the KITTI dataset using pretrained weights from [Monodepth2](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip). These pretrained weights are placed into the folder pretrained. Then we simply run this script.
```
python scripts/depth_simple.py
```
Now, we copy the repository from [rain-rendering](https://github.com/astra-vision/rain-rendering), Following the provided steps on their GitHub page, create the required environment: 
```
conda create --name py36_weatheraugment python=3.6 opencv numpy matplotlib tqdm imageio pillow natsort glob2 scipy scikit-learn scikit-image pexpect -y

conda activate py36_weatheraugment

pip install pyclipper imutils
```
Download the Columbia Uni. [rain streak database](https://www.cs.columbia.edu/CAVE/databases/rain_streak_db/databases.zip) and extract files in 3rdparty/rainstreakdb

From here we employ that you place our KITTI_RAW.py (inside scripts folder) file into rain-rendering-master/config/. Now add the following line of code to line 361 in rain_rendering-master/common/generator.py
```
depth = cv2.resize(depth, (bg.shape[1], bg.shape[0]))
```
Also, replace line 209 with the below code:
```
out_dir = os.path.join(out_seq_dir, 'image_02/', weather)
```
Additionally, replace line 324 with; 
```
out_rainy_path = os.path.join(out_dir, '{}.png'.format(file_name[:-4]))
```
and comment out lines 457 and 468.

Finally, you will need particles as provided by [rain streak database](https://www.cs.columbia.edu/CAVE/databases/rain_streak_db/databases.zip). For ease of use, I have provided the particle files which should be extracted in /rain-rendering-master/data/particles/ [found here](https://drive.google.com/file/d/1-nmBojZDz_-FXkUbreIyKOQlxuBeVLBp/view?usp=drive_link).

From here you can run this script. (max_thread on line 176 is set to 10, change this if you wish)
```
bash scripts/run_kitti_rain.sh
```
Please direct over to the [rain-rendering](https://github.com/astra-vision/rain-rendering) GitHub page for more information.

### Night, Dawn & Dusk
We first copy the repository from [CoMoGAN](https://github.com/astra-vision/CoMoGAN), we then create a file inside CoMoGAN-main called logs and place pretrained weights provided by CoMoGAN inside (CoMoGAN-main/logs/pretrained/).
Now we must add the following code to line 13 in CoMoGAN-main/data/base_dataset.py.
```
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```
Next, run the comogan.sh script to create Night, Dawn and Dusk augmentations on the clear and rain images.
```
conda activate robustdepth
bash scripts/comogan.sh 
```
Please direct over to the [CoMoGAN](https://github.com/astra-vision/CoMoGAN) GitHub page for more information.

### Fog 
For fog generation, we have used a script strongly inspired by [rain-rendering](https://github.com/astra-vision/rain-rendering), which will create a foggy augmentation for rain, night, clear, dawn, dusk, dawn+rain, night+rain, dusk+rain images. As this script was personally provided by the authors of [rain-rendering](https://github.com/astra-vision/rain-rendering) we choose not to share it as we have not received permission.
<!---```
bash scripts/fogOffical.sh 
```--->
#### File Format
```
├── KITTI_RAW
    ├── 2011_09_26
    ├── 2011_09_28
    │   ├── 2011_09_28_drive_0001_sync
    │   ├── 2011_09_28_drive_0002_sync
    |   │   ├── image_00
    |   │   ├── image_01
    |   │   ├── image_02
    |   │   |   ├── B
    |   │   |   ├── blur
    |   │   |   ├── data
    |   │   |   ├── dawn
    |   │   |   ├── dawn+fog
    |   │   |   ├── dawn+rain
    |   │   |   ├── dawn+rain+fog
    |   │   |   ├── defocus_blur
    |   │   |   ├── depth
    |   │   |   ├── dusk
    |   │   |   ├── dusk+fog
    |   │   |   ├── dusk+rain
    |   │   |   ├── dusk+rain+fog
    |   │   |   ├── elastic_transform
    |   │   |   ├── fog
    |   │   |   ├── fog+night
    |   │   |   ├── frost
    |   │   |   ├── G
    |   │   |   ├── gaussian_noise
    |   │   |   ├── glass_blur
    |   │   |   ├── greyscale
    |   │   |   ├── ground_snow
    |   │   |   ├── impulse_noise
    |   │   |   ├── jpeg_compression
    |   │   |   ├── night
    |   │   |   ├── pixelate
    |   │   |   ├── R
    |   │   |   ├── rain
    |   │   |   ├── rain+fog
    |   │   |   ├── rain+fog+night
    |   │   |   ├── rain_gan
    |   │   |   ├── rain+night
    |   │   |   ├── shot_noise
    |   │   |   ├── snow
    |   │   |   ├── zoom_blur
    |   │   |   ├── timestamps.txt
    |   |   ├── image_03
    |   │   ├── oxts
    |   │   ├── velodyne_points
    │   ├── calib_cam_to_cam.txt
    │   ├── calib_imu_to_velo.txt
    │   ├── calib_velo_to_cam.txt
    ├── 2011_09_29
    ├── 2011_09_30
    └── 2011_10_03
```
## Pretrained Models

| Model Name          | *Sunny* Abs_Rel | *Bad Weather* Abs_Rel | Model resolution  | Model  |
|-------------------------|-------------------|--------------------------|-----------------|------|
| [`ViT`](https://drive.google.com/drive/folders/1oKT2oAPp-7altFTvPKR2d7FdgXN9xMG3?usp=sharing)          | 0.100 | 0.114 | 640 x 192                | ViT        |
| [`Resnet18`](https://drive.google.com/drive/folders/1QSHZjOk6Ufw52BGjJmuxV7PJQNisH5Kk?usp=sharing)        | 0.115 | 0.133 | 640 x 192                |  Resnet18          |

##KITTI Ground Truth 

We must prepare ground truth files for validation and training.
```
python Robust-Depth/export_gt_depth.py --data_path data/KITTI_RAW --split eigen
python Robust-Depth/export_gt_depth.py --data_path KITTI_RAW --split eigen_zhou
# The following can be ignored for now
python Robust-Depth/export_gt_depth.py --data_path KITTI_RAW --split eigen_benchmark
```

## Training

The models can be trained on the KITTI dataset by running: 
```
bash Robust-Depth/experiments/train_all.sh
```
The hyperparameters are defined in the script file and set at their defaults as stated in the paper.

To train with the vision transformer please add --ViT to train_all.sh and see MonoViT's repository for any issues.

Feel free to vary which augmentations are used.

### Adding your own augmentations

Finally, as Robust-Depth can have many further applications, we provide a simple step-by-step solution to train with one's own augmentations. Here we will add a near-infrared augmentation as an example. 

1. First create the augmentation on the entire KITTI dataset in the same format as above (in this case called NIR)
2. Enter Robust-Depth/options.py and add self.parser.add_argument("--do_NIR", help="NIR augmentation", action="store_true")
3. Inside Robust-Depth/trainer.py, add do_NIR_aug = self.opt.NIR to line 155 and line 181. Then add NIR:{self.opt.NIR} to line 233
4. Inside Robust-Depth/datasets/mono_dataset.py, add do_NIR_aug=False to line 70 and self.do_NIR_aug = do_NIR_aug to line 110
5. Inside Robust-Depth/datasets/mono_dataset.py, add 'NIR':self.do_NIR_aug to line 303 (where 'NIR' is the augmented images folder name)
6. Now inside the Robust-Depth/experiments/train_all.sh split add --do_NIR (removing other augmentations if you wish) and proceed with training

## Evaluation
We provide the evaluation for the KITTI dataset. To create the necessary ground truth for the KITTI dataset 

### Testing
Evaluation for Cityscape Foggy, DrivingStereo and NuScenes-Night coming soon. 

### KITTI 

```
python Robust-Depth/evaluate_depth.py --eval_mono --load_weights_folder {weights_directory}
```

### KITTI Robust

```
python Robust-Depth/evaluate_depth.py --eval_mono --load_weights_folder {weights_directory} --robust_test
```

### KITTI Benchmark Robust

```
python Robust-Depth/evaluate_depth.py --eval_mono --load_weights_folder {weights_directory} --robust_test --eval_split eigen_benchmark
```

### KITTI Robust specific

```
python Robust-Depth/evaluate_depth.py --eval_mono --load_weights_folder {weights_directory} --robust_test --robust_augment blur
```




## References

* [Monodepth2](https://github.com/nianticlabs/monodepth2) (ICCV 2019)
* [MonoViT](https://github.com/zxcqlf/MonoViT) 



