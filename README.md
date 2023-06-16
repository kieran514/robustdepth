# Self-supervised Monocular Depth Estimation: Let's Talk About The Weather


 >**Self-supervised Monocular Depth Estimation: Let's Talk About The Weather
 >
 >[[PDF](LINK)] [[Video](LINK)]



https://github.com/kieran514/Robust-Depth/assets/51883968/1946b0c7-2166-4c7a-95e1-0038eb0acee8



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
wget -i scripts/kitti_archives_to_download.txt -P kitti_data/
```
then unzip,
```
cd kitti_data
unzip "*.zip"
cd ..
```
Then convert to jpg
```
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

### Testing
Download the [Cityscape foggy dataset](https://www.cityscapes-dataset.com/downloads/), the [DrivingStereo weather dataset](https://drivingstereo-dataset.github.io/), and the entire [Nuscenens dataset](https://www.nuscenes.org/nuscenes#download).

## Creating Augmentations For Any Dataset
Here we can create any augmentations we like before we start training. After creating augmented data following the steps below, you can train with just those augmented images. 

### Motion Blur & Snow
We first copy the repo from [AutoMold](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library) into the main branch, we then execute the snow_motion.py script provided in the scripts folder to create motion blur and snow augmentations.
```bash
python snow_motion.py --data_path {data_directory}
```
Please direct over to the [AutoMold](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library) GitHub page for more information.

### Corruptions & RGBG
We first copy the repo from [robustness](https://github.com/hendrycks/robustness) into the robustness folder, we then execute the script provided in the robustness folder to create corruptions, red, blue, green and grey augmentations. (Make sure to change the datapath)
```bash
bash robustness/robustness.sh 
```
Please direct over to the [robustness](https://github.com/hendrycks/robustness) GitHub page for more information.

### Fog & Rain
We first copy the repo from [rain-rendering](https://github.com/astra-vision/rain-rendering) into the rain-rendering folder, we then execute the script provided in the rain-rendering folder to create NIGHT, DAWN and DUSK image augmentations. 

For fog generation, we were given this script personally and will only share it when I have permission. 

(Make sure to change the datapath)
```bash
bash rain-rendering/rain-rendering.sh 
```
Please direct over to the [rain-rendering](https://github.com/astra-vision/rain-rendering) GitHub page for more information.

### Night, Dawn & Dusk
We first copy the repo from [CoMoGAN](https://github.com/astra-vision/CoMoGAN) into the CoMoGAN folder, we then execute the script provided in the CoMoGAN folder to create NIGHT, DAWN and DUSK image augmentations. (Make sure to change the datapath)
```bash
bash CoMoGAN/comogan.sh 
```
Please direct over to the [CoMoGAN](https://github.com/astra-vision/CoMoGAN) GitHub page for more information.

#### File Format
The final file format should look like this;


## Pretrained Models

| Model Name          | Training modality | Imagenet pretrained? | Model resolution  | Model  |
|-------------------------|-------------------|--------------------------|-----------------|------|
| [`ViT`](https://drive.google.com/drive/folders/1oKT2oAPp-7altFTvPKR2d7FdgXN9xMG3?usp=sharing)          | Mono              | Yes | 640 x 192                | ViT        |
| [`Resnet18`](https://drive.google.com/drive/folders/1QSHZjOk6Ufw52BGjJmuxV7PJQNisH5Kk?usp=sharing)        | Mono            | Yes | 640 x 192                |  Resnet18          |



<!-- [ViT](https://drive.google.com/drive/folders/1oKT2oAPp-7altFTvPKR2d7FdgXN9xMG3?usp=sharing)
[Resnet18](https://drive.google.com/drive/folders/1QSHZjOk6Ufw52BGjJmuxV7PJQNisH5Kk?usp=sharing) -->

## Training

The models can be trained on the KITTI dataset by running: 

```bash
bash experiments/train.sh
```

The hyperparameters are defined in each script file and set at their defaults as stated in the paper.

Feel free to vary which augmentations are used a fill list with their mappings are provided here:

--do_gauss Gaussian Noise
--do_shot Shot Noise
--do_impulse Impulse Noise
--do_defocus Defocus Blur
--do_glass Glass Blur

--do_zoom Zoom 
--do_snow Snow
--do_frost Frost
--do_elastic Elastic transform
--do_pixelate Pixelation

--do_jpeg_comp Jpeg Compresion
--do_color Brightness effect
--do_blur Motion Blur
--do_night Night time 
--do_fog Fog

--do_rain Rain 
--do_scale RA scailing
--do_tiling Tiling split
--do_vertical Vertical split
--do_erase Random erase

--do_flip Horizontal Flip
--do_greyscale Grey
--do_ground_snow Gropund Snow
--do_dusk Dusk time 
--do_dawn Dawn time

--R Red
--G Green
--B Blue 


## Evaluation
We test on each dataset:


### KITTI 

```
python Robust-Depth/evaluate_depth_MD2.py --eval_mono --load_weights_folder {weights_directory}
```

### KITTI Robust

```
python Robust-Depth/evaluate_depth_MD2.py --eval_mono --load_weights_folder {weights_directory} --robust_test
```
### KITTI Robust specific

```
python Robust-Depth/evaluate_depth_MD2.py --eval_mono --load_weights_folder {weights_directory} --robust_test --robust_augment blur
```

### KITTI Benchmark Robust

```
python Robust-Depth/evaluate_depth_MD2.py --eval_mono --load_weights_folder {weights_directory} --robust_test --eval_split eigen_benchmark
 
```

### DrivingStereo 

```
python Robust-Depth/evaluate_depth_MD2.py --eval_mono --load_weights_folder {weights_directory} --data_path {data_path} --eval_split foggy

```

### NuScenes 

```
python Robust-Depth/evaluate_depth_MD2.py --eval_mono --load_weights_folder {weights_directory} --data_path {data_path} --eval_split nuScenes_test_night
```

### Foggy CityScape 

```
python Robust-Depth/evaluate_depth_MD2.py --eval_mono --load_weights_folder {weights_directory} --eval_split cityscape --data_path {data_path} --foggy
```


## References

* [Monodepth2](https://github.com/nianticlabs/monodepth2) (ICCV 2019)
* [MonoViT](https://github.com/zxcqlf/MonoViT) 



