# Self-supervised Monocular Depth Estimation: Let's Talk About The Weather


 >**Self-supervised Monocular Depth Estimation: Let's Talk About The Weather
 >
 >[[PDF](LINK)] [[Video](LINK)]


https://github.com/kieran514/Robust-Depth/assets/51883968/a0cb2528-c351-42fb-be19-a2837800bcad



## Installation Setup

The models were trained using CUDA 11.1, Python 3.7.x (conda environment), and PyTorch 1.8.0.

Create a conda environment with the PyTorch library:

```bash
conda create -n my_env python=3.7.4 pytorch=1.8.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda activate my_env
```

Install prerequisite packages listed in requirements.txt:

```bash
conda install -c conda-forge matplotlib tqdm timm einops esri mmcv-full esri mmsegmentation
```

## Datasets

We use the raw [KITTI dataset](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) and specifically follow the instructions of [Monodepth2](https://github.com/nianticlabs/monodepth2)

When testing you need to downlaod the cityscape foggy dataset, the drivingstereo weather dataset (here), and the entire Nuscenens dataset (here). The splitds for the test have been provided. I dont believe I can legally provide the cityscape or the nuscenes files (dont need to for the drvivng stereo). Check so that I can. 

## Creating Augmentations For Any Dataset

### Night, Dawn & Dusk
We fist copy the repo from [CoMoGAN](https://github.com/astra-vision/CoMoGAN) into the CoMoGAN folder, we then exicute the script privded in the CoMoGAN folder to create NIGHT, DAWN and DUSK image augmetations. (Make sure to change the datapath)
```bash
bash CoMoGAN/comogan.sh 
```
Please direct over to the [CoMoGAN](https://github.com/astra-vision/CoMoGAN) github page for more information.

### Fog & Rain
We fist copy the repo from [rain-rendering](https://github.com/astra-vision/rain-rendering) into the rain-rendering folder, we then exicute the script privded in the rain-rendering folder to create NIGHT, DAWN and DUSK image augmetations. 

For fog generation, we were given this script persoanlly and will only share when I have premission. 

(Make sure to change the datapath)
```bash
bash rain-rendering/rain-rendering.sh 
```
Please direct over to the [rain-rendering](https://github.com/astra-vision/rain-rendering) github page for more information.

### Motion Blur & Snow
We fist copy the repo from [AutoMold](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library) into the AutoMold folder, we then exicute the script provided in the CoMoGAN folder to create Motion blur and snow augmetntaions. (Make sure to change the datapath)
```bash
bash AutoMold/automold.sh 
```
Please direct over to the [AutoMold](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library) github page for more information.

### Corruptions & RGBG
We fist copy the repo from [robustness](https://github.com/hendrycks/robustness) into the robustness folder, we then exicute the script provided in the robustness folder to create corruptions, red, blue, green and grey augmetntaions. (Make sure to change the datapath)
```bash
bash robustness/robustness.sh 
```
Please direct over to the [robustness](https://github.com/hendrycks/robustness) github page for more information.


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

Feel free to vary which augmentations are used a fill list with there mappings are provided here:

--do_gauss Gaussian Noise
--do_shot Shot Noise
--do_impulse Impluse Noise
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
We on each dataset:


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



