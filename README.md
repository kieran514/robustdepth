# Self-supervised Monocular Depth Estimation: Let's Talk About The Weather


 >**Self-supervised Monocular Depth Estimation: Let's Talk About The Weather
 >
 >[[PDF](LINK)] [[Video](LINK)]



https://user-images.githubusercontent.com/51883968/221583053-4819d852-a74e-4a93-a3b5-172422142d1a.mp4


## Install

The models were trained using CUDA 11.1, Python 3.7.x (conda environment), and PyTorch 1.8.0.

Create a conda environment with the PyTorch library:

```bash
conda create -n my_env python=3.7.4 pytorch=1.8.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda activate my_env
```

Install prerequisite packages listed in requirements.txt:

```bash
pip3 install -r requirements.txt
```

## Datasets

We use the raw [KITTI dataset](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) and specifically follow the intruction of [Monodepth2](https://github.com/nianticlabs/monodepth2)

## Creating Augmentations


### Creating Your Own Augmentations


## Selecting Augmentations


## Pretrained Models

[ViT](https://drive.google.com/drive/folders/1xY7n3kNhpoy1VM4ohmHYN1Oc_SVWwAWY?usp=sharing)
[Resnet18](https://drive.google.com/drive/folders/1xY7n3kNhpoy1VM4ohmHYN1Oc_SVWwAWY?usp=sharing)

## Training

The models can be trained on the KITTI dataset by running: 

```bash
bash experiments/train.sh
```

The hyperparameters are defined in each script file and set at their defaults as stated in the paper.


## Evaluation
We evaluate the models by running:


### KITTI 

```bash
python evalurate_depth_MD2.py ...
```

### KITTI Robust

```bash
python evalurate_depth_MD2.py ...
```

### DrivingStereo 

```bash
python evalurate_depth_MD2.py ...
```

### NuScenes 

```bash
python evalurate_depth_MD2.py ...
```

### Foggy CityScape 

```bash
python evalurate_depth_MD2.py ...
```


## References

* [Monodepth2](https://github.com/nianticlabs/monodepth2) (ICCV 2019)


