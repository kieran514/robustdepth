#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:37:53 2017

@author: sulekahraman
"""

"""
AVERAGE BETA OVER THE RAYYY

===============================================
Simplex Noise for Heterogenous Fog with Pinhole Camera Model
===============================================

Papers used: 
1. Haze Visibility Enhancement: A Survey and Quantitative Benchmarking
 Yu Li, Shaodi You, Michael S. Brown, and Robby T. Tan,

2. Vision and the Atmosphere
SRINIVASA G. NARASIMHAN AND SHREE K. NAYAR

3. Simplex noise demystified, Stefan Gustavson

This code was provided by;

https://github.com/astra-vision/rain-rendering

@article{tremblay2020rain,
  title={Rain Rendering for Evaluating and Improving Robustness to Bad Weather},
  author={Tremblay, Maxime and Halder, Shirsendu S. and de Charette, Raoul and Lalonde, Jean-François},
  journal={International Journal of Computer Vision},
  year={2020}
}

"""

# ------------------------------------------------------------
# MODULES
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import SimplexNoise
import time
import concurrent.futures

plt.ion()
import os 
import glob
import pdb
import random
import cv2
from pathlib import Path
random.seed(42)
import time
cwd = os.getcwd()

if __name__ == "__main__":

    sequence = ['2011_09_26/',
            '2011_09_28/',
            '2011_09_29/',
            '2011_09_30/',
            '2011_10_03/']

    test = list(open(cwd + "/splits/eigen/test_files_og.txt", "r"))
    start = time.time()

    def process(data_imgs):
        
        image_path = os.path.join(sync, f'image_02/{data_imgs}')
        depth_path = os.path.join(sync, 'image_02/depth')

        for file_image in sorted(os.listdir(image_path)):
            final_image_path = os.path.join(image_path, file_image)
            final_depth_path = os.path.join(depth_path, file_image.replace('.jpg', '.png'))

            checker = os.path.join(seq, sync) + " {:010d}".format(int(file_image[:-4])) + ' l\n'

            if checker in test:
                beta=1
            else:
                beta = beta_origin

            if data_imgs == 'data':
                brightness = 255
                output = 'fog'
            elif data_imgs == 'night':
                brightness= 50
                output = 'fog+night'
            elif data_imgs == 'dawn':
                brightness= 75
                output = 'dawn+fog'
            elif data_imgs == 'dusk':
                brightness= 75
                output = 'dusk+fog'
            elif data_imgs == 'dawn+rain':
                brightness= 75
                output = 'dawn+rain+fog'
            elif data_imgs == 'rain+night':
                brightness= 50
                output = 'rain+fog+night'
            elif data_imgs == 'rain':
                brightness= 200
                output = 'rain+fog'
            elif data_imgs == 'dusk+rain':
                brightness= 75
                output = 'dusk+rain+fog'

            LInf = np.array([brightness-5, brightness, brightness]) 

            image = cv2.imread(final_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

            depth = cv2.imread(final_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            w, h = (depth.shape[1], depth.shape[0])
            image = cv2.resize(image, (w, h))
            depth = depth * 0.01

            height, width = image.shape[:2]

            k = 0.5

            fu_inv = 1.0 / (0.58 * w)
            fv_inv = 1.0 / (1.92 * h)

            x_ = np.linspace(0., width, width, endpoint=False) - (0.5 * w)
            y_ = np.linspace(0., height, height, endpoint=False) + (0.5 * h)

            y_, x_ = np.meshgrid(y_, x_, indexing='ij')

            depthN = 1
            noise = np.zeros_like(x_)
            simplex = SimplexNoise.SimplexNoise()  
            simplex.setup(depth)
            for i in range(depthN):
                Z = depth * i / depthN
                X = Z * x_ * fu_inv  
                Y = Z * y_ * fv_inv
                noise += simplex.noise3d(X / 2000., Y / 2000., Z / 2000.) / depthN

            transmission_noise = np.zeros_like(image, dtype=np.float64)
            direct_trans_noise = np.zeros_like(image)
            airlight_noise = np.zeros_like(image)

            beta_noise_ave = beta * (1.5 + k * noise)

            transmission_noise[:, :, 0] = np.exp(-beta_noise_ave * depth)
            transmission_noise[:, :, 1] = np.exp(-beta_noise_ave * depth)
            transmission_noise[:, :, 2] = np.exp(-beta_noise_ave * depth)
            direct_trans_noise = image * transmission_noise
            airlight_noise = LInf * (1 - transmission_noise)
            foggy_noise = direct_trans_noise + airlight_noise
            foggy_noise = np.asarray(foggy_noise, dtype=np.uint8)

            output_fog = os.path.join(sync, f'image_02/{output}/')
            output_im_fog = os.path.join(output_fog, file_image)

            Path(output_fog).mkdir(parents=True, exist_ok=True)

            pil_image = Image.fromarray(foggy_noise).convert('RGB')
            pil_image.save(output_im_fog)

    print('---------------------------------------\n')
    print('Creating Fog\n')
    print('---------------------------------------\n')

    for seq in sequence:
        seq = os.path.join(os.getcwd(), 'data/KITTI_RAW', seq)
        syncs = [os.path.join(seq, x) for x in sorted(os.listdir(seq))]
        for sync in syncs:
            if sync[-4:] == 'sync':
                global beta_origin
                beta_origin = random.random()
                print(f'Working in {sync} with a beta value {beta_origin}\n')
                data_imgs = ['data', 'night', 'dawn', 'dusk', 'rain+night', 'rain', 'dusk+rain', 'dawn+rain']
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    executor.map(process, data_imgs)

    end = time.time()
    print(f'Time taken {end - start}')





