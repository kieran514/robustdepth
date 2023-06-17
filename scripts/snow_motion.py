
from re import S
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
# import time
# plt.ion()
import os 
import glob
# import pdb
import random
import concurrent.futures
from pathlib import Path

import sys
cwd = os.getcwd()
sys.path.insert(1, cwd + '/Automold')

import Automold as am

# import math
random.seed(42)
import time

if __name__ == "__main__":
    start = time.time()

    sequence = ['2011_09_26/',
            '2011_09_28/',
            '2011_09_29/',
            '2011_09_30/',
            '2011_10_03/']
    
    test = list(open(cwd + "/splits/eigen/test_files_og.txt", "r"))
    start = time.time()

    def process(sync):
        if sync[-4:] == 'sync':
            img_path = os.path.join(sync, 'image_02/data')
            severity = random.random()
            for image in sorted(os.listdir(img_path)):
                final_img_path = os.path.join(img_path, image)

                checker = os.path.join(seq, sync) + " {:010d}".format(int(image[:-4])) + ' l\n'

                if checker in test:
                    severity=1

                im = Image.open(final_img_path)

                blur_image = am.add_speed(np.array(im), speed_coeff=severity) 
                snow_image = am.add_snow(np.array(im), snow_coeff=severity) 

                output_blur = os.path.join(sync, 'image_02/blur/')
                output_snow = os.path.join(sync, 'image_02/ground_snow/')

                Path(output_blur).mkdir(parents=True, exist_ok=True)
                Path(output_snow).mkdir(parents=True, exist_ok=True)

                output_im_blur = os.path.join(output_blur, image)
                output_im_snow = os.path.join(output_snow, image)

                blur_image = Image.fromarray(np.uint8(blur_image)).convert('RGB')
                snow_image = Image.fromarray(np.uint8(snow_image)).convert('RGB')

                blur_image.save(output_im_blur)
                snow_image.save(output_im_snow)

    print('---------------------------------------\n')
    print('Creating Augmented Snow and Motion Blur\n')
    print('---------------------------------------')

    for seq in sequence:
        seq = os.path.join(os.getcwd(), 'data/KITTI_RAW', seq)
        sync = [os.path.join(seq, x) for x in sorted(os.listdir(seq))]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(process, sync)

    end = time.time()
    print(f'Time taken {end - start}')








