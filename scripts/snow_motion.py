
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

# Setting max severity for the test data 

if __name__ == "__main__":

    sequence = ['2011_09_26/',
            '2011_09_28/',
            '2011_09_29/',
            '2011_09_30/',
            '2011_10_03/']
    
    # sequence = '2011_09_26/'

    test = list(open(cwd + "/splits/eigen/test_files_og.txt", "r"))

    def process(seq):
        main_path = os.getcwd()
        
        sequences = os.path.join(main_path, 'data/KITTI_RAW', seq)
        for sync in os.listdir(sequences):
            if sync[-4:] == 'sync':
                img_path = os.path.join(sequences, sync, 'image_02/data')
                severity = random.random()
                for image in sorted(os.listdir(img_path)):
                    final_img_path = os.path.join(img_path, image)

                    checker = os.path.join(seq, sync) + " {:010d}".format(int(image[:-4])) + ' l\n'

                    if checker in test:
                        severity=1

                    im = Image.open(final_img_path)

                    blur_image = am.add_speed(np.array(im), speed_coeff=severity) 

                    output_blur = os.path.join(sequences, sync, 'image_02/blur/')

                    Path(output_blur).mkdir(parents=True, exist_ok=True)
                    output_im_blur = os.path.join(output_blur, image)
                    blur_image = Image.fromarray(np.uint8(blur_image)).convert('RGB')
                    blur_image.save(output_im_blur)


    print('---------------------------------------\n')
    print('Creating Augmented Snow and Motion Blur\n')
    print('---------------------------------------\n')

    # process(sequence)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process, sequence)







