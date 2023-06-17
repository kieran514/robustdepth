
import numpy as np
from PIL import Image
import os 
import random
import concurrent.futures
from pathlib import Path

import sys
cwd = os.getcwd()
sys.path.insert(1, cwd + '/corruption/')

from corruptions import *
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

    def process(sync):
        matrix_R=(1, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0)

        matrix_G=(0, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 0, 0)

        matrix_B=(0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 1, 0)

        if sync[-4:] == 'sync':
            img_path = os.path.join(sync, 'image_02/data')
            severity = random.randint(1, 5)
            for image in sorted(os.listdir(img_path)):
                final_img_path = os.path.join(img_path, image)

                checker = os.path.join(seq, sync) + " {:010d}".format(int(image[:-4])) + ' l\n'

                if checker in test:
                    severity=5

                im = Image.open(final_img_path)
                np_im = np.array(im)

                gaussian_noise_image = gaussian_noise(np_im, severity=severity) 
                shot_noise_image = shot_noise(np_im, severity=severity) 
                impulse_noise_image = impulse_noise(np_im, severity=severity) 
                defocus_blur_image = defocus_blur(np_im, severity=severity) 
                glass_blur_image = glass_blur(np_im, severity=severity) 
                zoom_blur_image = zoom_blur(np_im, severity=severity) 
                snow_image = snow(np_im, severity=severity) 
                frost_image = frost(np_im, severity=severity) 
                elastic_transform_image = elastic_transform(np_im, severity=severity) 
                pixelate_ = pixelate(im, severity=severity) 
                jpeg_compression_ = jpeg_compression(im, severity=severity) 
                grey = im.convert('L')
                image_R = im.convert("RGB", matrix_R)
                image_G = im.convert("RGB", matrix_G)
                image_B = im.convert("RGB", matrix_B)

                output_gaussian_noise = os.path.join(sync, 'image_02/gaussian_noise/')
                output_shot_noise = os.path.join(sync, 'image_02/shot_noise/')
                output_impulse_noise = os.path.join(sync, 'image_02/impulse_noise/')
                output_defocus_blur = os.path.join(sync, 'image_02/defocus_blur/')
                output_glass_blur = os.path.join(sync, 'image_02/glass_blur/')
                output_zoom_blur = os.path.join(sync, 'image_02/zoom_blur/')
                output_snow = os.path.join(sync, 'image_02/snow/')
                output_frost = os.path.join(sync, 'image_02/frost/')
                output_elastic_transform = os.path.join(sync, 'image_02/elastic_transform/')
                output_pixelate = os.path.join(sync, 'image_02/pixelate/')
                output_jpeg_compression = os.path.join(sync, 'image_02/jpeg_compression/')
                output_grey = os.path.join(sync, 'image_02/greyscale/')
                output_R = os.path.join(sync, 'image_02/R/')
                output_G = os.path.join(sync, 'image_02/G/')
                output_B = os.path.join(sync, 'image_02/B/')

                Path(output_gaussian_noise).mkdir(parents=True, exist_ok=True)
                Path(output_shot_noise).mkdir(parents=True, exist_ok=True)
                Path(output_impulse_noise).mkdir(parents=True, exist_ok=True)
                Path(output_defocus_blur).mkdir(parents=True, exist_ok=True)
                Path(output_glass_blur).mkdir(parents=True, exist_ok=True)
                Path(output_zoom_blur).mkdir(parents=True, exist_ok=True)
                Path(output_snow).mkdir(parents=True, exist_ok=True)
                Path(output_frost).mkdir(parents=True, exist_ok=True)
                Path(output_elastic_transform).mkdir(parents=True, exist_ok=True)
                Path(output_pixelate).mkdir(parents=True, exist_ok=True)
                Path(output_jpeg_compression).mkdir(parents=True, exist_ok=True)
                Path(output_grey).mkdir(parents=True, exist_ok=True)
                Path(output_R).mkdir(parents=True, exist_ok=True)
                Path(output_G).mkdir(parents=True, exist_ok=True)
                Path(output_B).mkdir(parents=True, exist_ok=True)

                output_im_gaussian_noise = os.path.join(output_gaussian_noise, image)
                output_im_shot_noise = os.path.join(output_shot_noise, image)
                output_im_impulse_noise = os.path.join(output_impulse_noise, image)
                output_im_defocus_blur = os.path.join(output_defocus_blur, image)
                output_im_glass_blur = os.path.join(output_glass_blur, image)
                output_im_zoom_blur = os.path.join(output_zoom_blur, image)
                output_im_snow = os.path.join(output_snow, image)
                output_im_frost = os.path.join(output_frost, image)
                output_im_elastic_transform = os.path.join(output_elastic_transform, image)
                output_im_pixelate = os.path.join(output_pixelate, image)
                output_im_jpeg_compression = os.path.join(output_jpeg_compression, image)
                output_grey_im = os.path.join(output_grey, image)
                output_red_im = os.path.join(output_R, image)
                output_green_im = os.path.join(output_G, image)
                output_blue_im = os.path.join(output_B, image)

                gaussian_noise_ = Image.fromarray(np.uint8(gaussian_noise_image)).convert('RGB')
                shot_noise_ = Image.fromarray(np.uint8(shot_noise_image)).convert('RGB')
                impulse_noise_ = Image.fromarray(np.uint8(impulse_noise_image)).convert('RGB')
                defocus_blur_ = Image.fromarray(np.uint8(defocus_blur_image)).convert('RGB')
                glass_blur_ = Image.fromarray(np.uint8(glass_blur_image)).convert('RGB')
                zoom_blur_ = Image.fromarray(np.uint8(zoom_blur_image)).convert('RGB')
                snow_ = Image.fromarray(np.uint8(snow_image)).convert('RGB')
                frost_ = Image.fromarray(np.uint8(frost_image)).convert('RGB')
                elastic_transform_ = Image.fromarray(np.uint8(elastic_transform_image)).convert('RGB')

                gaussian_noise_.save(output_im_gaussian_noise)
                shot_noise_.save(output_im_shot_noise)
                impulse_noise_.save(output_im_impulse_noise)
                defocus_blur_.save(output_im_defocus_blur)
                glass_blur_.save(output_im_glass_blur)
                zoom_blur_.save(output_im_zoom_blur)
                snow_.save(output_im_snow)
                frost_.save(output_im_frost)
                elastic_transform_.save(output_im_elastic_transform)
                pixelate_.save(output_im_pixelate)
                jpeg_compression_.save(output_im_jpeg_compression)
                grey.save(output_grey_im)
                image_R.save(output_red_im)
                image_G.save(output_green_im)
                image_B.save(output_blue_im)

    print('---------------------------------------\n')
    print('Creating Corruptions & RGB-GREY\n')
    print('---------------------------------------')

    for seq in sequence:
        seq = os.path.join(os.getcwd(), 'data/KITTI_RAW', seq)
        sync = [os.path.join(seq, x) for x in sorted(os.listdir(seq))]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(process, sync)

    end = time.time()
    print(f'Time taken {end - start}')







