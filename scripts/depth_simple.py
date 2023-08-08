import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import tqdm
import sys
from pathlib import Path
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, os.path.join(os.getcwd(), 'Robust-Depth'))
import networks
import torch
from torchvision import transforms, datasets

from layers import disp_to_depth

def test_simple():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = os.path.join(os.getcwd(), "pretrained", "mono+stereo_640x192")
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    sequence = ['2011_09_26/',
            '2011_09_28/',
            '2011_09_29/',
            '2011_09_30/',
            '2011_10_03/']

    for seq in sequence:
        for sync in os.listdir(os.path.join(os.getcwd(), 'data/KITTI_RAW', seq)):
            sync = os.path.join(os.getcwd(), 'data/KITTI_RAW', seq, sync)
            if sync[-4:] == 'sync':
                img_path = os.path.join(sync, 'image_02/data')
                with torch.no_grad():
                    for image in tqdm.tqdm(sorted(os.listdir(img_path))):
                        final_img_path = os.path.join(img_path, image)

                        input_image = pil.open(final_img_path).convert('RGB')
                        original_width, original_height = input_image.size
                        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                        input_image = input_image.to(device)
                        features = encoder(input_image)
                        outputs = depth_decoder(features)

                        disp = outputs[("disp", 0)]
                        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
                        depth_resized = torch.nn.functional.interpolate(
                            depth, (original_height, original_width), mode="bilinear", align_corners=False)

                        disp_resized_np = depth_resized.squeeze().cpu().numpy() *255
                        vmax = np.percentile(disp_resized_np, 95)
                        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                        mapper = cm.ScalarMappable(norm=normalizer)
                        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                        im = pil.fromarray(colormapped_im)
                        im = im.convert('L')

                        output_depth = os.path.join(sync, 'image_02/depth/')

                        Path(output_depth).mkdir(parents=True, exist_ok=True)

                        name_dest_im = os.path.join(output_depth, "{}.png".format(image[:-4]))
                        im.save(name_dest_im)

if __name__ == '__main__':
    test_simple()