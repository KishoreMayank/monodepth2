import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import time

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def depth_check(args):
    depth = args.depth_output
    depth_array = np.load(depth, allow_pickle=True)
    print(depth_array.shape)



def parse_args():
    parser = argparse.ArgumentParser(
        description='Numpy array to depth estimation')

    parser.add_argument('--depth_output', type=str, help='path to a test image or folder of images', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    count = depth_check(args)
    total_time = time.time() - start
    print("Total time taken: ", total_time)
    print(f"Processed every {args.fps} frame to get an average time of {total_time/count}")