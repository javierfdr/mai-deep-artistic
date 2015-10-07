#!/usr/bin/env python

import os
import argparse
import numpy as np
import scipy.misc
import deeppy as dp

from matconvnet import vgg19_net
from style_network import StyleNetwork
from neural_artistic_style import imread
from neural_artistic_style import weight_array
from neural_artistic_style import to_bc01
from gram_net import GramNet


def weight_tuple(s):
    print("You called me!")
    try:
        conv_idx, weight = map(float, s.split(','))
        return conv_idx, weight
    except:
        raise argparse.ArgumentTypeError('weights must by "int,float"')


def run():
    parser = argparse.ArgumentParser(
        description='Neural artistic style. Generates an image by combining '
                    'the subject from one image and the style from another.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Added by Phil
    parser.add_argument('--src1', required=True, type=str,
                        help='First image to be compared.')
    parser.add_argument('--vgg19', default='imagenet-vgg-verydeep-19.mat',
                        type=str, help='VGG-19 .mat file.')
    parser.add_argument('--pool-method', default='avg', type=str,
                        choices=['max', 'avg'], help='Subsampling scheme.')
    parser.add_argument('--style-weights', nargs='*', type=weight_tuple,
                        default=[(0, 1), (2, 1), (4, 1), (8, 1), (12, 1)],
                        help='List of style weights (conv_idx,weight).')
    args = parser.parse_args()

    layers, img_mean = vgg19_net(args.vgg19, pool_method=args.pool_method)

    # Inputs
    pixel_mean = np.mean(img_mean, axis=(0, 1))
    style_img = imread(args.src1) - pixel_mean

    # Setup network
    style_weights = weight_array(args.style_weights)
    net = GramNet(layers, to_bc01(style_img), style_weights)

    for i, gram in enumerate(net.style_grams):
        if gram is not None:
            print("{}: Shape: {}".format(i, net.style_grams[i].shape))
        else:
            print("{}: None".format(i))

if __name__ == "__main__":
    run()
