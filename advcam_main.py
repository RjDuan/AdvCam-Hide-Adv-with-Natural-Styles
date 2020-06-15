# -*- coding: utf-8 -*-
import argparse
from PIL import Image
import numpy as np
import os
from advcam_physical_attack import attack
from utils import *


parser = argparse.ArgumentParser()
# Input Options

parser.add_argument("--content_image_path", dest='content_image_path',  nargs='?',
                    help="Path to the content image",default='')
parser.add_argument("--style_image_path",   dest='style_image_path',    nargs='?',
                    help="Path to the style image",default='')
parser.add_argument("--content_seg_path",   dest='content_seg_path',    nargs='?',
                    help="Path to the style segmentation",default='')
parser.add_argument("--style_seg_path",     dest='style_seg_path',      nargs='?',
                    help="Path to the style segmentation",default='')
parser.add_argument("--background_path",    dest='background_path',     nargs='?',
                    help="Path to init image", default='')
parser.add_argument("--result_dir",             dest='result_dir',              nargs='?',
                    help='Path to save the results', default='')
parser.add_argument("--serial",             dest='serial',              nargs='?',
                    help='Path to save the serial out_iter_X.png', default='')

# Training Optimizer Options
parser.add_argument("--max_iter",           dest='max_iter',            nargs='?', type=int,
                    help='maximum image iteration', default=4000)
parser.add_argument("--learning_rate",      dest='learning_rate',       nargs='?', type=float,
                    help='learning rate for adam optimizer', default=1.0)
parser.add_argument("--save_iter",          dest='save_iter',           nargs='?', type=int,
                    help='save temporary result per iterations', default=50)


# Weight Options
parser.add_argument("--content_weight",     dest='content_weight',      nargs='?', type=float,
                    help="weight of content loss", default=5e0)
parser.add_argument("--style_weight",       dest='style_weight',        nargs='?', type=float,
                    help="weight of style loss", default=1e2)
parser.add_argument("--tv_weight",          dest='tv_weight',           nargs='?', type=float,
                    help="weight of total variational loss", default=1e-3)
parser.add_argument("--attack_weight",      dest='attack_weight',       nargs='?', type=float,
                    help="weight of attack loss", default=5e3)



# Attack Options
parser.add_argument("--targeted_attack",       dest='targeted_attack',        nargs='?', type=int,
                    help="if True, targeted attack", default=1)
parser.add_argument("--target_label",       dest='target_label',        nargs='?', type = int,
                    help="The target label for target attack", default=184)
parser.add_argument("--true_label",       dest='true_label',        nargs='?', type = int,
                    help="The target label for target attack", default=8)
parser.add_argument("--cross_class",       dest='cross_class',        nargs='?', type = bool,
                    help="if True, apply photostyle transfer attack between different class", default=False)


# test mode
parser.add_argument("--test_mode",             dest='test_mode',              nargs='?',
                    help="content/tv/affine/all", default='all')


args = parser.parse_args()


if __name__ == "__main__":
    content_name = args.content_image_path.split('/')[-1]
    args.style_image_path = os.path.join(args.style_image_path, content_name)

    content_seg_name = content_name.split('.')[0] + '.jpg'
    args.content_seg_path = os.path.join(args.content_seg_path, content_name)

    style_name = args.style_image_path.split('/')[-1]
    args.style_seg_path = os.path.join(args.style_seg_path, style_name)

    result_path = os.path.join(args.result_dir, content_name.split('.')[0], str(args.target_label) + '_' + str(args.attack_weight)+'_'+ style_name.split('.')[0] )
    path_exist = os.path.exists(result_path)
    args.serial = result_path
    if not path_exist:
        os.makedirs(result_path)

    attack(args)

