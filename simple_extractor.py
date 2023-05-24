#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""
from datetime import datetime
import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset

from datasets.simple_image_extractor import SimpleImageDataset

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe'],
        # 'indexs': [1,2,13,14,15]
        'indexs': [5, 6, 7, 12]
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'],
        'indexs': [1,2,11,12,13]
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
        'indexs': [1,3,4]
    }
}


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='checkpoints/final.pth', help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='inputs', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='outputs', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")
    parser.add_argument("--image-name", type=str, default='123.jpg', help="whether to save the logits.")


    return parser.parse_args()


def get_palette(num_cls, indexs):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        # lab = j
        # default to white
        palette[j * 3 + 0] = 255
        palette[j * 3 + 1] = 255
        palette[j * 3 + 2] = 255
        try:
            if (indexs.index(j) >= 0):
                # black the face, arm..
                palette[j * 3 + 0] = 0
                palette[j * 3 + 1] = 0
                palette[j * 3 + 2] = 0
        except ValueError:
            pass
        
        # i = 0
        # while lab:
        #     palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
        #     palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
        #     palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
        #     i += 1
        #     lab >>= 3
    return palette


def main():
    # test
    img = Image.open('inputs/123.jpg').convert('RGBA')

    result = parse_image(img)

    result.show()

    return
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    indexs = dataset_settings[args.dataset]['indexs']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    dataset = SimpleFolderDataset(root=args.input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    palette = get_palette(num_classes, indexs)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            output = model(image.cuda())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            parsing_result_path = os.path.join(args.output_dir, img_name[:-4] + '.png')
            output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            output_img.putpalette(palette)
            output_img.save(parsing_result_path)
            if args.logits:
                logits_result_path = os.path.join(args.output_dir, img_name[:-4] + '.npy')
                np.save(logits_result_path, logits_result)
    return

def parse_image(image, output_dir=None):
    dataset = 'atr'

    num_classes = dataset_settings[dataset]['num_classes']
    input_size = dataset_settings[dataset]['input_size']

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    model_restore = '/checkpoint/atr.pth'
    state_dict = torch.load(model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    indexs = dataset_settings[dataset]['indexs']
    palette = get_palette(num_classes, indexs)

    dataset = SimpleImageDataset(img=image, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    output_img = None
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            output = model(image.cuda())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            output_img.putpalette(palette)

    if output_img is not None and output_dir is not None:
        # 临时保存
        now = datetime.now()
        output_image_name = 'human_parse_' + now.strftime("%H:%M:%S") + '.png'
        parsing_result_path = os.path.join(output_dir, output_image_name)
        output_img.save(parsing_result_path)

    return output_img


if __name__ == '__main__':
    main()
