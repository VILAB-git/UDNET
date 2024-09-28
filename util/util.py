"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
import torchvision


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        # import pdb; pdb.set_trace()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        bin_ = image_numpy.shape[0] // 2
        if bin_ > 1:
            output_images = np.zeros([bin_, image_numpy.shape[1], image_numpy.shape[2], 3])
            for i in range(bin_):
                # import pdb; pdb.set_trace()
                output_images[i] = gen_event_images_pos_neg(image_numpy[2*i:2*i+2])
            image_output = np.concatenate(output_images, axis=1)
        elif bin_ == 1:
            image_output = gen_event_images_pos_neg(image_numpy)
        else:
            image_output = input_image
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_output = input_image
    return image_output.astype(imtype)

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def correct_resize_label(t, size):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i, :1]
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        resized_t = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def correct_resize(t, size, mode=Image.BICUBIC):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i:i + 1]
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def gen_event_images_pos_neg(event_volume):
    """
    event_cnt: np.ndarray, HxWx2, 0 for positive, 1 for negative

    'gray': white for positive, black for negative
    'green_red': green for positive, red for negative
    'blue_red': blue for positive, red for negative
    """
    # print(event_volume.shape)
    event_cnt = event_volume.transpose(1,2,0)+1.0
    pos = event_cnt[:, :, 0]
    neg = event_cnt[:, :, 1]
    pos_max = np.percentile(pos, 99)
    pos_min = np.percentile(pos, 1)
    neg_max = np.percentile(neg, 99)
    neg_min = np.percentile(neg, 1)
    max = pos_max if pos_max > neg_max else neg_max

    # if is_norm:
    if pos_min != max:
        pos = (pos - pos_min) / (max - pos_min)
    if neg_min != max:
        neg = (neg - neg_min) / (max - neg_min)
    # else:
    #     mask_pos_nonzero = pos != 0
    #     mask_neg_nonzero = neg != 0
    #     mask_posnonnorm = (pos >= neg) * mask_pos_nonzero
    #     mask_negnonnorm = (pos < neg) * mask_neg_nonzero
    #     pos[mask_posnonnorm] = 1
    #     neg[mask_posnonnorm] = 0
    #     neg[mask_negnonnorm] = 1
    #     pos[mask_negnonnorm] = 0

    pos = np.clip(pos, 0, 1)
    neg = np.clip(neg, 0, 1)

    event_image = np.ones((event_cnt.shape[0], event_cnt.shape[1]))
    event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)

    mask_pos = pos > 0
    mask_neg = neg > 0
    mask_not_pos = pos == 0
    mask_not_neg = neg == 0

    # if is_black_background:
    #     event_image *= 0
    #     event_image[:, :, 1][mask_pos] = 0
    #     event_image[:, :, 0][mask_pos] = pos[mask_pos]
    #     event_image[:, :, 2][mask_pos * mask_not_neg] = 0
    #     event_image[:, :, 2][mask_neg] = neg[mask_neg]
    #     event_image[:, :, 1][mask_neg] = 0
    #     event_image[:, :, 0][mask_neg * mask_not_pos] = 0
    # else:
    # only pos
    event_image[:, :, 0][mask_pos * mask_not_neg] = 1 
    event_image[:, :, 1][mask_pos * mask_not_neg] = 1 - pos[mask_pos * mask_not_neg]
    event_image[:, :, 2][mask_pos * mask_not_neg] = 1 - pos[mask_pos * mask_not_neg]
    # only neg
    event_image[:, :, 2][mask_neg * mask_not_pos] = 1
    event_image[:, :, 0][mask_neg * mask_not_pos] = 1 - neg[mask_neg * mask_not_pos]
    event_image[:, :, 1][mask_neg * mask_not_pos] = 1 - neg[mask_neg * mask_not_pos]
    ######### pos + neg
    mask_posoverneg = pos >= neg
    mask_negoverpos = pos < neg
    # pos >= neg
    event_image[:, :, 0][mask_pos * mask_neg * mask_posoverneg] = 1 
    event_image[:, :, 1][mask_pos * mask_neg * mask_posoverneg] = 1 - pos[mask_pos * mask_neg * mask_posoverneg]
    event_image[:, :, 2][mask_pos * mask_neg * mask_posoverneg] = 1 - pos[mask_pos * mask_neg * mask_posoverneg]
    # pos < neg
    event_image[:, :, 2][mask_pos * mask_neg * mask_negoverpos] = 1
    event_image[:, :, 0][mask_pos * mask_neg * mask_negoverpos] = 1 - neg[mask_pos * mask_neg * mask_negoverpos]
    event_image[:, :, 1][mask_pos * mask_neg * mask_negoverpos] = 1 - neg[mask_pos * mask_neg * mask_negoverpos]

    event_image = (event_image * 255)#.astype(np.uint8)

    return event_image