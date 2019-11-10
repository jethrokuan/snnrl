"""Utility functions for model."""

import torch
from PIL import Image, ImageOps
import numpy as np

import torchvision
import torchvision.transforms.functional as F


import os
import os.path as osp
import shutil
import torch


def save_checkpoint(state, model, output_path, checkpoint, is_best=False):
    """Saves the model in specified output path."""
    filepath = osp.join(output_path, "model_{}.pt".format(checkpoint))
    if not osp.exists(output_path):
        os.mkdir(output_path)

    model_dict = {"model": model}
    torch.save({**state, **model_dict}, filepath)
    if is_best:
        shutil.copyfile(filepath, osp.join(output_path, "best.pt"))


class Invert(object):
    """Inverts the color channels of an PIL Image
    while leaving intact the alpha channel.
    """

    def invert(self, img):
        r"""Invert the input PIL Image.
        Args:
            img (PIL Image): Image to be inverted.
        Returns:
            PIL Image: Inverted image.
        """
        if not F._is_pil_image(img):
            raise TypeError("img should be PIL Image. Got {}".format(type(img)))

        if img.mode == "RGBA":
            r, g, b, a = img.split()
            rgb = Image.merge("RGB", (r, g, b))
            inv = ImageOps.invert(rgb)
            r, g, b = inv.split()
            inv = Image.merge("RGBA", (r, g, b, a))
        elif img.mode == "LA":
            l, a = img.split()
            l = ImageOps.invert(l)
            inv = Image.merge("LA", (l, a))
        else:
            inv = ImageOps.invert(img)
        return inv

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be inverted.
        Returns:
            PIL Image: Inverted image.
        """
        return self.invert(img)

    def __repr__(self):
        return self.__class__.__name__ + "()"
