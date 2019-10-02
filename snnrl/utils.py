"""Utility functions for model."""

import os.path as osp
import shutil
import torch

def save_checkpoint(model, output_path, checkpoint, is_best=False):
    """Saves the model in specified output path."""
    filepath = osp.join(output_path, "model_{}.pt".format(checkpoint))
    if not osp.exists(output_path):
        os.mkdir(output_path)

    torch.save(model.state_dict(), filepath)
    if is_best:
        shutil.copyfile(filepath, osp.join(output_path, "best.pt"))
