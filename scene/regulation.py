"""Regularization losses for HexPlane-based deformation fields."""

import torch


def compute_plane_tv(t):
    """Total variation regularization on 2D feature planes."""
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., :h - 1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, :w - 1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)


def compute_plane_smoothness(t):
    """Second-order smoothness regularization on temporal planes."""
    batch_size, c, h, w = t.shape
    first_difference = t[..., 1:, :] - t[..., :h - 1, :]
    second_difference = first_difference[..., 1:, :] - first_difference[..., :h - 2, :]
    return torch.square(second_difference).mean()
