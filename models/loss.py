# Author: Yash Bhalgat

from math import exp, log, floor
import torch
import torch.nn.functional as F
import pdb

try:
    from .utils import hash
except Exception:
    # Fallback for running this file outside package context
    from models.utils import hash


def total_variation_loss(
    embeddings, min_resolution, max_resolution, level, log2_hashmap_size, n_levels=16
):
    # ensure computations run on the same device as embeddings
    device = embeddings.weight.device

    # Get resolution (scalar int on CPU; cheap) then create tensors on device
    b = exp((log(max_resolution) - log(min_resolution)) / (n_levels - 1))
    resolution_val = int(floor(min_resolution * (b**level)))

    # Cube size to apply TV loss (scalar ints)
    min_cube_size = int(min_resolution - 1)
    max_cube_size = 50  # can be tuned
    if min_cube_size > max_cube_size:
        print("ALERT! min cuboid size greater than max!")
        pdb.set_trace()
    cube_size_val = int(
        max(min_cube_size, min(max_cube_size, resolution_val // 10))
    )

    # Sample cuboid (all tensors created on target device)
    high = max(1, resolution_val - cube_size_val)
    min_vertex = torch.randint(0, high, (3,), device=device)
    idx_x = torch.arange(min_vertex[0], min_vertex[0] + cube_size_val + 1, device=device)
    idx_y = torch.arange(min_vertex[1], min_vertex[1] + cube_size_val + 1, device=device)
    idx_z = torch.arange(min_vertex[2], min_vertex[2] + cube_size_val + 1, device=device)

    # meshgrid on device
    gx, gy, gz = torch.meshgrid(idx_x, idx_y, idx_z, indexing='ij')
    cube_indices = torch.stack([gx, gy, gz], dim=-1)

    # hashing and embedding lookup on device
    hashed_indices = hash(cube_indices, log2_hashmap_size)
    hashed_indices = hashed_indices.to(device)
    cube_embeddings = embeddings(hashed_indices)
    # hashed_idx_offset_x = hash(idx+torch.tensor([1,0,0]), log2_hashmap_size)
    # hashed_idx_offset_y = hash(idx+torch.tensor([0,1,0]), log2_hashmap_size)
    # hashed_idx_offset_z = hash(idx+torch.tensor([0,0,1]), log2_hashmap_size)

    # Compute loss
    # tv_x = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_x), 2).sum()
    # tv_y = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_y), 2).sum()
    # tv_z = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_z), 2).sum()
    tv_x = torch.pow(
        cube_embeddings[1:, :, :, :] - cube_embeddings[:-1, :, :, :], 2
    ).sum()
    tv_y = torch.pow(
        cube_embeddings[:, 1:, :, :] - cube_embeddings[:, :-1, :, :], 2
    ).sum()
    tv_z = torch.pow(
        cube_embeddings[:, :, 1:, :] - cube_embeddings[:, :, :-1, :], 2
    ).sum()

    return (tv_x + tv_y + tv_z) / cube_size_val


def sigma_sparsity_loss(sigmas):
    # Using Cauchy Sparsity loss on sigma values
    return torch.log(1.0 + 2 * sigmas**2).sum(dim=-1)
