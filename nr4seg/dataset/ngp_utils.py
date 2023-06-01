import numpy as np
import torch
from packaging import version as pver


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose):
    new_pose = np.array(
        [
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3]],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3]],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return new_pose


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, error_map=None):
    """get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        direction_norms: [B, N, 1]
        inds: [B, N]
    """

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
    )
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    direction_norms = torch.norm(directions, dim=-1, keepdim=True)
    directions = directions / direction_norms
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results["rays_o"] = rays_o
    results["rays_d"] = rays_d
    results["direction_norms"] = direction_norms

    return results


# color palette for nyu40 labels
nyu40_colour_code = np.array([
    (0, 0, 0),
    (174, 199, 232),  # wall
    (152, 223, 138),  # floor
    (31, 119, 180),  # cabinet
    (255, 187, 120),  # bed
    (188, 189, 34),  # chair
    (140, 86, 75),  # sofa
    (255, 152, 150),  # table
    (214, 39, 40),  # door
    (197, 176, 213),  # window
    (148, 103, 189),  # bookshelf
    (196, 156, 148),  # picture
    (23, 190, 207),  # counter
    (178, 76, 76),  # blinds
    (247, 182, 210),  # desk
    (66, 188, 102),  # shelves
    (219, 219, 141),  # curtain
    (140, 57, 197),  # dresser
    (202, 185, 52),  # pillow
    (51, 176, 203),  # mirror
    (200, 54, 131),  # floor
    (92, 193, 61),  # clothes
    (78, 71, 183),  # ceiling
    (172, 114, 82),  # books
    (255, 127, 14),  # refrigerator
    (91, 163, 138),  # tv
    (153, 98, 156),  # paper
    (140, 153, 101),  # towel
    (158, 218, 229),  # shower curtain
    (100, 125, 154),  # box
    (178, 127, 135),  # white board
    (120, 185, 128),  # person
    (146, 111, 194),  # night stand
    (44, 160, 44),  # toilet
    (112, 128, 144),  # sink
    (96, 207, 209),  # lamp
    (227, 119, 194),  # bathtub
    (213, 92, 176),  # bag
    (94, 106, 211),  # other struct
    (82, 84, 163),  # otherfurn
    (100, 85, 144),  # other prop
]).astype(np.uint8)
