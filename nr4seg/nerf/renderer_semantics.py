import math
import numpy as np
import torch
import torch.nn as nn
import trimesh

from .raymarching import raymarching


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0.0 + 0.5 / n_samples,
                           1.0 - 0.5 / n_samples,
                           steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print("[visualize points]", pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class SemanticNeRFRenderer(nn.Module):

    def __init__(
        self,
        bound=1,
        cuda_ray=False,
        density_scale=1,  # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
        num_semantic_classes=41,
    ):
        super().__init__()

        self.epoch = 1
        self.weights = np.zeros([0])
        self.weights_sum = np.zeros([0])

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.density_scale = density_scale
        self.num_semantic_classes = num_semantic_classes

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor(
            [-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer("aabb_train", aabb_train)
        self.register_buffer("aabb_infer", aabb_infer)

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade] +
                                       [128] * 3)  # [CAS, H, H, H]
            self.register_buffer("density_grid", density_grid)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(
                16, 2, dtype=torch.int32)  # 16 is hardcoded for averaging...
            self.register_buffer("step_counter", step_counter)
            self.mean_count = 0
            self.local_step = 0

    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def run(self,
            rays_o,
            rays_d,
            direction_norms,
            num_steps=256,
            upsample_steps=256,
            bg_color=None,
            perturb=False,
            epoch=None,
            **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # direction_norms: [B, N, 1]
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        direction_norms = direction_norms.contiguous().view(-1)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        z_vals = torch.linspace(0.0, 1.0, num_steps,
                                device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=device)

            z_vals = lower + (upper - lower) * t_rand

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(
            -1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        # query density and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))

        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]
                deltas = torch.cat(
                    [deltas, 1e10 * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(
                    -deltas * self.density_scale *
                    density_outputs["sigma"].squeeze(-1))  # [N, T]
                alphas_shifted = torch.cat(
                    [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15],
                    dim=-1,
                )  # [N, T+1]
                weights = (alphas *
                           torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
                          )  # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]
                             )  # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid,
                                        weights[:, 1:-1],
                                        upsample_steps,
                                        det=False).detach()  # [N, t]

                new_xyzs = rays_o.unsqueeze(
                    -2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(
                        -1)  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]),
                                     aabb[3:])  # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            # new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1)  # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1)  # [N, T+t, 3]
            xyzs = torch.gather(xyzs,
                                dim=1,
                                index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat(
                    [density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(
                    tmp_output,
                    dim=1,
                    index=z_index.unsqueeze(-1).expand_as(tmp_output),
                )

        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat([deltas, 1e10 * torch.ones_like(deltas[..., :1])],
                           dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale *
                               density_outputs["sigma"].squeeze(-1))  # [N, T+t]
        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15],
            dim=-1)  # [N, T+t+1]
        weights = (alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
                  )  # [N, T+t]

        mask_rgb = weights > 1e-4  # hard coded
        mask_semantics = weights > 1e-4  # hard coded

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        rgbs = self.color(xyzs.reshape(-1, 3),
                          dirs.reshape(-1, 3),
                          mask=mask_rgb.reshape(-1),
                          **density_outputs)
        rgbs = rgbs.view(N, -1, 3)  # [N, T+t, 3]

        local_semantics = self.semantics(xyzs.reshape(-1, 3),
                                         dirs.reshape(-1, 3),
                                         mask=mask_semantics.reshape(-1),
                                         **density_outputs)
        local_semantics = local_semantics.view(
            N, -1, self.num_semantic_classes)  # [N, T+t, 3]

        # calculate weight_sum (mask)
        weights_semantics = weights.clone().detach()
        weights[torch.logical_not(mask_rgb)] = 0.0

        # calculate depth
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        # depth = torch.sum(weights * ori_z_vals, dim=-1)
        depth = torch.sum(weights * z_vals, dim=-1)
        depth = depth / direction_norms

        # calculate color

        image = torch.sum(weights.unsqueeze(-1) * rgbs,
                          dim=-2)  # [N, 3], in [0, 1]
        weights_semantics[torch.logical_not(mask_semantics)] = 0.0
        semantics = torch.sum(weights_semantics.unsqueeze(-1) * local_semantics,
                              dim=-2)  # [N, C], in [0, 1]

        # mix background color
        if bg_color is None:
            bg_color = 1

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)
        semantics = semantics.view(*prefix, self.num_semantic_classes)

        return {
            "depth": depth,
            "image": image,
            "semantics": semantics,
        }

    def render(self,
               rays_o,
               rays_d,
               direction_norms,
               staged=False,
               max_ray_batch=4096,
               bg_color=None,
               perturb=False,
               epoch=None,
               **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # direction_norms: [B, N, 1]
        # return: pred_rgb: [B, N, 3]

        _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            semantics = torch.empty((B, N, self.num_semantic_classes),
                                    device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b + 1, head:tail],
                                    rays_d[b:b + 1, head:tail],
                                    direction_norms=direction_norms[b:b + 1,
                                                                    head:tail],
                                    bg_color=bg_color,
                                    perturb=perturb,
                                    epoch=epoch,
                                    **kwargs)
                    depth[b:b + 1, head:tail] = results_["depth"]
                    image[b:b + 1, head:tail] = results_["image"]
                    semantics[b:b + 1, head:tail] = results_["semantics"]
                    head += max_ray_batch

            results = {}
            results["depth"] = depth
            results["image"] = image
            results["semantics"] = semantics

        else:
            results = _run(rays_o,
                           rays_d,
                           direction_norms=direction_norms,
                           bg_color=bg_color,
                           perturb=perturb,
                           epoch=epoch,
                           **kwargs)

        return results
