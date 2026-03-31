"""
renderer.py — Volume Rendering
Based on Mildenhall et al. 2020 (arxiv: 2003.08934)

Volume rendering composites discrete samples along a ray into
a single pixel color using the rendering equation:

    C(r) = ∑ T_i * alpha_i * c_i

where:
    T_i     = exp(-∑_{j<i} sigma_j * delta_j)   accumulated transmittance
    alpha_i = 1 - exp(-sigma_i * delta_i)        opacity at sample i
    delta_i = t_{i+1} - t_i                      distance between samples
    c_i     = RGB color at sample i

Intuitively:
    - T_i  = how much light survives to reach sample i (starts at 1, decays)
    - alpha = how much light is absorbed at this sample
    - weight = T_i * alpha_i = contribution of sample i to final color
"""

import torch
import torch.nn as nn
from typing import Optional


# ─────────────────────────────────────────────
# POINT SAMPLING
# ─────────────────────────────────────────────

def sample_coarse(
    rays_o:     torch.Tensor,
    rays_d:     torch.Tensor,
    near:       float,
    far:        float,
    n_samples:  int,
    randomize:  bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stratified sampling of points along each ray.
    Divides [near, far] into n_samples bins and samples once per bin.

    Stratified (randomized) sampling is better than uniform because
    it prevents the network from overfitting to fixed sample positions.

    Args:
        rays_o:    (N, 3) ray origins
        rays_d:    (N, 3) ray directions (normalized)
        near, far: depth bounds
        n_samples: number of points per ray
        randomize: perturb sample positions within each bin (True for training)
    Returns:
        pts: (N, n_samples, 3) 3D sample coordinates
        t:   (N, n_samples)   depth values along each ray
    """
    N = rays_o.shape[0]

    # Evenly spaced bin edges in [near, far]
    t = torch.linspace(near, far, n_samples, device=rays_o.device)
    t = t.expand(N, n_samples).clone()  # (N, n_samples)

    if randomize:
        # Stratified: sample uniformly within each bin
        mid   = 0.5 * (t[:, :-1] + t[:, 1:])
        upper = torch.cat([mid, t[:, -1:]], dim=-1)
        lower = torch.cat([t[:, :1],  mid], dim=-1)
        t     = lower + (upper - lower) * torch.rand_like(t)

    # r(t) = o + t * d
    pts = rays_o[:, None, :] + t[:, :, None] * rays_d[:, None, :]
    return pts, t


def sample_fine(
    rays_o:   torch.Tensor,
    rays_d:   torch.Tensor,
    t_coarse: torch.Tensor,
    weights:  torch.Tensor,
    n_fine:   int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hierarchical fine sampling — sample more points where coarse
    weights are high (i.e. where the surface likely is).

    This is the second pass in the full NeRF paper.
    Coarse pass gives rough weight distribution → fine pass
    concentrates samples near actual geometry.

    Args:
        rays_o:   (N, 3)
        rays_d:   (N, 3)
        t_coarse: (N, n_coarse) coarse depth samples
        weights:  (N, n_coarse) weights from coarse volume render
        n_fine:   number of additional fine samples
    Returns:
        pts_combined: (N, n_coarse + n_fine, 3)
        t_combined:   (N, n_coarse + n_fine)
    """
    # Normalize weights to form a PDF
    weights     = weights + 1e-5                          # prevent zero weights
    pdf         = weights / weights.sum(dim=-1, keepdim=True)
    cdf         = torch.cumsum(pdf, dim=-1)
    cdf         = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

    # Sample from CDF using inverse transform sampling
    u           = torch.rand(weights.shape[0], n_fine, device=weights.device)
    u           = u.contiguous()

    # Invert CDF
    inds        = torch.searchsorted(cdf.contiguous(), u, right=True)
    below       = (inds - 1).clamp(min=0)
    above       = inds.clamp(max=cdf.shape[-1] - 1)

    inds_g      = torch.stack([below, above], dim=-1)     # (N, n_fine, 2)
    cdf_g       = torch.gather(cdf,        1, inds_g.view(cdf.shape[0], -1)).view(*inds_g.shape)
    bins_g      = torch.gather(t_coarse,   1, inds_g.view(t_coarse.shape[0], -1)).view(*inds_g.shape)

    denom       = cdf_g[..., 1] - cdf_g[..., 0]
    denom       = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t_fine      = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])

    # Combine coarse + fine samples, sort by depth
    t_combined, _ = torch.sort(torch.cat([t_coarse, t_fine.detach()], dim=-1), dim=-1)
    pts_combined  = rays_o[:, None, :] + t_combined[:, :, None] * rays_d[:, None, :]

    return pts_combined, t_combined


# ─────────────────────────────────────────────
# VOLUME RENDERING
# ─────────────────────────────────────────────

def volume_render(
    rgb:    torch.Tensor,
    sigma:  torch.Tensor,
    t:      torch.Tensor,
    rays_d: torch.Tensor,
    white_background: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Composite samples along each ray into pixel color and depth.

    Args:
        rgb:              (N, n_samples, 3)  predicted colors
        sigma:            (N, n_samples, 1)  predicted densities
        t:                (N, n_samples)     depth values
        rays_d:           (N, 3)             ray directions
        white_background: if True, composite against white background
    Returns:
        dict with keys:
            color_map:    (N, 3)  final rendered color
            depth_map:    (N,)    expected depth
            weights:      (N, n_samples)  per-sample weights (for fine sampling)
            acc_map:      (N,)    accumulated opacity (0=empty, 1=solid)
    """
    # Delta: distance between consecutive samples
    delta = t[:, 1:] - t[:, :-1]                           # (N, n_samples-1)
    # Last segment stretches to infinity
    delta = torch.cat([delta, torch.full_like(delta[:, :1], 1e10)], dim=-1)

    # Scale delta by actual ray length (directions are normalized)
    delta = delta * torch.norm(rays_d[:, None, :], dim=-1)  # (N, n_samples)

    sigma = sigma[..., 0]                                   # (N, n_samples)

    # Alpha: how opaque is each sample
    alpha = 1.0 - torch.exp(-sigma * delta)                 # (N, n_samples)

    # Transmittance: probability light reaches sample i without being absorbed
    # T_i = prod_{j=0}^{i-1} (1 - alpha_j)
    transmittance = torch.cumprod(
        torch.cat([
            torch.ones_like(alpha[:, :1]),      # T_0 = 1 (nothing before first sample)
            1.0 - alpha[:, :-1] + 1e-10,        # small eps for numerical stability
        ], dim=-1),
        dim=-1,
    )                                                       # (N, n_samples)

    # Weight: contribution of each sample to final pixel
    weights = transmittance * alpha                         # (N, n_samples)

    # Final color: weighted sum
    color_map = torch.sum(weights[..., None] * rgb, dim=1) # (N, 3)

    # Accumulated opacity
    acc_map   = weights.sum(dim=-1)                        # (N,)

    # White background composite
    if white_background:
        color_map = color_map + (1.0 - acc_map[..., None])

    # Expected depth
    depth_map = torch.sum(weights * t, dim=-1)             # (N,)

    return {
        "color_map": color_map,
        "depth_map": depth_map,
        "weights":   weights,
        "acc_map":   acc_map,
    }


# ─────────────────────────────────────────────
# FULL RENDER PASS
# ─────────────────────────────────────────────

def render_rays(
    rays_o:           torch.Tensor,
    rays_d:           torch.Tensor,
    model_coarse:     nn.Module,
    pos_enc_xyz:      nn.Module,
    pos_enc_dir:      nn.Module,
    near:             float,
    far:              float,
    n_coarse:         int,
    randomize:        bool = True,
    model_fine:       Optional[nn.Module] = None,
    n_fine:           int = 0,
    white_background: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Full render pipeline for a batch of rays.

    If model_fine is provided, performs two-pass hierarchical sampling:
        1. Coarse pass: uniform stratified sampling
        2. Fine pass:   importance sampling based on coarse weights

    Args:
        rays_o, rays_d:   (N, 3) ray origins and directions
        model_coarse:     coarse NeRF MLP
        pos_enc_xyz:      positional encoding for xyz coords
        pos_enc_dir:      positional encoding for view directions
        near, far:        depth bounds
        n_coarse:         samples per ray for coarse pass
        randomize:        stratified sampling noise (True for training)
        model_fine:       optional fine NeRF MLP
        n_fine:           additional samples for fine pass
        white_background: composite against white background
    Returns:
        dict with coarse and (optionally) fine render outputs
    """
    # ── Coarse pass ────────────────────────────────────────────────

    pts_c, t_c = sample_coarse(rays_o, rays_d, near, far, n_coarse, randomize)
    N, S, _    = pts_c.shape

    # Encode xyz
    pts_enc = pos_enc_xyz(pts_c.reshape(-1, 3))            # (N*S, xyz_enc_dim)

    # Encode view direction — same direction for all points on a ray
    dirs     = rays_d[:, None, :].expand_as(pts_c)        # (N, S, 3)
    dirs_enc = pos_enc_dir(dirs.reshape(-1, 3))            # (N*S, dir_enc_dim)

    # Query coarse model
    rgb_c, sigma_c = model_coarse(pts_enc, dirs_enc)
    rgb_c   = rgb_c.reshape(N, S, 3)
    sigma_c = sigma_c.reshape(N, S, 1)

    # Coarse volume render
    out_coarse = volume_render(rgb_c, sigma_c, t_c, rays_d, white_background)

    if model_fine is None or n_fine == 0:
        return {"coarse": out_coarse}

    # ── Fine pass ──────────────────────────────────────────────────

    pts_f, t_f = sample_fine(rays_o, rays_d, t_c, out_coarse["weights"], n_fine)
    N, S_f, _  = pts_f.shape

    # Encode fine points
    pts_enc_f = pos_enc_xyz(pts_f.reshape(-1, 3))
    dirs_f    = rays_d[:, None, :].expand_as(pts_f)
    dirs_enc_f = pos_enc_dir(dirs_f.reshape(-1, 3))

    # Query fine model
    rgb_f, sigma_f = model_fine(pts_enc_f, dirs_enc_f)
    rgb_f   = rgb_f.reshape(N, S_f, 3)
    sigma_f = sigma_f.reshape(N, S_f, 1)

    # Fine volume render
    out_fine = volume_render(rgb_f, sigma_f, t_f, rays_d, white_background)

    return {
        "coarse": out_coarse,
        "fine":   out_fine,
    }