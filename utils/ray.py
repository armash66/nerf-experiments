import torch

def get_rays(H, W, focal, c2w):
    """
    H, W: image height and width
    focal: focal length
    c2w: camera-to-world matrix (4x4)

    Returns:
        rays_o: (H, W, 3) ray origins
        rays_d: (H, W, 3) ray directions
    """

    i, j = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy'
    )

    # Convert pixel coordinates to camera space
    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)  # (H, W, 3)

    # Rotate rays into world space
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)

    # Origin is camera position
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d