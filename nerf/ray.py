import torch

def get_rays(H, W, focal, c2w):
    """
    H, W: image height and width
    focal: focal length
    c2w: camera-to-world matrix (4x4)
    Returns:
        rays_o: (H, W, 3) ray origins
        rays_d: (H, W, 3) ray directions (normalized)
    """
    device = c2w.device  # match device to c2w

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='xy'
    )

    # Convert pixel coordinates to camera space directions
    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,  # flip y: image y-axis points down, camera up
        -torch.ones_like(i)       # looking down -z axis (OpenGL convention)
    ], dim=-1)  # (H, W, 3)

    # Rotate ray directions from camera space to world space
    rays_d = torch.sum(
        dirs[..., None, :] * c2w[:3, :3], dim=-1
    )  # (H, W, 3)

    # Normalize direction vectors
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Ray origins: camera position in world space, broadcast to (H, W, 3)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d