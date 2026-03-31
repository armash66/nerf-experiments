"""
model.py — NeRF MLP
Based on Mildenhall et al. 2020 (arxiv: 2003.08934)

The network takes a positionally encoded 3D coordinate (and optionally
a viewing direction) and outputs:
    - RGB color   (3 values, range [0, 1])
    - Volume density sigma (1 value, range [0, inf))

Architecture (from paper):
    - 8 fully connected layers, 256 hidden units, ReLU activations
    - Skip connection: re-inject input at layer 5
    - Density head branches off at layer 8 (no view dependence)
    - Color head takes density features + encoded view direction
"""

import torch
import torch.nn as nn


class NeRFMLP(nn.Module):
    """
    Full NeRF MLP with view-dependent color.

    Two-stage architecture:
        Stage 1 (geometry):  encoded_xyz → features + sigma
        Stage 2 (color):     features + encoded_dir → RGB

    This separation is intentional — density (geometry) should not
    depend on viewing direction, only color should. A surface exists
    regardless of where you look at it from.

    Args:
        xyz_dim:    output dim of positional encoding for xyz coords
        dir_dim:    output dim of positional encoding for view direction
        hidden_dim: width of hidden layers (256 in original paper)
        num_layers: total layers in geometry network (8 in original paper)
        skip_layer: which layer to inject the skip connection (5 in paper)
    """

    def __init__(
        self,
        xyz_dim:    int,
        dir_dim:    int,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_layer: int = 5,
    ):
        super().__init__()

        self.skip_layer = skip_layer
        self.num_layers = num_layers

        # ── Stage 1: Geometry network ──────────────────────────────
        # Processes encoded xyz → dense feature vector + sigma
        # Skip connection at skip_layer re-injects the original input
        # to prevent vanishing gradients in deep network

        self.xyz_layers = nn.ModuleList()
        in_dim = xyz_dim
        for i in range(num_layers):
            # At skip layer, input is concatenated with original xyz encoding
            if i == skip_layer:
                in_dim = hidden_dim + xyz_dim
            self.xyz_layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        # Density output — branched from geometry features
        # No activation here — ReLU applied externally to keep sigma >= 0
        self.sigma_head = nn.Linear(hidden_dim, 1)

        # Feature projection before color head
        self.feature_proj = nn.Linear(hidden_dim, hidden_dim)

        # ── Stage 2: Color network ─────────────────────────────────
        # Takes geometry features + view direction → RGB
        # Shallow (1 layer) since color variation is simpler than geometry

        self.color_layer = nn.Linear(hidden_dim + dir_dim, hidden_dim // 2)
        self.color_head  = nn.Linear(hidden_dim // 2, 3)

        # Activations
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()   # smoother than ReLU for density

    def forward(
        self,
        xyz_encoded: torch.Tensor,
        dir_encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz_encoded: (..., xyz_dim)  positionally encoded 3D coords
            dir_encoded: (..., dir_dim)  positionally encoded view directions
        Returns:
            rgb:   (..., 3)  color values in [0, 1]
            sigma: (..., 1)  volume density >= 0
        """

        # ── Stage 1: Geometry ──────────────────────────────────────
        h = xyz_encoded
        for i, layer in enumerate(self.xyz_layers):
            if i == self.skip_layer:
                h = torch.cat([h, xyz_encoded], dim=-1)  # skip connection
            h = self.relu(layer(h))

        # Density — use softplus for smoother gradients near zero
        sigma = self.softplus(self.sigma_head(h))        # (..., 1)

        # Project features for color stage
        features = self.feature_proj(h)                  # (..., hidden_dim)

        # ── Stage 2: Color ─────────────────────────────────────────
        # Concatenate geometry features with view direction
        color_input = torch.cat([features, dir_encoded], dim=-1)
        color_feat  = self.relu(self.color_layer(color_input))
        rgb         = self.sigmoid(self.color_head(color_feat))  # (..., 3)

        return rgb, sigma


class TinyNeRFMLP(nn.Module):
    """
    Simplified NeRF MLP without view-dependent color.
    Faster to train — good for verifying your pipeline works
    before switching to the full model.

    No skip connection, no view direction input.
    Use this first, switch to NeRFMLP once pipeline is confirmed working.

    Args:
        xyz_dim:    output dim of positional encoding for xyz
        hidden_dim: width of hidden layers
        num_layers: number of hidden layers
    """

    def __init__(
        self,
        xyz_dim:    int,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()

        layers = []
        in_dim = xyz_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim  = hidden_dim

        self.backbone     = nn.Sequential(*layers)
        self.sigma_head   = nn.Sequential(nn.Linear(hidden_dim, 1),  nn.Softplus())
        self.color_head   = nn.Sequential(nn.Linear(hidden_dim, 3),  nn.Sigmoid())

    def forward(
        self,
        xyz_encoded: torch.Tensor,
        dir_encoded: torch.Tensor = None,   # ignored, kept for API compatibility
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz_encoded: (..., xyz_dim)
            dir_encoded: ignored (kept for API compatibility with NeRFMLP)
        Returns:
            rgb:   (..., 3)
            sigma: (..., 1)
        """
        features = self.backbone(xyz_encoded)
        rgb      = self.color_head(features)
        sigma    = self.sigma_head(features)
        return rgb, sigma