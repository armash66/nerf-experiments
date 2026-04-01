import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs: int, input_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freqs", freqs)
        self.output_dim = input_dim + 2 * num_freqs * input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = [x]
        for freq in self.freqs:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)