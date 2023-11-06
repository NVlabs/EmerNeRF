from torch import Tensor
import torch.nn as nn
from typing import Optional, Tuple
import torch


class MLP(nn.Module):
    """A simple MLP with skip connections."""

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        num_layers: int = 3,
        hidden_dims: Optional[int] = 256,
        skip_connections: Optional[Tuple[int]] = [0],
    ) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.n_output_dims = out_dims
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(in_dims, out_dims))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    layers.append(nn.Linear(in_dims, hidden_dims))
                elif i in skip_connections:
                    layers.append(nn.Linear(in_dims + hidden_dims, hidden_dims))
                else:
                    layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.Linear(hidden_dims, out_dims))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        input = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                x = torch.cat([x, input], -1)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.functional.relu(x)
        return x
