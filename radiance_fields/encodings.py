import logging

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import json

import third_party.tcnn_modules as tcnn

logger = logging.getLogger()


class XYZ_Encoder(nn.Module):
    encoder_type = "XYZ_Encoder"
    """Encode XYZ coordinates or directions to a vector."""

    def __init__(self, n_input_dims):
        super().__init__()
        self.n_input_dims = n_input_dims

    @property
    def n_output_dims(self) -> int:
        raise NotImplementedError


class SHEncoder(XYZ_Encoder):
    encoder_type = "SHEncoder"
    """Spherical harmonic encoding

    Args:
        levels: Number of spherical harmonic levels to encode.
    """

    def __init__(self, n_input_dims: int = 3, levels: int = 4) -> None:
        super().__init__(n_input_dims)
        if levels <= 0 or levels > 4:
            raise ValueError(
                f"Spherical harmonic encoding only supports 1 to 4 levels, requested {levels}"
            )
        self.levels = levels
        self.n_input_dims = n_input_dims
        self.encoding = tcnn.Encoding(
            n_input_dims=n_input_dims,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": levels,
            },
        )

    @property
    def n_output_dims(self) -> int:
        return self.levels**2

    @torch.no_grad()
    def forward(self, in_tensor: Tensor) -> Tensor:
        return self.encoding(in_tensor)


class SinusoidalEncoder(XYZ_Encoder):
    encoder_type = "SinusoidalEncoder"
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(
        self,
        n_input_dims: int = 3,
        min_deg: int = 0,
        max_deg: int = 10,
        enable_identity: bool = True,
    ):
        super().__init__(n_input_dims)
        self.n_input_dims = n_input_dims
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.enable_identity = enable_identity
        self.register_buffer(
            "scales", Tensor([2**i for i in range(min_deg, max_deg + 1)])
        )

    @property
    def n_output_dims(self) -> int:
        return (
            int(self.enable_identity) + (self.max_deg - self.min_deg + 1) * 2
        ) * self.n_input_dims

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., n_input_dims]
        Returns:
            encoded: [..., n_output_dims]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[..., None, :] * self.scales[:, None]),
            list(x.shape[:-1])
            + [(self.max_deg - self.min_deg + 1) * self.n_input_dims],
        )
        encoded = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
        if self.enable_identity:
            encoded = torch.cat([x] + [encoded], dim=-1)
        return encoded


class HashEncoder(XYZ_Encoder):
    encoder_type = "HashEncoder"

    def __init__(
        self,
        n_input_dims: int = 3,
        n_levels: int = 16,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        log2_hashmap_size: int = 19,
        n_features_per_level: int = 2,
        dtype=torch.float32,
        verbose: bool = True,
    ) -> None:
        super().__init__(n_input_dims)
        self.n_input_dims = n_input_dims
        self.num_levels = n_levels
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.n_features_per_level = n_features_per_level
        self.num_parameters = 2**log2_hashmap_size * n_features_per_level * n_levels

        self.growth_factor = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        )
        self.encoding_config = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": self.growth_factor,
            "interpolation": "linear",
        }
        self.tcnn_encoding = tcnn.Encoding(
            n_input_dims=n_input_dims,
            encoding_config=self.encoding_config,
            dtype=dtype,
        )
        self.num_parameters = self.tcnn_encoding.params.shape
        if verbose:
            logger.info(f"TCNN encoding config: \n {json.dumps(self.encoding_config)}")
            logger.info(
                f"TCNN encoding number of params: {self.tcnn_encoding.params.numel() / 1e6}M \n"
                f"dtype: {self.tcnn_encoding.params.dtype} \n"
            )

    @property
    def n_output_dims(self) -> int:
        return self.tcnn_encoding.n_output_dims

    def forward(self, in_tensor: Tensor) -> Tensor:
        return self.tcnn_encoding(in_tensor)


def build_xyz_encoder_from_cfg(xyz_encoder_cfg, verbose=True) -> XYZ_Encoder:
    if xyz_encoder_cfg.type == "HashEncoder":
        return HashEncoder(
            n_input_dims=xyz_encoder_cfg.n_input_dims,
            n_levels=xyz_encoder_cfg.n_levels,
            n_features_per_level=xyz_encoder_cfg.n_features_per_level,
            base_resolution=xyz_encoder_cfg.base_resolution,
            max_resolution=xyz_encoder_cfg.max_resolution,
            log2_hashmap_size=xyz_encoder_cfg.log2_hashmap_size,
            verbose=verbose,
        )
    elif xyz_encoder_cfg.type == "SHEncoder":
        return SHEncoder(
            n_input_dims=xyz_encoder_cfg.n_input_dims,
            levels=xyz_encoder_cfg.levels,
        )
    elif xyz_encoder_cfg.type == "SinusoidalEncoder":
        return SinusoidalEncoder(
            n_input_dims=xyz_encoder_cfg.n_input_dims,
            min_deg=xyz_encoder_cfg.min_deg,
            max_deg=xyz_encoder_cfg.max_deg,
            enable_identity=xyz_encoder_cfg.enable_identity,
        )
    else:
        raise NotImplementedError(f"Unknown nerf encoder type: {xyz_encoder_cfg.type}")
