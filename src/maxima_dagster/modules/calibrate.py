"""Masked Swin initialization and hybrid refinement for Maxima XRD."""

from __future__ import annotations

import hashlib
import math
from os import PathLike
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyFAI.detectors import Eiger2CdTe_1M, detector_factory
from pyFAI.geometry import Geometry
from pyFAI.io.ponifile import PoniFile
from transformers import SwinConfig, SwinModel

from . import calibration_recipe as recipe
from .masking import effective_mask
from .refinement import HybridRefiner, RefinementPriors


HC_KEV_M = 1.2398419843320026e-9
DEFAULT_WAVELENGTH_M = recipe.PRIMARY_WAVELENGTH_M
GEOMETRY_PARAMETERS = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")


def resolve_wavelength(*, energy: float | None, wavelength: float | None) -> float:
    if wavelength is not None and energy is not None:
        raise ValueError("Provide either `energy` or `wavelength`, not both.")
    if wavelength is not None:
        if not math.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("`wavelength` must be > 0 and finite.")
        return wavelength
    if energy is not None:
        if not math.isfinite(energy) or energy <= 0:
            raise ValueError("`energy` must be > 0 and finite.")
        return HC_KEV_M / energy
    return DEFAULT_WAVELENGTH_M


class MaximaSwin(nn.Module):
    """Mean-pool Swin tokens before the checkpoint's regression head."""

    def __init__(self, backbone: SwinModel, head: nn.Module):
        super().__init__()
        self.swin = backbone
        self.head = head
        self.pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.swin(pixel_values=pixel_values).last_hidden_state
        return self.head(self.pooler(tokens.transpose(1, 2)).flatten(1))


def _load_checkpoint(stream: BinaryIO) -> dict[str, torch.Tensor]:
    stream.seek(0)
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != recipe.CHECKPOINT_SHA256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {recipe.CHECKPOINT_FILENAME}: "
            f"expected {recipe.CHECKPOINT_SHA256}, got {digest}."
        )
    stream.seek(0)
    return torch.load(stream, map_location="cpu", weights_only=True)


def _preprocess(image: np.ndarray, mask: np.ndarray, image_size: int) -> torch.Tensor:
    """Apply training-time XRD preprocessing using the shared exclusion mask."""
    image = np.log1p(np.where(mask, 0, image).astype(np.float32, copy=False))
    valid_pixels = image[~mask]
    if valid_pixels.size:
        low, high = np.percentile(valid_pixels, (0.1, 99.9))
        if high > low:
            image = np.clip((image - low) / (high - low), 0, 1)
        else:
            image.fill(0)
    image[mask] = 0

    height, width = image.shape
    side = max(height, width)
    tensor = torch.from_numpy(image.astype(np.float32, copy=False))
    tensor = tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    tensor = F.pad(tensor, (0, side - width, 0, side - height))
    tensor = F.interpolate(
        tensor, (image_size, image_size), mode="bilinear", align_corners=False, antialias=True
    )
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (tensor - mean) / std


class MaximaCalibrator:
    """Load the frozen MAXIMA checkpoint and refine geometry in detector coordinates."""

    def __init__(
        self,
        model_src: str | PathLike[str] | BinaryIO,
        calibrant: str = recipe.CALIBRANT_NAME,
        detector: str = recipe.DETECTOR_NAME,
        energy: float | None = None,
        wavelength: float | None = None,
        image_size: int = recipe.IMAGE_SIZE,
        backbone: str = recipe.BACKBONE_NAME,
        hidden_dim: int = recipe.BACKBONE_FEATURES,
        device: str | None = None,
    ):
        self.wavelength = resolve_wavelength(energy=energy, wavelength=wavelength)
        self.detector = detector_factory(detector)
        if calibrant.casefold() != recipe.CALIBRANT_NAME.casefold() or not isinstance(
            self.detector, Eiger2CdTe_1M
        ):
            raise ValueError(
                f"Checkpoint {recipe.CHECKPOINT_VERSION} requires "
                f"{recipe.CALIBRANT_NAME} and {recipe.DETECTOR_NAME}."
            )
        if (image_size, backbone, hidden_dim) != (
            recipe.IMAGE_SIZE, recipe.BACKBONE_NAME, recipe.BACKBONE_FEATURES
        ):
            raise ValueError(f"Architecture must match checkpoint {recipe.CHECKPOINT_VERSION}.")
        self.image_size = image_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        secondary_wavelength = self.wavelength * (
            recipe.SECONDARY_WAVELENGTH_M / recipe.PRIMARY_WAVELENGTH_M
        )
        self.refiner = HybridRefiner(
            self.detector,
            recipe.CALIBRANT_NAME,
            self.wavelength,
            secondary_wavelength,
            priors=RefinementPriors(
                translation_lower=recipe.TRANSLATION_LOWER,
                translation_upper=recipe.TRANSLATION_UPPER,
                mean_tilts=recipe.MEAN_TILTS,
            ),
        )
        self.last_refinement_diagnostics: dict | None = None
        self.model = self._build_model()
        self._load_weights(model_src)
        self.model.to(self.device).eval()

    def _build_model(self) -> MaximaSwin:
        config = SwinConfig(
            image_size=self.image_size,
            patch_size=4,
            num_channels=3,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            drop_path_rate=0.1,
            hidden_act="gelu",
            use_absolute_embeddings=False,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
        )
        head = nn.Sequential(
            nn.Linear(recipe.BACKBONE_FEATURES, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 6),
        )
        return MaximaSwin(SwinModel(config), head)

    def _load_weights(self, model_src: str | PathLike[str] | BinaryIO) -> None:
        if isinstance(model_src, (str, PathLike)):
            with open(model_src, "rb") as stream:
                state = _load_checkpoint(stream)
        else:
            state = _load_checkpoint(model_src)
        self.model.load_state_dict(state, strict=True)

    def _infer(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        tensor = _preprocess(image, mask, self.image_size).to(self.device)
        with torch.inference_mode():
            normalized = self.model(tensor)[0, :5]
            centers = normalized.new_tensor(recipe.PARAMETER_CENTERS)
            scales = normalized.new_tensor(recipe.PARAMETER_SCALES)
            physical = (normalized * scales + centers).cpu().numpy()
        return np.append(physical.astype(float), 0.0)

    def calibrate(
        self,
        image: np.ndarray,
        output_path: str | PathLike[str] | None = None,
        *,
        factory_mask: np.ndarray | None = None,
    ) -> Geometry:
        """Infer and refine one frame, writing only geometry that passed support checks."""
        self.last_refinement_diagnostics = None
        image = np.asarray(image)
        mask = effective_mask(image, factory_mask)
        seed = self._infer(image, mask)
        refinement = self.refiner.refine(image.astype(np.float32, copy=False), mask, seed)
        self.last_refinement_diagnostics = refinement.diagnostics
        geometry = Geometry(
            **dict(zip(GEOMETRY_PARAMETERS, map(float, refinement.parameters), strict=True)),
            detector=self.detector,
            wavelength=self.wavelength,
        )
        if output_path is not None:
            with Path(output_path).open("w", encoding="utf-8") as stream:
                PoniFile(geometry).write(stream)
        return geometry


def calibrate_image(
    image_path: str | PathLike[str],
    model_path: str | PathLike[str],
    output_path: str | PathLike[str] | None = None,
    calibrant: str = recipe.CALIBRANT_NAME,
    detector: str = recipe.DETECTOR_NAME,
    energy: float | None = None,
    wavelength: float | None = None,
    image_size: int = recipe.IMAGE_SIZE,
    backbone: str = recipe.BACKBONE_NAME,
    hidden_dim: int = recipe.BACKBONE_FEATURES,
    device: str | None = None,
    *,
    factory_mask: np.ndarray | None = None,
) -> Geometry:
    """Calibrate a Fabio-readable frame from a local file."""
    import fabio

    with fabio.open(str(image_path)) as image:
        calibrator = MaximaCalibrator(
            model_src=model_path,
            calibrant=calibrant,
            detector=detector,
            energy=energy,
            wavelength=wavelength,
            image_size=image_size,
            backbone=backbone,
            hidden_dim=hidden_dim,
            device=device,
        )
        return calibrator.calibrate(image.data, output_path, factory_mask=factory_mask)


__all__ = ["MaximaCalibrator", "calibrate_image"]
