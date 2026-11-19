"""
Hybrid refinement for the alumina calibration geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from pyFAI.calibrant import CALIBRANT_FACTORY
from pyFAI.detectors import Detector
from pyFAI.geometry import Geometry
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates, median_filter
from scipy.optimize import least_squares, minimize
from skimage.feature import peak_local_max


_GEOMETRY_FIELDS = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")
# The reference (012), (104), and (113) families in pyFAI's alumina reflection table
_FAMILIES = np.array([0, 1, 4])
_FAMILY_SPACINGS_ANGSTROM = (3.47871214, 2.55009722, 2.08460357)
_TRANSLATIONS = np.array([0, 1, 2])
_FIT_PARAMETERS = np.arange(5)
_FIT_SCALE = np.array([0.1, 0.03, 0.03, 0.02, 0.02])
_FIT_LOWER = np.array([0.06, -0.03, -0.03, -0.10, -0.10])
_FIT_UPPER = np.array([0.24, 0.10, 0.10, 0.10, 0.10])
_DOUBLET_GATES_DEG = (0.5, 0.25, 0.1, 0.05, 0.03)
_CENTROID_HALF_WIDTH_DEG = 0.22


@dataclass(frozen=True)
class RefinementPriors:
    """Training translation ranges (m) and mean tilts (rad) for local search."""

    translation_lower: tuple[float, float, float]
    translation_upper: tuple[float, float, float]
    mean_tilts: tuple[float, float]

    def __post_init__(self):
        lower = np.asarray(self.translation_lower, dtype=np.float32)
        upper = np.asarray(self.translation_upper, dtype=np.float32)
        tilts = np.asarray(self.mean_tilts, dtype=np.float32)
        if lower.shape != (3,) or upper.shape != (3,) or tilts.shape != (2,):
            raise ValueError("Refinement needs three translation bounds and two mean tilts")
        if not np.isfinite(np.r_[lower, upper, tilts]).all() or not (lower < upper).all():
            raise ValueError("Refinement priors must be finite with strictly increasing bounds")


@dataclass(frozen=True)
class RefinementResult:
    parameters: np.ndarray
    diagnostics: dict[str, Any]


class RefinementError(RuntimeError):
    """The seed or fitted ring support is insufficient for calibration."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = {} if diagnostics is None else diagnostics


@dataclass(frozen=True)
class _StripSupport:
    attempted: int
    accepted: int
    family_counts: dict[str, int]

    @property
    def sufficient(self) -> bool:
        return self.accepted / max(self.attempted, 1) >= 0.8 and all(
            self.family_counts.get(str(family), 0) >= 10 for family in _FAMILIES
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "strips_attempted": self.attempted,
            "strips_accepted": self.accepted,
            "family_counts": self.family_counts,
        }


def _filtered_signal(image: np.ndarray, mask: np.ndarray, *, dtype: type) -> np.ndarray:
    smooth = median_filter(np.where(mask, 0, image).astype(dtype), size=3)
    signal = np.maximum(smooth - gaussian_filter(smooth, sigma=10), 0)
    signal[mask] = 0
    return signal


def _extract_peaks(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    positive = signal[(~mask) & (signal > 0)]
    if positive.size:
        signal = np.minimum(signal, np.percentile(positive, 99.5))
    allowed = ~binary_dilation(mask, iterations=2)
    return peak_local_max(
        signal,
        min_distance=2,
        threshold_rel=0.15,
        exclude_border=20,
        labels=allowed.astype(np.uint8),
    )


def _fit_doublet_profiles(
    profiles: np.ndarray, offsets: np.ndarray, separations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Locate the primary line in each radial profile with a fixed-separation doublet model.

    The secondary line sits inside the strip window, so an intensity centroid is pulled
    toward it. Each profile is modelled as two Gaussians sharing a width, with the
    secondary offset by the known doublet separation and its amplitude left free.
    """
    half_width = offsets[-1]
    min_sigma = 2 * (offsets[1] - offsets[0])
    centers = np.full(len(profiles), np.nan)
    sigmas = np.full(len(profiles), np.nan)
    for index, (profile, separation) in enumerate(zip(profiles, separations)):
        peak = float(profile.max())
        if not np.isfinite(peak) or peak <= 0:
            continue

        def model(values: np.ndarray) -> np.ndarray:
            center, sigma, primary, secondary = values
            return primary * np.exp(-0.5 * ((offsets - center) / sigma) ** 2) + secondary * np.exp(
                -0.5 * ((offsets - center - separation) / sigma) ** 2
            )

        start = np.array([offsets[profile.argmax()], 2 * min_sigma, peak, 0.5 * peak])
        lower = np.array([-half_width, min_sigma, 0.0, 0.0])
        upper = np.array([half_width, half_width, 10 * peak, 10 * peak])
        fit = least_squares(
            lambda values: model(values) - profile,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            x_scale=np.array([min_sigma, min_sigma, peak, peak]),
        )
        if fit.success:
            centers[index], sigmas[index] = fit.x[:2]
    return centers, sigmas


class HybridRefiner:
    """Refine a neural geometry seed using a frame and its effective mask."""

    def __init__(
        self,
        detector: Detector,
        calibrant_name: str,
        wavelength: float,
        secondary_wavelength: float,
        *,
        priors: RefinementPriors,
    ):
        if (
            not np.isfinite([wavelength, secondary_wavelength]).all()
            or not 0 < wavelength < secondary_wavelength
        ):
            raise ValueError("Refinement needs positive, ordered doublet wavelengths")
        primary = CALIBRANT_FACTORY(calibrant_name)
        primary.wavelength = wavelength
        spacings = np.asarray(primary.dspacing)
        if len(spacings) <= _FAMILIES[-1] or not np.allclose(
            spacings[_FAMILIES], _FAMILY_SPACINGS_ANGSTROM, atol=1e-7, rtol=0
        ):
            raise ValueError(f"Unexpected alumina reflection definition for {calibrant_name!r}")
        secondary = CALIBRANT_FACTORY(calibrant_name)
        secondary.wavelength = secondary_wavelength
        primary_angles = np.asarray(primary.get_2th(), dtype=float)
        secondary_angles = np.asarray(secondary.get_2th(), dtype=float)
        if min(len(primary_angles), len(secondary_angles)) <= _FAMILIES[-1]:
            raise ValueError("Wavelengths must permit all three alumina reflection families")
        self._primary_angles = primary_angles
        self._secondary_angles = secondary_angles
        self._geometry = Geometry(detector=detector, wavelength=wavelength)
        self._priors = priors
        self._template_rays = self._make_template_rays()

    def _angles(self, parameters: np.ndarray, points: np.ndarray) -> np.ndarray:
        return self._geometry.tth(points[:, 0], points[:, 1], param=parameters)

    def _fit(
        self,
        seed: np.ndarray,
        points: np.ndarray,
        targets: np.ndarray,
        free: np.ndarray,
    ) -> np.ndarray:
        parameters = seed.copy()
        if len(points) < max(12, len(free) * 3):
            return parameters

        scale = _FIT_SCALE[free]
        lower, upper = _FIT_LOWER[free] / scale, _FIT_UPPER[free] / scale
        start = np.clip(parameters[free] / scale, lower + 1e-10, upper - 1e-10)

        def residual(values: np.ndarray) -> np.ndarray:
            trial = parameters.copy()
            trial[free] = values * scale
            return (self._angles(trial, points) - targets) / np.deg2rad(0.03)

        fit = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            loss="cauchy",
            f_scale=1.0,
            max_nfev=150,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        parameters[free] = fit.x * scale
        return parameters

    def _assign_and_fit(
        self,
        seed: np.ndarray,
        points: np.ndarray,
        families: np.ndarray,
        gate_deg: float,
        free: np.ndarray,
        *,
        doublet: bool,
    ) -> np.ndarray:
        if not len(points):
            return seed
        expected = self._primary_angles[families]
        if doublet:
            expected = np.r_[expected, self._secondary_angles[families]]
        separation = np.abs(self._angles(seed, points)[:, None] - expected)
        nearest = separation.argmin(axis=1)
        keep = separation[np.arange(len(points)), nearest] < np.deg2rad(gate_deg)
        return self._fit(seed, points[keep], expected[nearest[keep]], free)

    def _fit_doublet(
        self,
        seed: np.ndarray,
        points: np.ndarray,
        gates_deg: tuple[float, ...],
    ) -> np.ndarray:
        parameters = seed
        for gate in gates_deg:
            parameters = self._assign_and_fit(
                parameters, points, _FAMILIES, gate, _TRANSLATIONS, doublet=True
            )
        for _ in range(2):
            parameters = self._assign_and_fit(
                parameters, points, _FAMILIES, 0.03, _FIT_PARAMETERS, doublet=True
            )
        return parameters

    def _strip_centroids(
        self,
        signal: np.ndarray,
        mask: np.ndarray,
        parameters: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, _StripSupport]:
        if not len(points):
            return np.empty((0, 2)), np.empty(0), _StripSupport(0, 0, {})

        geometry = Geometry(
            **dict(zip(_GEOMETRY_FIELDS, parameters)),
            detector=self._geometry.detector,
            wavelength=self._geometry.wavelength,
        )
        rows, cols = points.T
        angles = geometry.tth(rows, cols)
        azimuth = geometry.chi(rows, cols)
        distance = np.abs(angles[:, None] - self._primary_angles[_FAMILIES])
        nearest = distance.argmin(axis=1)
        family = _FAMILIES[nearest]
        separation = distance[np.arange(len(points)), nearest]
        sectors = 180
        sector = np.floor((azimuth + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
        order = np.lexsort((separation, sector, family))
        seen = set()
        chosen = []
        for index in order:
            key = (family[index], sector[index])
            if separation[index] < np.deg2rad(0.35) and key not in seen:
                seen.add(key)
                chosen.append(index)
        if not chosen:
            return np.empty((0, 2)), np.empty(0), _StripSupport(0, 0, {})

        centers = points[chosen]
        family = family[chosen]
        rows, cols = centers.T
        dr = geometry.tth(rows + 0.5, cols) - geometry.tth(rows - 0.5, cols)
        dc = geometry.tth(rows, cols + 0.5) - geometry.tth(rows, cols - 0.5)
        gradient = np.sqrt(dr * dr + dc * dc)
        
        usable = (gradient > 0) & np.isfinite(gradient)
        centers, family = centers[usable], family[usable]
        dr, dc, gradient = dr[usable], dc[usable], gradient[usable]
        if not len(centers):
            return np.empty((0, 2)), np.empty(0), _StripSupport(len(chosen), 0, {})
        normal = np.c_[dr, dc] / gradient[:, None]
        tangent = np.c_[-normal[:, 1], normal[:, 0]]
        target = self._primary_angles[family]
        shift = (target - geometry.tth(centers[:, 0], centers[:, 1])) / gradient
        middle = centers + normal * shift[:, None]
        offsets = np.linspace(-_CENTROID_HALF_WIDTH_DEG, _CENTROID_HALF_WIDTH_DEG, 121) * np.pi / 180
        lateral = np.array([-2.0, 0.0, 2.0])
        radial_pixels = offsets[None, :, None, None] / gradient[:, None, None, None]
        coords = (
            middle[:, None, None, :]
            + normal[:, None, None, :] * radial_pixels
            + tangent[:, None, None, :] * lateral[None, None, :, None]
        )
        indices = [coords[..., 0], coords[..., 1]]
        invalid = map_coordinates(mask, indices, order=0, mode="constant", cval=True)
        sampled = map_coordinates(signal, indices, order=1, mode="constant", cval=0).mean(axis=2)

        edge = max(5, len(offsets) // 8)
        left = np.median(sampled[:, :edge], axis=1)
        right = np.median(sampled[:, -edge:], axis=1)
        baseline = left[:, None] + (
            (right - left)[:, None] * (offsets - offsets[0]) / (offsets[-1] - offsets[0])
        )
        intensity = np.maximum(sampled - baseline, 0)
        total = intensity.sum(axis=1)
        noise = np.std(np.c_[sampled[:, :edge], sampled[:, -edge:]], axis=1)
        half_width = offsets[-1]
        center, sigma = _fit_doublet_profiles(
            sampled - baseline,
            offsets,
            self._secondary_angles[family] - self._primary_angles[family],
        )
        keep = (
            (~invalid.any(axis=(1, 2)))
            & (intensity.max(axis=1) > 5 * np.maximum(noise, 1e-5))
            & (total > 1e-6)
            & (np.abs(center) < 0.5 * half_width)
            & (sigma < 0.7 * half_width)
        )
        centroids = middle + normal * (center / gradient)[:, None]
        counts = {str(f): int(np.count_nonzero(family[keep] == f)) for f in _FAMILIES}
        return centroids[keep], target[keep], _StripSupport(len(chosen), int(keep.sum()), counts)

    def _centroid_passes(
        self,
        seed: np.ndarray,
        signal: np.ndarray,
        mask: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, list[_StripSupport]]:
        parameters = seed
        support = []
        for _ in range(2):
            centroids, targets, strips = self._strip_centroids(signal, mask, parameters, points)
            parameters = self._fit(parameters, centroids, targets, _FIT_PARAMETERS)
            support.append(strips)
        return parameters, support

    def _make_template_rays(self) -> np.ndarray:
        expected = np.c_[self._primary_angles[_FAMILIES], self._secondary_angles[_FAMILIES]]
        theta = expected[..., None]
        azimuth = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        return np.stack(
            [
                np.sin(theta) * np.cos(azimuth),
                np.sin(theta) * np.sin(azimuth),
                np.broadcast_to(np.cos(theta), (*expected.shape, azimuth.size)),
            ],
            axis=-1,
        )

    def _template_score(self, parameters: np.ndarray, image: np.ndarray, mask: np.ndarray) -> float:
        rays = self._template_rays @ self._geometry.rotation_matrix(parameters)
        projected = rays * (parameters[0] / rays[..., 2])[..., None]
        pixel_sizes = (self._geometry.detector.pixel1, self._geometry.detector.pixel2)
        rings = (projected[..., :2] + parameters[1:3]) / pixel_sizes - 0.5
        coords = rings.reshape(-1, 2).T
        rows, cols = coords
        valid = (
            (rows >= 2) & (rows < image.shape[0] - 2)
            & (cols >= 2) & (cols < image.shape[1] - 2)
        )
        valid &= ~map_coordinates(mask, coords, order=0, mode="constant", cval=True)
        counts = valid.reshape(len(_FAMILIES), 2, -1).sum(axis=(1, 2))
        if (counts < 50).any():
            return -1e10
        signal = map_coordinates(image, coords, order=1, mode="constant", cval=0)
        signal[~valid] = 0
        mean_per_ring = signal.reshape(len(_FAMILIES), 2, -1).sum(axis=(1, 2)) / counts
        return float(np.log1p(mean_per_ring).sum())

    def _rescue(
        self,
        seed: np.ndarray,
        signal: np.ndarray,
        mask: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not len(points):
            raise RefinementError("Local intensity search has no detected peaks to refine")

        training_lower = np.asarray(self._priors.translation_lower, dtype=np.float32)
        training_upper = np.asarray(self._priors.translation_upper, dtype=np.float32)
        radius = 0.1 * (training_upper - training_lower)
        lower = np.maximum(training_lower, seed[:3] - radius)
        upper = np.minimum(training_upper, seed[:3] + radius)
        span = upper - lower
        if (span <= 0).any():
            raise RefinementError("Local intensity search window lies outside the training bounds")

        clean = np.sqrt(np.minimum(signal, np.percentile(signal[~mask], 99.9)))
        positive = clean[(~mask) & (clean > 0)]
        if not positive.size:
            raise RefinementError("Local intensity search has no usable signal")
        clean /= max(float(np.percentile(positive, 95)), 1e-12)
        images = [gaussian_filter(clean, sigma=sigma) for sigma in (4.0, 2.0, 1.2)]

        def vector(translations: np.ndarray) -> np.ndarray:
            return np.r_[translations, self._priors.mean_tilts, 0.0]

        axes = [np.linspace(a, b, 11) for a, b in zip(lower, upper)]
        coarse = []
        for coordinates in product(*axes):
            parameters = vector(coordinates)
            coarse.append((self._template_score(parameters, images[0], mask), parameters))
        coarse.sort(key=lambda candidate: candidate[0], reverse=True)

        starts = [vector(seed[:3])]
        for _, point in coarse:
            if all(np.linalg.norm((point[:3] - prior[:3]) / span) > 0.2 for prior in starts):
                starts.append(point)
            if len(starts) == 8:
                break

        candidates = []
        for start in starts:
            parameters = start.copy()
            for image in images:

                def objective(values: np.ndarray) -> float:
                    return -self._template_score(vector(lower + values * span), image, mask)

                optimum = minimize(
                    objective,
                    np.clip((parameters[:3] - lower) / span, 0, 1),
                    method="L-BFGS-B",
                    bounds=[(0, 1)] * 3,
                    options={"maxiter": 100, "ftol": 1e-10, "gtol": 1e-6, "eps": 1e-4},
                )
                parameters[:3] = lower + optimum.x * span
            score = self._template_score(parameters, images[-1], mask)
            candidates.append((score, parameters, "template"))
            fitted = self._fit_doublet(parameters, points, (0.4, 0.2, 0.1, 0.05, 0.03))
            if ((fitted[:3] >= lower) & (fitted[:3] <= upper)).all():
                score = self._template_score(fitted, images[-1], mask)
                candidates.append((score, fitted, "template_then_points"))

        score, selected, method = max(candidates, key=lambda candidate: candidate[0])
        if score <= -1e10:
            raise RefinementError("Local intensity search found insufficient visible ring coverage")
        return selected, {"method": method, "score": score, "candidates": len(candidates)}

    def refine(self, image: np.ndarray, mask: np.ndarray, seed: np.ndarray) -> RefinementResult:
        """Return a supported geometry or raise with the observed strip counts."""
        image = np.asarray(image)
        mask = np.asarray(mask, dtype=bool)
        if image.ndim != 2 or mask.shape != image.shape:
            raise RefinementError(
                f"Refinement needs a 2D frame and matching mask, got {image.shape} and {mask.shape}"
            )
        seed = np.array(seed, dtype=float, copy=True)
        if seed.shape != (6,) or not np.isfinite(seed[:5]).all() or seed[0] <= 0:
            raise RefinementError("Refinement needs six seed parameters with a finite, positive distance")
        seed[5] = 0.0

        signal = _filtered_signal(image, mask, dtype=np.float32)
        points = _extract_peaks(signal, mask).astype(float)
        # Strips subtract their own linear baseline; high-pass filtering would distort the doublet
        centroid_signal = np.where(mask, 0, image).astype(np.float64)
        parameters = seed
        for _ in range(3):
            parameters = self._assign_and_fit(
                parameters, points, _FAMILIES[:1], 0.8, _TRANSLATIONS, doublet=False
            )
        parameters = self._fit_doublet(parameters, points, _DOUBLET_GATES_DEG)
        parameters, support = self._centroid_passes(parameters, centroid_signal, mask, points)
        supported = all(strips.sufficient for strips in support)
        diagnostics = {
            "detected_peaks": len(points),
            "initial_centroid_passes": [strips.as_dict() for strips in support],
            "rescue_attempted": not supported,
        }
        if not supported:
            try:
                parameters, diagnostics["rescue"] = self._rescue(seed, signal, mask, points)
                parameters, support = self._centroid_passes(parameters, centroid_signal, mask, points)
            except RefinementError as exc:
                diagnostics["rescue_failure"] = str(exc)
        diagnostics["centroid_passes"] = [strips.as_dict() for strips in support]
        if not all(strips.sufficient for strips in support) or not np.isfinite(parameters).all():
            raise RefinementError(
                f"Hybrid calibration failed the ring-support checks: {diagnostics}", diagnostics
            )
        return RefinementResult(parameters, diagnostics)
