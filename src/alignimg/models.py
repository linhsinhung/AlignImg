"""Public data contracts for the AlignImg 2.x alignment engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AlignmentConfig:
    """Configuration shared by reference-free, global, and refine workflows."""

    max_iterations: int = 10
    top_l: int = 8
    angle_samples: int = 256
    proposal_angles_per_reference: int | None = None
    translation_range: float = 4.0
    translation_step: float = 1.0
    search_strategy: str = "proposal"
    candidate_scoring: str = "fourier"
    score_model: str = "fourier_ncc"
    reference_update: str = "fourier"
    coarse_angle_step: float = 6.0
    coarse_shift_step: float = 1.0
    local_angle_range: float = 15.0
    local_shift_range: float = 3.0
    adaptive_fraction: float = 0.999
    oversampling_order: int = 1
    max_adaptive_cells: int | None = None
    rescue_uncertain_particles: bool = False
    rescue_normalized_entropy_threshold: float | None = None
    rescue_map_posterior_threshold: float | None = None
    rescue_max_fraction: float = 0.05
    rescue_min_score_improvement: float = 0.0
    temperature_start: float = 0.08
    temperature_end: float = 0.02
    temperature_anneal_iterations: int | None = None
    mask_radius: float | None = None
    mask_soft_edge: float = 4.0
    min_frequency: float = 0.0
    max_frequency: float = 0.35
    mirror_search: bool = False
    random_seed: int = 0
    robust_weighting: bool = False
    keep_fraction: float = 0.80
    weight_temperature: float = 0.08
    pose_angle_sigma: float = 5.0
    pose_shift_sigma: float = 2.0
    halfset_diagnostics: bool = True
    center_references: bool = True
    lowpass_sigma: float = 0.0
    store_history: bool = True
    apply_final_pose_to_raw: bool = False
    batch_size: int | None = None
    memory_fraction: float = 0.80
    profile_execution: bool = False

    @classmethod
    def preset(cls, name: str) -> "AlignmentConfig":
        """Return a documented workflow preset without hiding its parameters."""
        presets = {
            "global_balanced": dict(
                search_strategy="proposal",
                top_l=8,
                proposal_angles_per_reference=6,
                temperature_start=0.08,
                temperature_end=0.05,
            ),
            "global_accurate": dict(
                search_strategy="proposal",
                top_l=8,
                proposal_angles_per_reference=8,
                temperature_start=0.08,
                temperature_end=0.05,
            ),
            "reference_free": dict(
                search_strategy="proposal",
                top_l=4,
                proposal_angles_per_reference=4,
                temperature_start=0.08,
                temperature_end=0.02,
                temperature_anneal_iterations=10,
            ),
            "refine": dict(
                search_strategy="quadratic_refine",
                max_iterations=5,
                top_l=8,
                local_angle_range=7.0,
                coarse_angle_step=1.0,
                local_shift_range=3.0,
                temperature_start=0.08,
                temperature_end=0.05,
                robust_weighting=True,
                pose_angle_sigma=5.0,
                pose_shift_sigma=2.0,
            ),
        }
        key = str(name).strip().lower()
        if key not in presets:
            raise ValueError(
                f"unknown alignment preset {name!r}; choose from {sorted(presets)}"
            )
        return cls(**presets[key])

    def normalized(self, *, workflow: str | None = None) -> "AlignmentConfig":
        values = asdict(self)
        values["max_iterations"] = int(values["max_iterations"])
        values["top_l"] = int(values["top_l"])
        values["angle_samples"] = int(values["angle_samples"])
        values["search_strategy"] = str(values["search_strategy"]).strip().lower()
        if values["search_strategy"] not in {
            "proposal",
            "adaptive_posterior",
            "quadratic_refine",
        }:
            raise ValueError(
                "search_strategy must be 'proposal', 'adaptive_posterior', "
                "or 'quadratic_refine'."
            )
        values["candidate_scoring"] = str(values["candidate_scoring"]).strip().lower()
        if values["candidate_scoring"] not in {"raster", "fourier"}:
            raise ValueError("candidate_scoring must be 'raster' or 'fourier'.")
        if (
            values["search_strategy"] == "quadratic_refine"
            and values["candidate_scoring"] != "fourier"
        ):
            raise ValueError("quadratic_refine requires candidate_scoring='fourier'.")
        if (
            values["search_strategy"] == "quadratic_refine"
            and values["rescue_uncertain_particles"]
        ):
            raise ValueError("quadratic_refine does not support rescue searches.")
        values["score_model"] = str(values["score_model"]).strip().lower()
        if values["score_model"] not in {
            "fourier_ncc",
            "whitened_fourier_ncc",
        }:
            raise ValueError(
                "score_model must be 'fourier_ncc' or 'whitened_fourier_ncc'."
            )
        values["reference_update"] = str(values["reference_update"]).strip().lower()
        if values["reference_update"] not in {"spatial", "fourier"}:
            raise ValueError("reference_update must be 'spatial' or 'fourier'.")
        if (
            values["search_strategy"] == "quadratic_refine"
            and values["reference_update"] != "fourier"
        ):
            raise ValueError("quadratic_refine requires reference_update='fourier'.")
        if values["max_iterations"] < 1:
            raise ValueError("max_iterations must be positive.")
        if not 1 <= values["top_l"] <= 16:
            raise ValueError("top_l must be in the range 1..16.")
        if values["angle_samples"] < 4:
            raise ValueError("angle_samples must be at least 4.")
        if values["proposal_angles_per_reference"] is not None:
            values["proposal_angles_per_reference"] = int(
                values["proposal_angles_per_reference"]
            )
            if (
                not 1
                <= values["proposal_angles_per_reference"]
                <= values["angle_samples"]
            ):
                raise ValueError(
                    "proposal_angles_per_reference must be in the range 1..angle_samples."
                )
        if float(values["translation_range"]) < 0:
            raise ValueError("translation_range must be non-negative.")
        if float(values["translation_step"]) <= 0:
            raise ValueError("translation_step must be positive.")
        for name in ("coarse_angle_step", "coarse_shift_step"):
            if float(values[name]) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("local_angle_range", "local_shift_range"):
            if float(values[name]) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not 0 < float(values["adaptive_fraction"]) <= 1:
            raise ValueError("adaptive_fraction must be in (0, 1].")
        values["oversampling_order"] = int(values["oversampling_order"])
        if values["oversampling_order"] < 0:
            raise ValueError("oversampling_order must be non-negative.")
        if values["max_adaptive_cells"] is not None:
            values["max_adaptive_cells"] = int(values["max_adaptive_cells"])
            if values["max_adaptive_cells"] < 1:
                raise ValueError("max_adaptive_cells must be positive when provided.")
        if values["rescue_normalized_entropy_threshold"] is not None:
            if not 0 <= float(values["rescue_normalized_entropy_threshold"]) <= 1:
                raise ValueError(
                    "rescue_normalized_entropy_threshold must be in [0, 1]."
                )
        if values["rescue_map_posterior_threshold"] is not None:
            if not 0 <= float(values["rescue_map_posterior_threshold"]) <= 1:
                raise ValueError("rescue_map_posterior_threshold must be in [0, 1].")
        if not 0 < float(values["rescue_max_fraction"]) <= 1:
            raise ValueError("rescue_max_fraction must be in (0, 1].")
        if float(values["rescue_min_score_improvement"]) < 0:
            raise ValueError("rescue_min_score_improvement must be non-negative.")
        for name in ("temperature_start", "temperature_end", "weight_temperature"):
            if float(values[name]) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("pose_angle_sigma", "pose_shift_sigma"):
            if float(values[name]) <= 0:
                raise ValueError(f"{name} must be positive.")
        if values["temperature_anneal_iterations"] is not None:
            values["temperature_anneal_iterations"] = int(
                values["temperature_anneal_iterations"]
            )
            if (
                not 1
                <= values["temperature_anneal_iterations"]
                <= values["max_iterations"]
            ):
                raise ValueError(
                    "temperature_anneal_iterations must be in the range "
                    "1..max_iterations."
                )
        if not 0 < float(values["keep_fraction"]) <= 1:
            raise ValueError("keep_fraction must be in (0, 1].")
        if not 0 < float(values["memory_fraction"]) <= 1:
            raise ValueError("memory_fraction must be in (0, 1].")
        if (
            not 0
            <= float(values["min_frequency"])
            < float(values["max_frequency"])
            <= 0.5
        ):
            raise ValueError(
                "frequencies must satisfy 0 <= min_frequency < max_frequency <= 0.5."
            )
        if values["batch_size"] is not None and int(values["batch_size"]) < 1:
            raise ValueError("batch_size must be positive when provided.")
        # Profiling is observational and must not change default-refine selection.
        if workflow == "refine" and replace(self, profile_execution=False) == AlignmentConfig():
            values["max_iterations"] = 5
            values["robust_weighting"] = True
        return AlignmentConfig(**values)


@dataclass(frozen=True)
class PoseSet:
    """Canonical transforms mapping input particles into reference coordinates."""

    angle_deg: np.ndarray
    shift_y_px: np.ndarray
    shift_x_px: np.ndarray
    mirror: np.ndarray

    def __post_init__(self) -> None:
        angle = np.asarray(self.angle_deg, dtype=np.float32)
        sy = np.asarray(self.shift_y_px, dtype=np.float32)
        sx = np.asarray(self.shift_x_px, dtype=np.float32)
        mirror = np.asarray(self.mirror, dtype=np.bool_)
        if not (angle.ndim == sy.ndim == sx.ndim == mirror.ndim == 1):
            raise ValueError("PoseSet fields must be one-dimensional.")
        if not (len(angle) == len(sy) == len(sx) == len(mirror)):
            raise ValueError("PoseSet fields must have the same length.")
        if not np.all(np.isfinite(np.stack((angle, sy, sx), axis=1))):
            raise ValueError("PoseSet numeric fields must be finite.")
        object.__setattr__(
            self, "angle_deg", ((angle + 180.0) % 360.0 - 180.0).astype(np.float32)
        )
        object.__setattr__(self, "shift_y_px", sy)
        object.__setattr__(self, "shift_x_px", sx)
        object.__setattr__(self, "mirror", mirror)

    def __len__(self) -> int:
        return len(self.angle_deg)

    @classmethod
    def identity(cls, count: int) -> "PoseSet":
        return cls(
            np.zeros(count, dtype=np.float32),
            np.zeros(count, dtype=np.float32),
            np.zeros(count, dtype=np.float32),
            np.zeros(count, dtype=np.bool_),
        )


@dataclass(frozen=True)
class CandidateSet:
    """Top-L pose/reference candidates retained for each particle."""

    reference_index: np.ndarray
    angle_deg: np.ndarray
    shift_y_px: np.ndarray
    shift_x_px: np.ndarray
    mirror: np.ndarray
    score: np.ndarray
    posterior: np.ndarray


@dataclass
class AlignmentResult:
    """Structured result returned by all AlignImg 2.x workflows."""

    references: np.ndarray
    poses: PoseSet
    reference_assignments: np.ndarray
    responsibilities: np.ndarray
    candidates: CandidateSet
    inlier_weights: np.ndarray
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    reference_history: list[np.ndarray] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _class_average_values: np.ndarray | None = field(default=None, repr=False)
    _pose_entropy_values: np.ndarray | None = field(default=None, repr=False)
    _map_posterior_values: np.ndarray | None = field(default=None, repr=False)

    @property
    def class_averages(self) -> np.ndarray:
        """Final output averages, optionally reconstructed from raw particles."""
        if self._class_average_values is not None:
            return np.asarray(self._class_average_values, dtype=np.float32)
        return np.asarray(self.references, dtype=np.float32)

    @property
    def pose_entropy(self) -> np.ndarray:
        """Entropy of each particle's full inferred posterior when available."""
        if self._pose_entropy_values is not None:
            return np.asarray(self._pose_entropy_values, dtype=np.float32)
        posterior = np.asarray(self.candidates.posterior, dtype=np.float64)
        terms = np.where(
            posterior > 0.0,
            posterior * np.log(np.maximum(posterior, 1e-30)),
            0.0,
        )
        return (-np.sum(terms, axis=1)).astype(np.float32)

    @property
    def map_posterior(self) -> np.ndarray:
        """Maximum probability in each particle's full posterior when available."""
        if self._map_posterior_values is not None:
            return np.asarray(self._map_posterior_values, dtype=np.float32)
        return np.max(self.candidates.posterior, axis=1).astype(np.float32)
