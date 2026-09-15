"""Serializable pipeline specification shared by the GUI and worker."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from alignimg import AlignmentConfig


MODES = ("reference_free", "reference_based")
BACKENDS = ("auto", "cuda", "cupy", "cpu")
REFINE_POLICIES = ("soft", "fixed", "free", "hard_trust")


@dataclass(frozen=True)
class RunSpec:
    particles: str
    output_directory: str
    run_name: str
    mode: str = "reference_free"
    references: str | None = None
    previous_result: str | None = None
    backend: str = "auto"
    n_components: int = 10
    run_global: bool = True
    refine_enabled: bool = True
    refine_class_policy: str = "soft"
    corrective_trust: float = 0.9
    global_config: dict[str, Any] = field(default_factory=dict)
    refine_config: dict[str, Any] = field(default_factory=dict)

    @property
    def run_directory(self) -> Path:
        return Path(self.output_directory).expanduser() / self.run_name

    def normalized(self) -> "RunSpec":
        mode = str(self.mode).strip().lower()
        backend = str(self.backend).strip().lower()
        policy = str(self.refine_class_policy).strip().lower()
        run_name = str(self.run_name).strip()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}.")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}.")
        if policy not in REFINE_POLICIES:
            raise ValueError(f"refine_class_policy must be one of {REFINE_POLICIES}.")
        if (
            not run_name
            or run_name in {".", ".."}
            or "/" in run_name
            or "\\" in run_name
        ):
            raise ValueError("run_name must be a single non-empty directory name.")
        if int(self.n_components) < 1:
            raise ValueError("n_components must be positive.")
        trust = float(self.corrective_trust)
        if not 0.0 <= trust <= 1.0:
            raise ValueError("corrective_trust must be in [0, 1].")
        if mode == "reference_based" and not self.references:
            raise ValueError("reference-based alignment requires a reference stack.")
        if mode == "reference_free" and not self.run_global:
            raise ValueError("reference-free alignment requires its global RF stage.")
        if not self.run_global and not self.refine_enabled:
            raise ValueError("at least one pipeline stage must be enabled.")
        if not self.run_global and not self.previous_result:
            raise ValueError("skipping global search requires a previous result NPZ.")

        if self.run_global:
            if self.global_config:
                global_config = AlignmentConfig(**self.global_config).normalized(
                    workflow=(
                        "reference_free" if mode == "reference_free" else "global"
                    )
                )
            else:
                preset = (
                    "reference_free" if mode == "reference_free" else "global_balanced"
                )
                global_config = AlignmentConfig.preset(preset).normalized(
                    workflow=(
                        "reference_free" if mode == "reference_free" else "global"
                    )
                )
        else:
            global_config = AlignmentConfig.preset("global_balanced")

        if self.refine_enabled:
            refine_config = (
                AlignmentConfig(**self.refine_config).normalized(workflow="refine")
                if self.refine_config
                else AlignmentConfig.preset("refine").normalized(workflow="refine")
            )
        else:
            refine_config = AlignmentConfig.preset("refine")

        return RunSpec(
            particles=str(Path(self.particles).expanduser()),
            references=(
                str(Path(self.references).expanduser())
                if mode == "reference_based" and self.references
                else None
            ),
            previous_result=(
                str(Path(self.previous_result).expanduser())
                if not self.run_global and self.previous_result
                else None
            ),
            output_directory=str(Path(self.output_directory).expanduser()),
            run_name=run_name,
            mode=mode,
            backend=backend,
            n_components=int(self.n_components),
            run_global=bool(self.run_global),
            refine_enabled=bool(self.refine_enabled),
            refine_class_policy=policy,
            corrective_trust=trust,
            global_config=asdict(global_config),
            refine_config=asdict(refine_config),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalized())

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> "RunSpec":
        return cls(**json.loads(path.read_text())).normalized()


def prepare_run_directory(spec: RunSpec) -> tuple[RunSpec, Path]:
    normalized = spec.normalized()
    run_directory = normalized.run_directory
    run_directory.parent.mkdir(parents=True, exist_ok=True)
    run_directory.mkdir(exist_ok=False)
    normalized.write(run_directory / "spec.json")
    return normalized, run_directory
