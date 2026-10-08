#!/usr/bin/env python3
"""Diagnostic-only local3050 A/B of the zero-shift polar center controller."""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np

from alignimg import _engine

if __package__:
    from tools import polar_resource_validation as integration
else:
    import polar_resource_validation as integration

fast, perf, baseline, mirror = (
    integration.fast,
    integration.perf,
    integration.baseline,
    integration.mirror,
)
PARTICLE_BATCH = 64
LEGACY_COMMIT = "4a51d41e10ef39bf8c8f09bb1c8a932316e8f4c5"
LEGACY_ENGINE_SHA256 = (
    "2650cf0c7781284bf554243096c3418f15a5ea06587c2a5c870aceb80688361f"
)
LEGACY_HELPER_SHA256 = (
    "4282947070d7f660f03f09fde118e0ce034bdfc30e46b31141a32e909e556fd7"
)


def legacy_center_update(
    candidate_values: dict[str, np.ndarray], shifts: list[tuple[float, float]]
) -> None:
    for reference_index, (shift_y, shift_x) in enumerate(shifts):
        selected = candidate_values["reference_index"] == reference_index
        candidate_values["shift_y_px"][selected] += np.float32(shift_y)
        candidate_values["shift_x_px"][selected] += np.float32(shift_x)
        if "_mstep_reference_index" in candidate_values:
            selected_mstep = (
                candidate_values["_mstep_reference_index"] == reference_index
            )
            candidate_values["_mstep_shift_y_px"][selected_mstep] += np.float32(shift_y)
            candidate_values["_mstep_shift_x_px"][selected_mstep] += np.float32(shift_x)
    if "_polar_center_y_px" in candidate_values:
        angle = np.deg2rad(candidate_values["angle_deg"].astype(np.float64))
        shift_y = candidate_values["shift_y_px"].astype(np.float64)
        shift_x = candidate_values["shift_x_px"].astype(np.float64)
        candidate_values["_polar_center_y_px"][:] = (
            -np.sin(angle) * shift_x - np.cos(angle) * shift_y
        ).astype(np.float32)
        candidate_values["_polar_center_x_px"][:] = (
            -np.cos(angle) * shift_x + np.sin(angle) * shift_y
        ).astype(np.float32)


def legacy_helper_sha256():
    # Pin the historical function's AST, allowing only its local name to differ.
    node = ast.parse(inspect.getsource(legacy_center_update)).body[0]
    node.name = "_apply_reference_center_shifts"
    # Python 3.13 omits empty fields by default; retain Python 3.12's format.
    options = (
        {"show_empty": True}
        if "show_empty" in inspect.signature(ast.dump).parameters
        else {}
    )
    return hashlib.sha256(ast.dump(node, **options).encode()).hexdigest()


@contextmanager
def center_controller(variant, traces, iterations):
    current = _engine._apply_reference_center_shifts
    controller = current if variant == "current" else legacy_center_update

    def apply(values, shifts):
        if len(traces) >= iterations:
            return controller(values, shifts)
        # Host arrays only. Trace the warm-up, never the timed repeats.
        centers = ("_polar_center_y_px", "_polar_center_x_px")
        names = ("angle_deg", "shift_y_px", "shift_x_px") + centers
        before = {key: values[key].copy() for key in names}
        zero = np.all(np.asarray(shifts) == 0, axis=1)[values["reference_index"]]
        controller(values, shifts)
        traces.append(
            {
                "phase": "warmup",
                "iteration": len(traces) + 1,
                "reference_center_shifts": shifts,
                "before_hashes": {k: fast._array_hash(v) for k, v in before.items()},
                "after_hashes": {k: fast._array_hash(values[k]) for k in names},
                "zero_shift_particle_count": int(np.count_nonzero(zero)),
                "zero_shift_center_max_delta_px": max(
                    float(
                        np.max(
                            np.abs(values[k][zero].astype(float) - before[k][zero]),
                            initial=0,
                        )
                    )
                    for k in centers
                ),
            }
        )

    with patch.object(_engine, "_apply_reference_center_shifts", apply):
        yield


def compare_results(expected, actual):
    """Keep the frozen keyset and thresholds; mismatch is diagnostic evidence."""
    subset = {name: actual[name] for name in expected}
    error = None
    try:
        perf.compare_arrays(expected, subset)
    except AssertionError as failure:
        error = str(failure)
    fields = {}
    for name, reference in expected.items():
        value = subset[name]
        assert reference.shape == value.shape, f"{name}: shape changed"
        assert np.isfinite(reference).all() and np.isfinite(value).all(), (
            f"{name}: non-finite"
        )
        entry = {"exact": bool(np.array_equal(reference, value))}
        try:
            perf.compare_arrays({name: reference}, {name: value})
            entry["within_frozen_tolerance"] = True
        except AssertionError:
            entry["within_frozen_tolerance"] = False
        if reference.dtype.kind not in "biu":
            delta = value.astype(float) - reference.astype(float)
            if name == "angle_deg":
                delta = (delta + 180) % 360 - 180
            entry.update(
                zip(
                    ("median_abs", "p95_abs", "maximum_abs"),
                    np.quantile(np.abs(delta), [0.5, 0.95, 1]).tolist(),
                )
            )
            if name in ("references", "class_averages"):
                entry["correlations"] = [
                    baseline.correlation(a, b) for a, b in zip(reference, value)
                ]
        fields[name] = entry
    return {"within_frozen_tolerance": error is None, "error": error, "fields": fields}


def conclusion(current, legacy):
    return {
        (False, True): "legacy_roundtrip_recovers_frozen_baseline",
        (False, False): "inconclusive_neither_matches_baseline",
        (True, True): "mismatch_not_reproduced",
        (True, False): "current_only_matches_baseline",
    }[current, legacy]


def result_path(output, variant):
    return output.with_name(f"{output.stem}.local3050.{variant}.result.npz")


def run_experiment(engine, output, report):
    assert legacy_helper_sha256() == LEGACY_HELPER_SHA256, "legacy helper changed"
    report["runtime"] = integration.runtime_info(engine)
    report["environment"] = mirror.environment_info(engine, 0)
    old, expected, hashes = integration.preflight(mirror.ROOT)
    report.update(inputs=hashes, baseline=old)
    images, _ = baseline.load_stack(mirror.ROOT / "data/local/test_align.mrcs")
    references, _ = baseline.load_stack(mirror.ROOT / "data/local/mu_aligned_mean.mrc")
    assert images.shape == (3050, 100, 100) and references.shape == (1, 100, 100)
    case = fast._case_values(images, references)
    config = mirror.ai.AlignmentConfig(**old["config"])
    assert config.max_iterations == 3 and config.batch_size == 512
    assert config.search_strategy == "polar_hard" and not config.mirror_search
    results = {}
    for variant in ("current", "legacy"):
        print(
            f"RUN {variant}: local3050; actual polar batch={PARTICLE_BATCH}; warm-up + 3 repeats + 1 profile",
            flush=True,
        )
        pending = {"planning_records": [], "execution_records": [], "center_trace": []}
        report["variants"][variant] = pending
        mirror.write_report(output, report)
        with integration.controlled_polar_batch(
            PARTICLE_BATCH, pending["planning_records"]
        ):
            with center_controller(
                variant, pending["center_trace"], config.max_iterations
            ):
                entry = fast.run_case(
                    "local3050",
                    case,
                    config=config,
                    backend=engine,
                    measured_repeats=3,
                    profile_execution=True,
                    output=output,
                    variant=f"center-{variant}",
                    result_label=variant,
                    execution_records=pending["execution_records"],
                )
        pending.update(entry)
        pending["resource_checks"] = integration.resource_checks(pending, engine)
        records = pending["execution_records"]
        pending["traced_warmup_matches_primary"] = records[0]["result_hashes"] == next(
            r["result_hashes"] for r in records if r["phase"] == "unprofiled"
        )
        with np.load(pending["result"], allow_pickle=False) as saved:
            results[variant] = {name: saved[name].copy() for name in saved.files}
        pending["vs_frozen_baseline"] = compare_results(expected, results[variant])
        mirror.write_report(output, report)
    assert (
        report["variants"]["current"]["input_sha256"]
        == report["variants"]["legacy"]["input_sha256"]
    )
    assert (
        report["variants"]["current"]["config"]
        == report["variants"]["legacy"]["config"]
        == old["config"]
    )
    report["current_vs_legacy"] = compare_results(
        {name: results["legacy"][name] for name in expected}, results["current"]
    )
    report["reference_history_max_abs_by_iteration"] = np.max(
        np.abs(
            results["current"]["reference_history"].astype(float)
            - results["legacy"]["reference_history"].astype(float)
        ),
        axis=(1, 2, 3),
    ).tolist()
    report["reference_history_index_zero"] = "initial references"
    report["conclusion"] = conclusion(
        *(
            report["variants"][v]["vs_frozen_baseline"]["within_frozen_tolerance"]
            for v in ("current", "legacy")
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "cupy"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (
        args.output,
        *(result_path(args.output, v) for v in ("current", "legacy")),
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    report = {
        "schema": "alignimg.polar-center-ab.v1",
        "task": "T6 center A/B",
        "status": "running",
        "diagnostic_only": True,
        "started_utc": mirror.utc_now(),
        "source": mirror.source_manifest(),
        "backend": args.backend,
        "actual_polar_batch": PARTICLE_BATCH,
        "particle_storage_policy": "streaming",
        "legacy_helper": {
            "commit": LEGACY_COMMIT,
            "engine_sha256": LEGACY_ENGINE_SHA256,
            "normalized_function_ast_sha256": LEGACY_HELPER_SHA256,
        },
        "variants": {},
        "limits": [
            "Only the center-update helper changes between A/B; both use the current engine and controlled streaming batch.",
            "Frozen 2.3.0 used an older runtime/batch; failure of legacy mode alone cannot disprove a contribution from the no-op change.",
            "Completed diagnostics do not pass T6, change thresholds or authorize reverting the no-op fix.",
            "Warm-up trace is excluded from measured times; timings are descriptive, not the 3050 release performance gate.",
        ],
    }
    original = _engine._apply_reference_center_shifts
    started = time.perf_counter()
    mirror.write_report(args.output, report)
    try:
        run_experiment(args.backend, args.output, report)
        report["status"] = "completed"
    except Exception as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    finally:
        report["controller_restored"] = (
            _engine._apply_reference_center_shifts is original
        )
        report["wall_seconds"] = time.perf_counter() - started
        mirror.write_report(args.output, report)
    print(
        f"DIAGNOSTIC {report['status'].upper()}: {args.output}; conclusion={report.get('conclusion', 'not_reached')}",
        flush=True,
    )
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
