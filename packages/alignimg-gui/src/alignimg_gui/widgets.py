"""Reusable Qt widgets for paths and alignment result inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .artifacts import load_result_bundle


class PathField(QWidget):
    pathChanged = pyqtSignal(str)

    def __init__(
        self,
        *,
        directory: bool = False,
        file_filter: str = "All files (*)",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.directory = directory
        self.file_filter = file_filter
        self.edit = QLineEdit()
        self.edit.editingFinished.connect(lambda: self.pathChanged.emit(self.text()))
        self.button = QPushButton("Browse…")
        self.button.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:
        self.edit.setText(value)
        self.pathChanged.emit(self.text())

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self.edit.setEnabled(enabled)
        self.button.setEnabled(enabled)

    def _browse(self) -> None:
        start = self.text() or str(Path.cwd())
        if self.directory:
            value = QFileDialog.getExistingDirectory(self, "Select directory", start)
        else:
            value, _ = QFileDialog.getOpenFileName(
                self, "Select file", start, self.file_filter
            )
        if value:
            self.setText(value)


def image_pixmap(
    image: np.ndarray,
    *,
    size: int,
    limits: tuple[float, float] | None = None,
) -> QPixmap:
    values = np.asarray(image, dtype=np.float32)
    if limits is None:
        low, high = np.percentile(values, (1.0, 99.0))
    else:
        low, high = limits
    scaled = np.clip((values - low) / max(float(high - low), 1e-8), 0.0, 1.0)
    pixels = np.ascontiguousarray(np.round(scaled * 255.0).astype(np.uint8))
    qimage = QImage(
        pixels.data,
        pixels.shape[1],
        pixels.shape[0],
        pixels.strides[0],
        QImage.Format.Format_Grayscale8,
    ).copy()
    return QPixmap.fromImage(qimage).scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class ClassTile(QFrame):
    selected = pyqtSignal(int)

    def __init__(
        self,
        index: int,
        image: np.ndarray,
        caption: str,
        *,
        limits: tuple[float, float] | None,
    ) -> None:
        super().__init__()
        self.index = index
        self.setFrameShape(QFrame.Shape.StyledPanel)
        picture = QLabel()
        picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        picture.setPixmap(image_pixmap(image, size=128, limits=limits))
        label = QLabel(caption)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(picture)
        layout.addWidget(label)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.selected.emit(self.index)
        super().mousePressEvent(event)


class ClassResultsWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.history: np.ndarray | None = None
        self.assignments: np.ndarray | None = None
        self.responsibilities: np.ndarray | None = None
        self.inlier_weights: np.ndarray | None = None
        self.report: dict[str, Any] = {}
        self.history_labels: list[str] = []
        self.selected_class = 0

        self.contrast = QComboBox()
        self.contrast.addItem("Per-class contrast", "per_class")
        self.contrast.addItem("Global contrast", "global")
        self.iteration = QSlider(Qt.Orientation.Horizontal)
        self.iteration.setEnabled(False)
        self.iteration_label = QLabel("No result loaded")
        self.preview = QLabel("Run an alignment or load a report")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(300)
        self.preview_info = QLabel()
        self.preview_info.setAlignment(Qt.AlignmentFlag.AlignCenter)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Contrast:"))
        controls.addWidget(self.contrast)
        controls.addSpacing(12)
        controls.addWidget(self.iteration_label)
        controls.addWidget(self.iteration, 1)

        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        self.gallery_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.gallery_widget)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.preview)
        layout.addWidget(self.preview_info)
        layout.addWidget(scroll, 1)
        self.iteration.valueChanged.connect(self.refresh)
        self.contrast.currentIndexChanged.connect(self.refresh)

    def load_report(self, path: str | Path) -> None:
        report, arrays = load_result_bundle(path)
        self.report = report
        self.history = np.asarray(arrays["reference_history"], dtype=np.float32)
        self.assignments = np.asarray(arrays["assignments"], dtype=np.int32)
        self.responsibilities = np.asarray(arrays["responsibilities"], dtype=np.float32)
        self.inlier_weights = np.asarray(arrays["inlier_weights"], dtype=np.float32)
        self.history_labels = self._history_labels(len(self.history))
        self.selected_class = 0
        self.iteration.blockSignals(True)
        self.iteration.setRange(0, len(self.history) - 1)
        self.iteration.setValue(len(self.history) - 1)
        self.iteration.blockSignals(False)
        self.iteration.setEnabled(len(self.history) > 1)
        self.refresh()

    def _history_labels(self, count: int) -> list[str]:
        stages = self.report.get("stages", {})
        global_count = len(stages.get("global", {}).get("diagnostics", []))
        refine_count = len(stages.get("refine", {}).get("diagnostics", []))
        initial = "Input references" if not global_count and refine_count else "Initial"
        labels = [initial]
        labels.extend(
            f"Global {index}/{global_count}" for index in range(1, global_count + 1)
        )
        labels.extend(
            f"Refine {index}/{refine_count}" for index in range(1, refine_count + 1)
        )
        if len(labels) != count:
            return [f"Reference history {index}/{count - 1}" for index in range(count)]
        return labels

    def _limits(self, references: np.ndarray) -> tuple[float, float] | None:
        if self.contrast.currentData() == "global":
            low, high = np.percentile(references, (1.0, 99.0))
            return float(low), float(high)
        return None

    def _frc_resolution(self, class_index: int) -> float | None:
        diagnostics = self.report.get("diagnostics", [])
        pixel_size = self.report.get("input", {}).get("pixel_size_angstrom")
        if not diagnostics or not pixel_size:
            return None
        cutoffs = diagnostics[-1].get("frc_0143_stable_cutoff_cyc_per_px")
        weights = diagnostics[-1].get("halfset_effective_weight")
        if cutoffs is None or class_index >= len(cutoffs):
            return None
        if weights is not None:
            half = np.asarray(weights, dtype=np.float32)
            if half.ndim == 2 and np.min(half[:, class_index]) < 10.0:
                return None
        cutoff = float(cutoffs[class_index])
        return float(pixel_size) / cutoff if cutoff > 0 else None

    def _effective_weights(self) -> np.ndarray | None:
        if self.responsibilities is None or self.inlier_weights is None:
            return None
        return np.sum(self.responsibilities * self.inlier_weights[:, None], axis=0)

    def refresh(self) -> None:
        if self.history is None:
            return
        index = self.iteration.value()
        references = self.history[index]
        limits = self._limits(references)
        self.iteration_label.setText(self.history_labels[index])
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        counts = (
            np.bincount(self.assignments, minlength=len(references))
            if self.assignments is not None
            else np.zeros(len(references), dtype=int)
        )
        effective = self._effective_weights()
        columns = max(1, min(5, len(references)))
        for class_index, reference in enumerate(references):
            caption = f"K{class_index}  N={int(counts[class_index])}"
            if effective is not None:
                caption += f"  Neff={effective[class_index]:.1f}"
            resolution = self._frc_resolution(class_index)
            if resolution is not None:
                caption += f"\nFRC={resolution:.1f} Å"
            tile = ClassTile(class_index, reference, caption, limits=limits)
            tile.selected.connect(self.select_class)
            self.gallery_layout.addWidget(
                tile, class_index // columns, class_index % columns
            )
        self.select_class(min(self.selected_class, len(references) - 1))

    def select_class(self, class_index: int) -> None:
        if self.history is None:
            return
        self.selected_class = class_index
        references = self.history[self.iteration.value()]
        limits = self._limits(references)
        self.preview.setPixmap(
            image_pixmap(references[class_index], size=384, limits=limits)
        )
        resolution = self._frc_resolution(class_index)
        text = f"Class K{class_index}"
        if resolution is not None:
            text += f" · stable FRC {resolution:.2f} Å"
        self.preview_info.setText(text)


class DiagnosticsWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ncc = pg.PlotWidget(title="Mean expected Fourier NCC")
        self.change = pg.PlotWidget(title="Reference relative change")
        self.confidence = pg.PlotWidget(title="Mean max responsibility")
        self.occupancy = pg.PlotWidget(title="Final hard occupancy")
        for plot in (self.ncc, self.change, self.confidence, self.occupancy):
            plot.showGrid(x=True, y=True, alpha=0.25)
        layout = QGridLayout(self)
        layout.addWidget(self.ncc, 0, 0)
        layout.addWidget(self.change, 0, 1)
        layout.addWidget(self.confidence, 1, 0)
        layout.addWidget(self.occupancy, 1, 1)

    def load_report(self, path: str | Path) -> None:
        report, arrays = load_result_bundle(path)
        diagnostics = report.get("diagnostics", [])
        x = np.arange(1, len(diagnostics) + 1)
        self.ncc.clear()
        self.change.clear()
        self.confidence.clear()
        self.occupancy.clear()
        if diagnostics:
            self.ncc.plot(
                x,
                [item.get("mean_expected_fourier_ncc", np.nan) for item in diagnostics],
                pen=pg.mkPen("#4ea1ff", width=2),
                symbol="o",
            )
            self.change.plot(
                x,
                [item.get("reference_relative_change", np.nan) for item in diagnostics],
                pen=pg.mkPen("#f0a44b", width=2),
                symbol="o",
            )
            self.confidence.plot(
                x,
                [item.get("mean_max_responsibility", np.nan) for item in diagnostics],
                pen=pg.mkPen("#65c18c", width=2),
                symbol="o",
            )
            global_count = len(
                report.get("stages", {}).get("global", {}).get("diagnostics", [])
            )
            if global_count and global_count < len(diagnostics):
                for plot in (self.ncc, self.change, self.confidence):
                    plot.addItem(
                        pg.InfiniteLine(
                            pos=global_count + 0.5,
                            angle=90,
                            pen=pg.mkPen(
                                "#aaaaaa", width=1, style=Qt.PenStyle.DashLine
                            ),
                            label="refine",
                        )
                    )
        assignments = np.asarray(arrays["assignments"], dtype=np.int32)
        reference_count = int(np.asarray(arrays["responsibilities"]).shape[1])
        counts = np.bincount(assignments, minlength=reference_count)
        self.occupancy.addItem(
            pg.BarGraphItem(
                x=np.arange(reference_count),
                height=counts,
                width=0.8,
                brush="#9b7ede",
            )
        )


class SummaryWidget(QPlainTextEdit):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)

    def load_report(self, path: str | Path) -> None:
        report = json.loads(Path(path).read_text())
        metadata = report.get("metadata", {})
        diagnostics = report.get("diagnostics", [])
        rescue = None
        if diagnostics and any("rescue_particle_count" in item for item in diagnostics):
            rescue = {
                "used_total": sum(
                    item.get("rescue_particle_count", 0) for item in diagnostics
                ),
                "accepted_total": sum(
                    item.get("rescue_accepted_count", 0) for item in diagnostics
                ),
                "rejected_total": sum(
                    item.get("rescue_rejected_count", 0) for item in diagnostics
                ),
                "scheduled_trajectory": [
                    item.get("rescue_scheduled_count", 0) for item in diagnostics
                ],
                "mean_score_gain_trajectory": [
                    item.get("mean_rescue_score_gain", 0.0) for item in diagnostics
                ],
            }
        memory_plans = metadata.get("gpu_memory_plans", [])
        execution = {
            key: metadata.get(key)
            for key in (
                "engine",
                "backend",
                "gpu_device",
                "candidate_scoring",
                "score_model",
                "reference_update",
                "class_average_estimator",
                "class_average_transform_backend",
                "class_average_batch_size",
                "search_strategy",
                "fft_policy",
                "gpu_policy",
            )
            if metadata.get(key) is not None
        }
        if metadata.get("gpu_memory_plan_at_completion") is not None:
            execution["gpu_memory_plan_at_completion"] = metadata[
                "gpu_memory_plan_at_completion"
            ]
        if memory_plans:
            execution["gpu_memory_plan_stages"] = [
                {
                    key: plan.get(key)
                    for key in (
                        "stage",
                        "batch_size",
                        "requested_batch_size",
                        "budget_bytes",
                        "free_bytes",
                        "total_bytes",
                    )
                    if plan.get(key) is not None
                }
                for plan in memory_plans
            ]
        compact = {
            "status": report.get("status"),
            "final_stage": report.get("final_stage"),
            "input": report.get("input"),
            "summary": report.get("summary"),
            "stages": {
                name: {
                    "summary": stage.get("summary"),
                    "artifacts": stage.get("artifacts"),
                    "report": stage.get("report"),
                }
                for name, stage in report.get("stages", {}).items()
            },
            "execution": execution,
            "rescue": rescue,
            "environment": report.get("environment"),
            "spec": report.get("spec"),
            "artifacts": report.get("artifacts"),
        }
        self.setPlainText(json.dumps(compact, indent=2, sort_keys=True))
