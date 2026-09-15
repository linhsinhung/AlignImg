"""Main AlignImg workbench window."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import sys

from PyQt6.QtCore import QProcess, QTimer, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from alignimg import AlignmentConfig

from .artifacts import inspect_mrc_stack
from .spec import RunSpec, prepare_run_directory
from .widgets import (
    ClassResultsWidget,
    DiagnosticsWidget,
    PathField,
    SummaryWidget,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AlignImg Workbench 0.5")
        self.resize(1440, 920)
        self.process: QProcess | None = None
        self.current_run_directory: Path | None = None
        self.stdout_buffer = ""

        controls = self._build_controls()
        self.results = ClassResultsWidget()
        self.diagnostics = DiagnosticsWidget()
        self.summary = SummaryWidget()
        tabs = QTabWidget()
        tabs.addTab(self.results, "Class averages")
        tabs.addTab(self.diagnostics, "Diagnostics")
        tabs.addTab(self.summary, "Run summary")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(tabs)
        splitter.setSizes([430, 1010])
        self.setCentralWidget(splitter)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        dock = QDockWidget("Run log", self)
        dock.setWidget(self.log)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.statusBar().showMessage("Ready")
        self._mode_changed()

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    def _double(
        self,
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int = 3,
        step: float = 0.1,
    ) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    def _build_controls(self) -> QWidget:
        content = QWidget()
        root = QVBoxLayout(content)

        files = QGroupBox("Input and output")
        file_form = QFormLayout(files)
        self.particles = PathField(
            file_filter="MRC stacks (*.mrcs *.mrc);;All files (*)"
        )
        self.output = PathField(directory=True)
        self.output.setText(str(Path.cwd() / "alignimg-runs"))
        self.run_name = QLineEdit()
        self.run_name.setPlaceholderText("alignimg-YYYYMMDD-HHMMSS")
        self.input_summary = QLabel("Select a particle stack to inspect its shape.")
        self.input_summary.setWordWrap(True)
        ctf_notice = QLabel(
            "Alignment input is expected to be preprocessed and CTF-corrected. "
            "This GUI does not apply CTF correction."
        )
        ctf_notice.setWordWrap(True)
        ctf_notice.setStyleSheet("color: #a05a00;")
        file_form.addRow("Particles", self.particles)
        file_form.addRow("Output directory", self.output)
        file_form.addRow("Run name", self.run_name)
        file_form.addRow("Stack info", self.input_summary)
        file_form.addRow(ctf_notice)
        root.addWidget(files)

        workflow_group = QGroupBox("Alignment pipeline")
        workflow_layout = QVBoxLayout(workflow_group)
        self.mode_tabs = QTabWidget()

        reference_free_page = QWidget()
        reference_free_form = QFormLayout(reference_free_page)
        reference_free_note = QLabel(
            "No reference is available. Bootstrap references, run global RF "
            "alignment, then optionally polish with local refinement."
        )
        reference_free_note.setWordWrap(True)
        self.components = self._spin(1, 1000, 10)
        reference_free_form.addRow(reference_free_note)
        reference_free_form.addRow("Components K", self.components)
        self.mode_tabs.addTab(reference_free_page, "Reference-Free")

        reference_based_page = QWidget()
        reference_based_form = QFormLayout(reference_based_page)
        reference_based_note = QLabel(
            "Use a reference stack with K ≥ 1. Run global single/multi-reference "
            "search, or resume directly from a previous result."
        )
        reference_based_note.setWordWrap(True)
        self.references = PathField(
            file_filter="References (*.mrcs *.mrc);;All files (*)"
        )
        self.reference_start = QComboBox()
        self.reference_start.addItem("Global search", "global")
        self.reference_start.addItem("Refine previous result", "previous")
        self.previous = PathField(file_filter="AlignImg result (*.npz);;All files (*)")
        reference_based_form.addRow(reference_based_note)
        reference_based_form.addRow("Reference stack", self.references)
        reference_based_form.addRow("Start from", self.reference_start)
        reference_based_form.addRow("Previous result", self.previous)
        self.mode_tabs.addTab(reference_based_page, "Reference-Based / MRA")
        workflow_layout.addWidget(self.mode_tabs)

        workflow_form = QFormLayout()
        self.backend = QComboBox()
        for label, value in (
            ("Auto", "auto"),
            ("Native CUDA", "cuda"),
            ("CuPy fallback", "cupy"),
            ("CPU", "cpu"),
        ):
            self.backend.addItem(label, value)
        self.global_iterations = self._spin(1, 1000, 15)
        self.refine_enabled = QCheckBox("Run local refinement")
        self.refine_enabled.setChecked(True)
        self.refine_iterations = self._spin(1, 1000, 2)
        self.refine_search_strategy = QComboBox()
        self.refine_search_strategy.addItem(
            "Continuous quadratic (recommended)", "quadratic_refine"
        )
        self.refine_search_strategy.addItem(
            "Adaptive posterior (2.1 compatibility)", "adaptive_posterior"
        )
        self.refine_policy = QComboBox()
        self.refine_policy.addItem("Soft responsibilities (recommended)", "soft")
        self.refine_policy.addItem("Fixed assignments", "fixed")
        self.refine_policy.addItem("Free reassignment", "free")
        self.refine_policy.addItem("Hard assignments + trust", "hard_trust")
        self.batch_size = self._spin(1, 100000, 512)
        self.memory_fraction = self._double(0.05, 1.0, 0.8, decimals=2, step=0.05)
        self.apply_final_pose_to_raw = QCheckBox(
            "Reconstruct final average from raw particles"
        )
        self.corrective_trust = self._double(0.0, 1.0, 0.9, decimals=2, step=0.05)
        self.reset_preset = QPushButton("Reset pipeline preset")
        workflow_form.addRow("Backend", self.backend)
        workflow_form.addRow("Global iterations", self.global_iterations)
        workflow_form.addRow(self.refine_enabled)
        workflow_form.addRow("Refine iterations", self.refine_iterations)
        workflow_form.addRow("Refine pose search", self.refine_search_strategy)
        workflow_form.addRow("Refine class policy", self.refine_policy)
        workflow_form.addRow("Class-prior trust", self.corrective_trust)
        workflow_form.addRow("Batch size", self.batch_size)
        workflow_form.addRow("VRAM fraction", self.memory_fraction)
        workflow_form.addRow(self.apply_final_pose_to_raw)
        workflow_form.addRow(self.reset_preset)
        workflow_layout.addLayout(workflow_form)
        root.addWidget(workflow_group)

        self.advanced_toggle = QPushButton("Advanced parameters ▸")
        self.advanced_toggle.setCheckable(True)
        advanced = QGroupBox()
        self.advanced_group = advanced
        advanced.setVisible(False)
        advanced_form = QFormLayout(advanced)
        self.search_strategy = QComboBox()
        self.search_strategy.addItem("Polar proposal", "proposal")
        self.search_strategy.addItem("Adaptive posterior", "adaptive_posterior")
        self.candidate_scoring = QComboBox()
        self.candidate_scoring.addItem("Fourier-native", "fourier")
        self.candidate_scoring.addItem("Raster compatibility", "raster")
        self.score_model = QComboBox()
        self.score_model.addItem("Fourier NCC", "fourier_ncc")
        self.score_model.addItem(
            "Empirical-whitened Fourier NCC", "whitened_fourier_ncc"
        )
        self.reference_update = QComboBox()
        self.reference_update.addItem("Fourier M-step", "fourier")
        self.reference_update.addItem("Spatial compatibility", "spatial")
        self.top_l = self._spin(1, 16, 4)
        self.angle_samples = self._spin(4, 4096, 256)
        self.proposal_angles = self._spin(0, 4096, 4)
        self.proposal_angles.setSpecialValueText("Automatic")
        self.translation_range = self._double(0.0, 128.0, 4.0, step=1.0)
        self.translation_step = self._double(0.01, 16.0, 1.0, step=0.25)
        self.coarse_angle_step = self._double(0.01, 180.0, 6.0, step=1.0)
        self.coarse_shift_step = self._double(0.01, 32.0, 1.0, step=0.25)
        self.local_angle_range = self._double(0.0, 180.0, 15.0, step=1.0)
        self.local_shift_range = self._double(0.0, 128.0, 3.0, step=0.5)
        self.adaptive_fraction = self._double(0.001, 1.0, 0.999, decimals=3, step=0.001)
        self.oversampling_order = self._spin(0, 8, 1)
        self.max_adaptive_cells = self._spin(0, 1_000_000, 0)
        self.max_adaptive_cells.setSpecialValueText("No cap")
        self.rescue_uncertain = QCheckBox()
        self.temperature_start = self._double(0.001, 10.0, 0.08)
        self.temperature_end = self._double(0.001, 10.0, 0.02)
        self.anneal_iterations = self._spin(0, 1000, 10)
        self.anneal_iterations.setSpecialValueText("All iterations")
        self.min_frequency = self._double(0.0, 0.499, 0.0)
        self.max_frequency = self._double(0.001, 0.5, 0.35)
        self.mask_radius = self._double(0.0, 4096.0, 0.0, step=1.0)
        self.mask_radius.setSpecialValueText("Automatic")
        self.mask_soft_edge = self._double(0.0, 128.0, 4.0, step=1.0)
        self.lowpass_sigma = self._double(0.0, 128.0, 0.0)
        self.pose_angle_sigma = self._double(0.01, 180.0, 5.0)
        self.pose_shift_sigma = self._double(0.01, 128.0, 2.0)
        self.robust = QCheckBox()
        self.keep_fraction = self._double(0.01, 1.0, 0.8, decimals=2, step=0.05)
        self.weight_temperature = self._double(0.001, 10.0, 0.08)
        self.mirror = QCheckBox()
        self.halfset = QCheckBox()
        self.halfset.setChecked(True)
        self.center = QCheckBox()
        self.center.setChecked(True)
        self.store_history = QCheckBox()
        self.store_history.setChecked(True)
        self.random_seed = self._spin(0, 2**31 - 1, 0)
        for label, widget in (
            ("Global pose search", self.search_strategy),
            ("Candidate scoring", self.candidate_scoring),
            ("Score model", self.score_model),
            ("Reference update", self.reference_update),
            ("Top-L", self.top_l),
            ("Angle samples", self.angle_samples),
            ("Proposal angles/reference", self.proposal_angles),
            ("Translation range (px)", self.translation_range),
            ("Translation step (px)", self.translation_step),
            ("Coarse angle step (deg)", self.coarse_angle_step),
            ("Coarse shift step (px)", self.coarse_shift_step),
            ("Local angle range (deg)", self.local_angle_range),
            ("Local shift range (px)", self.local_shift_range),
            ("Adaptive posterior mass", self.adaptive_fraction),
            ("Oversampling order", self.oversampling_order),
            ("Maximum adaptive cells", self.max_adaptive_cells),
            ("Boundary rescue", self.rescue_uncertain),
            ("Temperature start", self.temperature_start),
            ("Temperature end", self.temperature_end),
            ("Anneal iterations", self.anneal_iterations),
            ("Minimum frequency", self.min_frequency),
            ("Maximum frequency", self.max_frequency),
            ("Mask radius", self.mask_radius),
            ("Mask soft edge", self.mask_soft_edge),
            ("Lowpass sigma", self.lowpass_sigma),
            ("Pose angle sigma", self.pose_angle_sigma),
            ("Pose shift sigma", self.pose_shift_sigma),
            ("Robust weighting", self.robust),
            ("Keep fraction", self.keep_fraction),
            ("Weight temperature", self.weight_temperature),
            ("Mirror search", self.mirror),
            ("Halfset diagnostics", self.halfset),
            ("Center references", self.center),
            ("Store history", self.store_history),
            ("Random seed", self.random_seed),
        ):
            advanced_form.addRow(label, widget)
        root.addWidget(self.advanced_toggle)
        root.addWidget(advanced)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.load_button = QPushButton("Load report…")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.load_button)
        root.addLayout(buttons)
        root.addWidget(self.progress)
        root.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setMinimumWidth(400)

        self.mode_tabs.currentChanged.connect(self._mode_changed)
        self.reference_start.currentIndexChanged.connect(self._reference_start_changed)
        self.refine_enabled.toggled.connect(self._refine_changed)
        self.refine_policy.currentIndexChanged.connect(self._refine_changed)
        self.refine_search_strategy.currentIndexChanged.connect(
            self._refine_strategy_changed
        )
        self.search_strategy.currentIndexChanged.connect(self._search_strategy_changed)
        self.reset_preset.clicked.connect(self._apply_preset)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.particles.pathChanged.connect(self._inspect_particles)
        self.run_button.clicked.connect(self.start_run)
        self.stop_button.clicked.connect(self.stop_run)
        self.load_button.clicked.connect(self.load_report)
        return scroll

    def _toggle_advanced(self, expanded: bool) -> None:
        self.advanced_group.setVisible(expanded)
        self.advanced_toggle.setText(
            "Advanced parameters ▾" if expanded else "Advanced parameters ▸"
        )

    def _inspect_particles(self, path: str) -> None:
        if not path:
            self.input_summary.setText("Select a particle stack to inspect its shape.")
            return
        try:
            info = inspect_mrc_stack(path)
            shape = " × ".join(str(value) for value in info["shape"])
            pixel = info["pixel_size_angstrom"]
            pixel_text = f"{pixel:.4g} Å/px" if pixel else "pixel size unavailable"
            gib = info["file_size_bytes"] / 2**30
            self.input_summary.setText(
                f"{shape} · {info['dtype']} · {pixel_text} · {gib:.3f} GiB"
            )
        except Exception as error:
            self.input_summary.setText(f"Cannot inspect stack: {error}")

    def _mode(self) -> str:
        return (
            "reference_free"
            if self.mode_tabs.currentIndex() == 0
            else "reference_based"
        )

    def _mode_changed(self) -> None:
        self._reference_start_changed()
        self._apply_preset()

    def _reference_start_changed(self) -> None:
        resume = (
            self._mode() == "reference_based"
            and self.reference_start.currentData() == "previous"
        )
        self.previous.setEnabled(resume)
        self.global_iterations.setEnabled(not resume)
        if resume:
            self.refine_enabled.setChecked(True)
            self.refine_enabled.setEnabled(False)
        else:
            self.refine_enabled.setEnabled(True)
        self._refine_changed()

    def _refine_changed(self) -> None:
        enabled = self.refine_enabled.isChecked()
        self.refine_iterations.setEnabled(enabled)
        self.refine_search_strategy.setEnabled(enabled)
        self.refine_policy.setEnabled(enabled)
        self.corrective_trust.setEnabled(
            enabled and self.refine_policy.currentData() in {"soft", "hard_trust"}
        )
        self._search_strategy_changed()

    def _search_strategy_changed(self) -> None:
        global_adaptive = self.search_strategy.currentData() == "adaptive_posterior"
        refine_strategy = self.refine_search_strategy.currentData()
        refine_local = self.refine_enabled.isChecked()
        refine_adaptive = refine_local and refine_strategy == "adaptive_posterior"
        self.proposal_angles.setEnabled(not global_adaptive)
        for widget in (
            self.coarse_angle_step,
            self.local_angle_range,
            self.local_shift_range,
        ):
            widget.setEnabled(global_adaptive or refine_local)
        self.coarse_shift_step.setEnabled(global_adaptive or refine_adaptive)
        for widget in (
            self.adaptive_fraction,
            self.oversampling_order,
            self.max_adaptive_cells,
            self.rescue_uncertain,
        ):
            widget.setEnabled(global_adaptive or refine_adaptive)

    def _refine_strategy_changed(self) -> None:
        if self.refine_search_strategy.currentData() == "quadratic_refine":
            config = AlignmentConfig.preset("refine")
        else:
            config = AlignmentConfig(
                search_strategy="adaptive_posterior",
                local_angle_range=15.0,
                coarse_angle_step=6.0,
                local_shift_range=3.0,
                coarse_shift_step=1.0,
                adaptive_fraction=0.999,
                oversampling_order=1,
            )
        self.coarse_angle_step.setValue(config.coarse_angle_step)
        self.coarse_shift_step.setValue(config.coarse_shift_step)
        self.local_angle_range.setValue(config.local_angle_range)
        self.local_shift_range.setValue(config.local_shift_range)
        self.adaptive_fraction.setValue(config.adaptive_fraction)
        self.oversampling_order.setValue(config.oversampling_order)
        self.max_adaptive_cells.setValue(config.max_adaptive_cells or 0)
        self.rescue_uncertain.setChecked(config.rescue_uncertain_particles)
        self._search_strategy_changed()

    def _apply_preset(self) -> None:
        mode = self._mode()
        preset_name = (
            "reference_free" if mode == "reference_free" else "global_balanced"
        )
        config = AlignmentConfig.preset(preset_name)
        if mode == "reference_free":
            config = replace(
                config, max_iterations=15, temperature_anneal_iterations=10
            )
        self.global_iterations.setValue(config.max_iterations)
        self.refine_enabled.setChecked(True)
        self.refine_iterations.setValue(2)
        refine_config = AlignmentConfig.preset("refine")
        self.refine_search_strategy.setCurrentIndex(
            self.refine_search_strategy.findData(refine_config.search_strategy)
        )
        self.refine_policy.setCurrentIndex(self.refine_policy.findData("soft"))
        self.corrective_trust.setValue(0.9)
        self.apply_final_pose_to_raw.setChecked(False)
        strategy_index = self.search_strategy.findData(config.search_strategy)
        self.search_strategy.setCurrentIndex(strategy_index)
        self.candidate_scoring.setCurrentIndex(
            self.candidate_scoring.findData(config.candidate_scoring)
        )
        self.score_model.setCurrentIndex(self.score_model.findData(config.score_model))
        self.reference_update.setCurrentIndex(
            self.reference_update.findData(config.reference_update)
        )
        self.top_l.setValue(config.top_l)
        self.angle_samples.setValue(config.angle_samples)
        self.proposal_angles.setValue(config.proposal_angles_per_reference or 0)
        self.translation_range.setValue(config.translation_range)
        self.translation_step.setValue(config.translation_step)
        self.coarse_angle_step.setValue(refine_config.coarse_angle_step)
        self.coarse_shift_step.setValue(refine_config.coarse_shift_step)
        self.local_angle_range.setValue(refine_config.local_angle_range)
        self.local_shift_range.setValue(refine_config.local_shift_range)
        self.adaptive_fraction.setValue(config.adaptive_fraction)
        self.oversampling_order.setValue(config.oversampling_order)
        self.max_adaptive_cells.setValue(config.max_adaptive_cells or 0)
        self.rescue_uncertain.setChecked(config.rescue_uncertain_particles)
        self.temperature_start.setValue(config.temperature_start)
        self.temperature_end.setValue(config.temperature_end)
        self.anneal_iterations.setValue(config.temperature_anneal_iterations or 0)
        self.min_frequency.setValue(config.min_frequency)
        self.max_frequency.setValue(config.max_frequency)
        self.mask_radius.setValue(config.mask_radius or 0.0)
        self.mask_soft_edge.setValue(config.mask_soft_edge)
        self.lowpass_sigma.setValue(config.lowpass_sigma)
        self.pose_angle_sigma.setValue(config.pose_angle_sigma)
        self.pose_shift_sigma.setValue(config.pose_shift_sigma)
        self.robust.setChecked(config.robust_weighting)
        self.keep_fraction.setValue(config.keep_fraction)
        self.weight_temperature.setValue(config.weight_temperature)
        self.mirror.setChecked(config.mirror_search)
        self.halfset.setChecked(config.halfset_diagnostics)
        self.center.setChecked(config.center_references)
        self.store_history.setChecked(config.store_history)
        self.random_seed.setValue(config.random_seed)
        self._search_strategy_changed()
        self._reference_start_changed()

    def _global_config(self) -> dict:
        return asdict(
            AlignmentConfig(
                max_iterations=self.global_iterations.value(),
                top_l=self.top_l.value(),
                angle_samples=self.angle_samples.value(),
                proposal_angles_per_reference=(self.proposal_angles.value() or None),
                translation_range=self.translation_range.value(),
                translation_step=self.translation_step.value(),
                search_strategy=self.search_strategy.currentData(),
                candidate_scoring=self.candidate_scoring.currentData(),
                score_model=self.score_model.currentData(),
                reference_update=self.reference_update.currentData(),
                coarse_angle_step=self.coarse_angle_step.value(),
                coarse_shift_step=self.coarse_shift_step.value(),
                local_angle_range=self.local_angle_range.value(),
                local_shift_range=self.local_shift_range.value(),
                adaptive_fraction=self.adaptive_fraction.value(),
                oversampling_order=self.oversampling_order.value(),
                max_adaptive_cells=(self.max_adaptive_cells.value() or None),
                rescue_uncertain_particles=self.rescue_uncertain.isChecked(),
                temperature_start=self.temperature_start.value(),
                temperature_end=self.temperature_end.value(),
                temperature_anneal_iterations=(self.anneal_iterations.value() or None),
                mask_radius=(self.mask_radius.value() or None),
                mask_soft_edge=self.mask_soft_edge.value(),
                min_frequency=self.min_frequency.value(),
                max_frequency=self.max_frequency.value(),
                mirror_search=self.mirror.isChecked(),
                random_seed=self.random_seed.value(),
                robust_weighting=self.robust.isChecked(),
                keep_fraction=self.keep_fraction.value(),
                weight_temperature=self.weight_temperature.value(),
                pose_angle_sigma=self.pose_angle_sigma.value(),
                pose_shift_sigma=self.pose_shift_sigma.value(),
                halfset_diagnostics=self.halfset.isChecked(),
                center_references=self.center.isChecked(),
                lowpass_sigma=self.lowpass_sigma.value(),
                store_history=self.store_history.isChecked(),
                apply_final_pose_to_raw=(
                    self.apply_final_pose_to_raw.isChecked()
                    and not self.refine_enabled.isChecked()
                ),
                batch_size=self.batch_size.value(),
                memory_fraction=self.memory_fraction.value(),
            )
        )

    def _refine_config(self) -> dict:
        return asdict(
            replace(
                AlignmentConfig.preset("refine"),
                max_iterations=self.refine_iterations.value(),
                search_strategy=self.refine_search_strategy.currentData(),
                candidate_scoring=self.candidate_scoring.currentData(),
                score_model=self.score_model.currentData(),
                reference_update=self.reference_update.currentData(),
                coarse_angle_step=self.coarse_angle_step.value(),
                coarse_shift_step=self.coarse_shift_step.value(),
                local_angle_range=self.local_angle_range.value(),
                local_shift_range=self.local_shift_range.value(),
                adaptive_fraction=self.adaptive_fraction.value(),
                oversampling_order=self.oversampling_order.value(),
                max_adaptive_cells=(self.max_adaptive_cells.value() or None),
                rescue_uncertain_particles=self.rescue_uncertain.isChecked(),
                mask_radius=(self.mask_radius.value() or None),
                mask_soft_edge=self.mask_soft_edge.value(),
                min_frequency=self.min_frequency.value(),
                max_frequency=self.max_frequency.value(),
                random_seed=self.random_seed.value(),
                keep_fraction=self.keep_fraction.value(),
                weight_temperature=self.weight_temperature.value(),
                pose_angle_sigma=self.pose_angle_sigma.value(),
                pose_shift_sigma=self.pose_shift_sigma.value(),
                halfset_diagnostics=self.halfset.isChecked(),
                center_references=self.center.isChecked(),
                lowpass_sigma=self.lowpass_sigma.value(),
                store_history=self.store_history.isChecked(),
                apply_final_pose_to_raw=self.apply_final_pose_to_raw.isChecked(),
                batch_size=self.batch_size.value(),
                memory_fraction=self.memory_fraction.value(),
            )
        )

    def _run_spec(self) -> RunSpec:
        name = self.run_name.text().strip()
        if not name:
            name = datetime.now().strftime("alignimg-%Y%m%d-%H%M%S")
            self.run_name.setText(name)
        mode = self._mode()
        run_global = not (
            mode == "reference_based"
            and self.reference_start.currentData() == "previous"
        )
        return RunSpec(
            particles=self.particles.text(),
            references=(
                (self.references.text() or None) if mode == "reference_based" else None
            ),
            previous_result=(self.previous.text() or None) if not run_global else None,
            output_directory=self.output.text(),
            run_name=name,
            mode=mode,
            backend=self.backend.currentData(),
            n_components=self.components.value(),
            run_global=run_global,
            refine_enabled=self.refine_enabled.isChecked(),
            refine_class_policy=self.refine_policy.currentData(),
            corrective_trust=self.corrective_trust.value(),
            global_config=self._global_config(),
            refine_config=self._refine_config(),
        ).normalized()

    def start_run(self) -> None:
        if self.process is not None:
            return
        try:
            spec = self._run_spec()
            for label, value in (
                ("particle stack", spec.particles),
                ("reference stack", spec.references),
                ("previous result", spec.previous_result),
            ):
                if value and not Path(value).is_file():
                    raise ValueError(f"{label} does not exist: {value}")
            spec, run_directory = prepare_run_directory(spec)
        except Exception as error:
            QMessageBox.critical(self, "Cannot start alignment", str(error))
            return

        self.current_run_directory = run_directory
        self.log.clear()
        self.stdout_buffer = ""
        self._append_log(f"Run directory: {run_directory}")
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(
            ["-m", "alignimg_gui.worker", "--spec", str(run_directory / "spec.json")]
        )
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.start()
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.statusBar().showMessage("Alignment running")

    def stop_run(self) -> None:
        if self.process is None:
            return
        self._append_log("Stopping worker…")
        self.process.terminate()
        process = self.process
        QTimer.singleShot(
            3000,
            lambda: (
                process.kill()
                if process.state() != QProcess.ProcessState.NotRunning
                else None
            ),
        )

    def _read_process_output(self) -> None:
        if self.process is None:
            return
        self.stdout_buffer += bytes(self.process.readAllStandardOutput()).decode(
            errors="replace"
        )
        lines = self.stdout_buffer.split("\n")
        self.stdout_buffer = lines.pop()
        for line in lines:
            if not line:
                continue
            try:
                event = json.loads(line)
                self._append_log(event.get("message", line))
            except json.JSONDecodeError:
                self._append_log(line)

    def _process_finished(self, exit_code: int, _status) -> None:
        if self.stdout_buffer:
            self._append_log(self.stdout_buffer)
            self.stdout_buffer = ""
        report_path = (
            self.current_run_directory / "report.json"
            if self.current_run_directory is not None
            else None
        )
        self.process = None
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if exit_code == 0 else 0)
        if exit_code == 0 and report_path is not None and report_path.exists():
            try:
                self._display_report(report_path)
                self.statusBar().showMessage("Alignment completed")
            except Exception as error:
                QMessageBox.warning(self, "Result display failed", str(error))
        else:
            self.statusBar().showMessage("Alignment failed or was stopped")

    def load_report(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load AlignImg GUI report",
            self.output.text() or str(Path.cwd()),
            "AlignImg report (report.json *.json);;JSON files (*.json)",
        )
        if path:
            try:
                self._display_report(Path(path))
            except Exception as error:
                QMessageBox.critical(self, "Cannot load report", str(error))

    def _display_report(self, path: Path) -> None:
        self.results.load_report(path)
        self.diagnostics.load_report(path)
        self.summary.load_report(path)
        self._append_log(f"Loaded report: {path}")

    def _append_log(self, value: str) -> None:
        self.log.appendPlainText(value.rstrip())
