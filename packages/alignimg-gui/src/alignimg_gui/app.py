"""Application entry point."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import PyQt6


def configure_qt_plugin_path() -> None:
    """Prefer the platform plugins bundled with a pip-installed PyQt6."""
    plugins = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    platforms = plugins / "platforms"
    use_bundled_plugins = plugins.is_dir() and "QT_PLUGIN_PATH" not in os.environ
    if use_bundled_plugins:
        os.environ["QT_PLUGIN_PATH"] = str(plugins)
    if platforms.is_dir() and "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms)
    if use_bundled_plugins:
        from PyQt6.QtCore import QCoreApplication

        QCoreApplication.setLibraryPaths([str(plugins)])


configure_qt_plugin_path()

import pyqtgraph as pg  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from .window import MainWindow  # noqa: E402


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("AlignImg Workbench")
    app.setOrganizationName("AlignImg")
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
