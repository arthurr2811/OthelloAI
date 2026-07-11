"""Smoke-Test: stellt sicher, dass die Pakete importierbar sind und pytest läuft.

Wird ersetzt/erweitert, sobald die Engine (Phase 1) steht.
"""

import importlib


def test_packages_importable():
    for name in ("othello", "agents", "az", "web", "config"):
        assert importlib.import_module(name) is not None
