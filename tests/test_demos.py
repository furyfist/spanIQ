"""Offline demos run from the bundled fixtures and write only to the chosen output directory."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")


@pytest.mark.parametrize(
    "module, stem",
    [
        ("spaniq.demos.model_swap", "model_swap"),
        ("spaniq.demos.prompt_injection", "prompt_injection"),
        ("spaniq.demos.rag_breakage", "rag_breakage"),
    ],
)
def test_offline_demo_writes_to_output_dir(tmp_path, monkeypatch, module, stem):
    import importlib
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    demo = importlib.import_module(module)
    out = tmp_path / "out"
    demo.run(offline=True, output_dir=out)

    assert (out / f"{stem}_report.txt").exists()
    assert (out / f"{stem}_alerts.jsonl").exists()
    assert list(out.glob("*.png"))
    # nothing written into the installed package
    assert not (Path(demo.__file__).parent / "output").exists()
