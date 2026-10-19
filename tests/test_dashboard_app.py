"""The Streamlit app renders end to end against a seeded database, using Streamlit's headless AppTest."""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from spaniq.monitor.timeline_store import TimelineStore  # noqa: E402

APP = pathlib.Path(__file__).parent.parent / "src" / "spaniq" / "dashboard" / "app.py"


@pytest.fixture()
def seeded_cwd(tmp_path, monkeypatch):
    # the app reads --db after "--" in argv, which AppTest cannot pass, so it uses ./spaniq.db
    monkeypatch.chdir(tmp_path)
    store = TimelineStore("spaniq.db")
    for i in range(30):
        store.record(
            f"t{i}",
            "pipe/gen",
            "ResponseDriftMetric",
            0.05 + i * 0.001,
            0.1,
            True,
            f"2024-01-01T00:00:{i:02d}+00:00",
            component="gen",
        )
    store.record_alert("2024-01-01T00:00:29+00:00", "ResponseDriftMetric", 0.2, 0.1, "ALERT: drift")
    return tmp_path


def test_app_renders_without_exceptions(seeded_cwd):
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_app_renders_on_empty_database(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_filters_list_components_from_the_database(seeded_cwd):
    def page():
        from spaniq.dashboard.components.filters import render_filters
        from spaniq.dashboard.config import DashboardConfig

        render_filters(DashboardConfig(db_path="spaniq.db"))

    at = AppTest.from_function(page, default_timeout=30)
    at.run()
    assert not at.exception
    assert at.sidebar.selectbox[0].options == ["All", "gen"]
