"""End-to-end CLI/demo pipeline test: the same path exercised by `cmvt-demo` and CI."""

from __future__ import annotations

from cmvt.cli import main


def test_main_runs_and_writes_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    exit_code = main([])
    assert exit_code == 0
    report_path = tmp_path / "reports" / "validation_report.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "# Automated Validation Report" in content
    out = capsys.readouterr().out
    assert "Synthetic cohorts" in out
    assert "Validation plan" in out
