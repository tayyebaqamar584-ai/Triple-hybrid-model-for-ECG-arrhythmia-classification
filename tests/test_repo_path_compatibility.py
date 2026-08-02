from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_archived_scripts_do_not_use_hardcoded_windows_project_path():
    for relative_path in [
        "src/_tune_proposed.py",
        "src/archive/_comparison_report.py",
        "src/archive/_generate_v3_metrics.py",
        "src/archive/_tune_proposed_optimized.py",
        "src/archive/_tune_proposed.py",
    ]:
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "d:\\ECG_Project_Complete" not in content
