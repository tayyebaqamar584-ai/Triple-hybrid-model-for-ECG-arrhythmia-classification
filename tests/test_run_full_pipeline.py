import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_full_pipeline",
        ROOT / "src" / "run_full_pipeline.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_python_executable_prefers_working_interpreter():
    module = load_module()
    resolved = module.resolve_python_executable()

    assert resolved == sys.executable
    assert resolved and Path(resolved).exists()


def test_run_step_raises_when_command_fails():
    module = load_module()

    try:
        module.run_step("intentional test failure", [sys.executable, "-c", "raise SystemExit(3)"])
    except RuntimeError as error:
        assert "intentional test failure" in str(error)
        assert "exit 3" in str(error)
    else:
        raise AssertionError("run_step must fail when a pipeline command fails")
