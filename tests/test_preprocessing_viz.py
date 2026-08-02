import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "preprocessing_visualization.py"

spec = importlib.util.spec_from_file_location("preprocessing_visualization", MODULE_PATH)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_filter_config_defaults():
    cfg = module.FilterConfig()
    assert cfg.lowcut_hz == 0.5
    assert cfg.highcut_hz == 40.0
    assert cfg.order == 3
    assert cfg.sampling_rate_hz == 360
