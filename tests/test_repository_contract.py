import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_phase0_config_is_fixed_contract() -> None:
    config_path = REPO_ROOT / "configs" / "phase0" / "tiny-decoder-fixed-v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["workload"] == "tiny-decoder-fixed-v1"
    assert config["model"]["layers"] == 4
    assert config["model"]["hidden_size"] == 512
    assert config["workload_config"]["batch_size"] == 2
    assert config["workload_config"]["sequence_length"] == 256
    assert config["workload_config"]["seed"] == 0
    assert config["measurement"] == {"warmup_steps": 2, "measured_steps": 8}


def test_generated_evidence_is_ignored() -> None:
    ignore_rules = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    for directory in ["artifacts/", "data/", "logs/", "runs/", "checkpoints/"]:
        assert directory in ignore_rules
