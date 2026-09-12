#!/usr/bin/env python3
"""Capture reproducibility metadata without requiring project dependencies."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


def main() -> None:
    metadata: dict[str, object] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "python_executable": sys.executable,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_branch": command_output(["git", "branch", "--show-current"]),
        "git_status": command_output(["git", "status", "--short", "--branch"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "epic_gpu_uuids": os.environ.get("EPIC_GPU_UUIDS"),
        "epic_gpu_lease_expires_at": os.environ.get("EPIC_GPU_LEASE_EXPIRES_AT"),
    }

    try:
        import torch

        metadata["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "visible_cuda_device_count": torch.cuda.device_count(),
            "visible_cuda_devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        metadata["torch"] = None

    if shutil.which("nvidia-smi"):
        metadata["nvidia_smi"] = command_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
        metadata["nvidia_processes"] = command_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader",
            ]
        )

    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
