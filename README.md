# Infra Experiments

Independent Git repository for the executable side of the `Research/Infra` workbench.

## Boundary

Tracked here:

- experiment entrypoints and reusable helpers;
- small, reviewable configs;
- smoke tests and environment-capture tools;
- documentation needed to reproduce a run.

Not tracked here:

- datasets and model weights;
- checkpoints, logs, profiler traces, or generated results;
- credentials and machine-specific paths;
- research conclusions that belong in Obsidian.

Generated evidence is written under `artifacts/<experiment>/<run-id>/` and ignored by Git.
Move important remote artifacts to persistent storage and record their exact path in the
corresponding Obsidian experiment note.

## Local setup

完整的 Conda 环境职责、安装、验证、升级和故障排查见 [环境与依赖维护](docs/环境与依赖维护.md)。

Use Python 3.10 or newer. PyTorch is intentionally not pinned in this repository because the
correct build depends on the local or server CUDA runtime.

```bash
cd code/infra-experiments
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch build appropriate for this machine, then:
python -m pip install -e '.[dev]'
```

Run the current Phase 0 workload:

```bash
./scripts/run_phase0.sh cpu local-cpu-smoke
```

The launcher records the repository state and machine environment before running. Add
`--profile` after the run id to write a profiler table and trace:

```bash
./scripts/run_phase0.sh cpu local-cpu-profile --profile
```

## Server workflow

The local repository has no remote until one is explicitly configured. After creating an
empty private repository on the chosen Git host:

```bash
git remote add origin <REMOTE_URL>
git push -u origin main
```

On the server, use a clean checkout rather than copying files ad hoc:

```bash
git clone <REMOTE_URL> infra-experiments
cd infra-experiments
git status --short --branch
git rev-parse HEAD
```

Before a CUDA launch, install the PyTorch build matching the server image/driver, inspect GPU
visibility and active processes, choose a persistent `ARTIFACT_ROOT`, then run:

```bash
export ARTIFACT_ROOT=/path/to/persistent/infra-experiments
./scripts/run_phase0.sh cuda <unique-run-id> --profile
```

Do not infer that a CUDA run is valid from exit code alone. Review `environment.json`, the
exact commit, correctness/loss, profiler output, and the experiment-specific acceptance gate.

## Layout

```text
configs/                  checked-in experiment contracts
experiments/              executable workloads grouped by phase or lab
scripts/                  launch and environment-capture helpers
tests/                    low-cost validation
artifacts/                generated local evidence, ignored by Git
```
