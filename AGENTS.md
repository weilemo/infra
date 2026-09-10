# Repository Instructions

## Scope

This repository contains executable LLM training and serving infrastructure experiments.
Keep conceptual notes and research conclusions in the parent Obsidian `Research/Infra`
workbench.

## Experiment contract

- Record the Git commit, exact command, environment, hardware, config, seed, warmup,
  repetitions, correctness gate, and resource budget.
- Write every run to a unique directory below `artifacts/`; never overwrite raw logs from a
  previous run.
- Keep datasets, model weights, checkpoints, profiler traces, and large results out of Git.
- Separate smoke success, completed execution, automatic evaluation, manual review, and
  scientific conclusions.
- On a server, inspect branch, commit, dirty state, GPU visibility, and active jobs before a
  launch. Do not modify or delete unknown remote outputs.

## Changes

- Prefer small experiment entrypoints and checked-in JSON configs.
- Add a smoke test or dry-run validation when adding an experiment.
- Do not hard-code credentials, personal absolute paths, GPU indices, or server hostnames.

