# READ-Grid

Reference code, configuration and result records for the manuscript

> Separating availability from policy quality: guarded edge fallback for demand
> response under communication outages

READ-Grid (*Resilient Edge-Native Autonomous Demand Response for Smart Grids*)
is an executable Python simulation of a distribution feeder in which a
cloud-hosted demand-response policy is guarded by a local command-acceptance
boundary at the edge gateway. The repository contains everything needed to
reproduce the numbers, tables and figures reported in the paper.

## What the study separates

Two claims that the literature often conflates:

1. **Command-path availability.** When the backhaul fails, does the feeder keep
   executing a policy at all?
2. **Policy quality.** Given that a policy executes, is a learned policy better
   than a tuned rule?

The guarded edge architecture establishes (1). It says nothing about (2), and
the paper reports that (2) is not established: the pre-registered primary
comparison between the corrected linear learner and a rule enumerated over 144
candidates is **inconclusive** at the declared margin δ = 0.02.

## Layout

```
src/readgrid/        simulator, controllers, authority arbiter, metrics
configs/             protocol_v2.yaml (frozen evaluation protocol) and variants
analysis/            every script that produces a table, figure or interval
models_v2/           trained linear value-function weights (five seeds per arm)
data/                IEEE 123-bus models and the 21 evaluation load profiles
runs/                Run 1 records (stage_main.jsonl) and the split assignment
runs_v2/             Run 2 records, analysis JSON, and the artifact registers
figures/             regenerated Figures 1-7 and v2_figure_sources.json
```

`runs_v2/stage_main_v2.jsonl.gz` is the gzip of the 59,850-record Run 2 stage
file (21 episodes x 95 faults x 30 policy instances).

## Reproducing the reported results

```bash
pip install -r requirements.lock.txt

# tables, contrasts, intervals
python analysis/s6_analysis_v2.py

# Figures 1-7, with live assertions on every value they display
python analysis/v2_figures.py
```

`analysis/v2_figures.py` does not merely plot. Each figure asserts the values it
draws against the archived JSON and, for the primary contrast, recomputes the
equal-weight-per-fault bootstrap from `runs_v2/stage_main_v2.jsonl.gz`. If a
number in the paper drifts, the script exits non-zero rather than drawing a
stale figure. The assertions are listed in `figures/v2_figure_sources.json`.

## Frozen protocol

The evaluation protocol was frozen before the held-out run:

```
configs/protocol_v2.yaml
SHA-256 8639ff7e0630b2ab6b8b0e134701182d3a2df559e0b8270f74bb4b266116b0de
```

The same hash is reported in the manuscript. `runs_v2/protocol_v2_frozen.yaml`
is the copy carried with the run records.

## Key definitions

| Quantity | Definition |
|---|---|
| Target achievement | fraction of eligible DR steps with net feeder power at or below the external cap |
| Selection score | `achievement - 1.0 * curtail_fraction - 0.0001 * switches_per_group_run` |
| Primary contrast | C4v2 - C3b on target achievement, each fault id weighted equally within an episode, episodes resampled (5,000 draws) |
| Equivalence margin | δ = 0.02 (2.0 percentage points), two one-sided tests |

## Licence

MIT. See `LICENSE`.

## Citing

If you use this code or the run records, please cite the manuscript above. A
DOI-bearing archive will be deposited on acceptance.
