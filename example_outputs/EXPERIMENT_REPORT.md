# Experiment report - Agentic gas-flow optimisation

**Status:** Executed local proof of concept. Not a validated OIPT or plasma-reactor model.

## Research question
Can a goal-driven AI workflow independently configure, run, evaluate and optimise a simplified engineering flow model while preserving user-defined geometric bounds and checking conservation?

## Modelling approach
2D steady pressure-driven Darcy/lubrication approximation, solved by cell-centred finite volumes using SciPy sparse linear algebra. Unit pressure at a user-defined inlet slot, zero pressure at the outlet, sealed remainder of boundaries, and a low-permeability baffle with variable aperture. It is a fast proxy used solely to demonstrate orchestration, not a full CFD/Navier-Stokes or plasma-etch/deposition model.

## Agent architecture
PLAN (feasible geometry candidate) -> VALIDATE (hard bounds) -> SIMULATE (sparse pressure solver) -> OBSERVE (uniformity, throughput, mass balance) -> LEARN (Random Forest surrogate) -> SELECT (uncertainty-aware acquisition) -> RECORD and REPORT.

## Reproducible conditions
- Seed: 42; grid: 60 x 36; evaluations including baseline: 23.
- Objective = CV of axial flux at x~0.67 over y in [0.12,0.88], plus a soft penalty for outlet flux below 0.65 (all units normalised).
- CV is standard deviation / mean; smaller indicates a more even profile.

## Measured outputs from this demonstration run

| Metric | Baseline | Selected design |
|---|---:|---:|
| Flux non-uniformity (CV) | 0.3384 | 0.0741 |
| Scalar objective (lower better) | 0.5912 | 0.0753 |
| Normalised outlet flux | 0.4446 | 0.6490 |
| Relative mass-balance error | 3.35e-13 | 2.19e-13 |

- Relative CV change against chosen starting geometry: **78.1% decrease**.
- Objective change against chosen starting geometry: **87.3% decrease**.
- Runtime in this environment: **1.8 seconds** (hardware and grid dependent).
- Best geometry: `{"inlet_center": 0.4436952311571625, "inlet_width": 0.15382519501081166, "gap_center": 0.4980930403289631, "gap_width": 0.7}`.

## Limitations and recommended industrial extension
These numbers show optimisation **within this simplified model only**. They do not measure OpenFOAM speedup, plasma uniformity, industrial CFD accuracy or real machine performance. The next stage is to connect the validated agent interface to an approved OpenFOAM case, add 3D geometry and plasma/process physics where relevant, verify mesh independence and compare predictions against experiment or trusted solver results. An engineering supervisor must approve objective definitions, constraints and designs.

## Submission contents
`agent_trace.csv`, `metrics.json`, `best_fields.npz`, `simulation_comparison.png`, `optimisation_progress.png`.
