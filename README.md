# Agentic AI for Automated Gas-Flow Simulation & Optimisation
### An independently reproducible, reduced-order CFD-workflow proof of concept

> **Scope declaration:** This repository is a working **agentic simulation-automation prototype**, not an industrial semiconductor CFD/plasma-etching simulator. To make it reproducible without OpenFOAM, the included numerical tool solves a **2D steady pressure-driven Darcy/lubrication-type gas-flow approximation** using finite volumes. It does **not** solve the full Navier-Stokes equations, plasma kinetics, species transport, thermal dynamics, or reactor chemistry. The clear next step is to connect the agent to a validated OpenFOAM or company-approved solver.

## Why this project?

Inspired by the University of Exeter x Oxford Instruments Plasma Technology (OIPT) KTP challenge: conventional CFD workflows require repetitive manual setup, execution and interpretation. As a Computer Science graduate specialising in AI/ML, I wanted to test a practical question: can an autonomous, safety-bounded AI workflow propose simulation configurations, run a physics tool, learn from its feedback, and optimise an engineering metric while maintaining traceability and basic physics checks?

Rather than claim expertise in proprietary semiconductor equipment, the initial study uses a small **transparent benchmark chamber** with a movable inlet and a low-permeability baffle. This puts the focus on agent orchestration, numerical methods, machine learning, validation and explainable industrial integration.

## What is implemented?

1. **Goal:** reduce axial flow non-uniformity in a nominal wafer sampling band, while penalising inadequate total outlet throughput.
2. **Planner:** autonomous constrained design proposals, initially exploring the parameter space and subsequently fitting a local Random Forest surrogate to simulation observations.
3. **Tools:** geometry validation, SciPy sparse finite-volume flow simulation, mass-conservation checking, metric calculation, trace logging and plotting.
4. **Reflection loop:** choose a next candidate using predicted objective and model uncertainty; automatically update from each run's result; keep the best measured case (not just the surrogate prediction).
5. **Artifacts:** traceable CSV of every step, JSON metrics, compressed field arrays, before/after plots, convergence history and automatically generated Markdown technical report.

**This agent does not require an LLM API key.** Its AI component is a trained, feedback-driven Random Forest surrogate planner. The deterministic supervisory loop is a deliberate safety and reproducibility choice; an LLM interface is a future extension, not a feature falsely claimed here.

## Architecture

```text
ENGINEERING GOAL: improve local flux uniformity, retain sufficient throughput
                       |
                       v
                 AUTONOMOUS AGENT
  Plan --> Geometry guardrails --> Simulation tool --> Physics checks
    ^                                               |
    |                                               v
    +<-- RF surrogate learning <-- Observations and scored metrics
                       |
                       v
             Best measured design + auditable report
```

## Run (Python 3.10 or later)

Mac / Linux:

```bash
cd Agentic_CFD_KTP
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py --iterations 22 --seed 42 --output example_outputs
python -m pytest -q
```

Windows PowerShell:

```powershell
cd Agentic_CFD_KTP
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_demo.py --iterations 22 --seed 42 --output example_outputs
python -m pytest -q
```

For a faster run: `python run_demo.py --iterations 10 --nx 40 --ny 24 --output quick_run`.
For a finer resolution: `python run_demo.py --iterations 30 --nx 90 --ny 54 --output fine_run`.

## Initial reference run

`example_outputs` contains a completed run (seed 42, grid 60 x 36, 22 new design evaluations plus one fixed starting point). **Only for the specified model and intentionally non-optimised starting design**:

| Measured quantity | Starting design | Best measured design |
|---|---:|---:|
| Axial flow CV (smaller = more uniform) | 0.3384 | 0.0741 |
| Objective (smaller = better) | 0.5912 | 0.0753 |
| Normalised outlet flux | 0.4446 | 0.6490 |
| Relative mass-balance residual | 3.35e-13 | 2.19e-13 |

The CV is **78.1% lower** than the chosen baseline under this model and grid. This is **not** an empirical claim of improvement in real semiconductor production or an OpenFOAM CFD benchmark. All design choices and outputs are logged for review. Optimiser-selected changes include alterations to slot position/width and baffle opening within documented geometric limits.

### What to show an interviewer

- `example_outputs/simulation_comparison.png` -- pressure fields, baffle location, baseline/optimised flow distribution.
- `example_outputs/optimisation_progress.png` -- autonomous decisions and measured results across iterations.
- `example_outputs/agent_trace.csv` -- phase, inputs, agent rationale, objective, errors and elapsed time per run.
- `example_outputs/EXPERIMENT_REPORT.md` -- research question, methodology, numerical results and limitations.

## Numerical model and design variables

Governing approximation: `div(k grad(p)) = 0` and `u = -k grad(p)` on a unit square. Pressure is 1 at the inlet opening, 0 at the right outlet; the remaining left edge and top/bottom walls are no-flux. The narrow vertical baffle is assigned low (non-zero) permeability outside a selectable aperture. Face fluxes use harmonic-mean coefficients; pressures are solved with SciPy sparse linear algebra. Continuity is checked by comparing inlet/outlet integrated flux.

Parameter bounds (all in normalised chamber height):

| Design parameter | Allowed range |
|---|---:|
| Inlet centre | 0.25 - 0.75 |
| Inlet aperture width | 0.12 - 0.70 |
| Baffle opening centre | 0.30 - 0.70 |
| Baffle opening width | 0.18 - 0.70 |

The agent also checks that apertures remain entirely inside the domain.

Objective = `CV(axial superficial flux measured at x ~ 0.67, y from 0.12 to 0.88)` + `0.8 x max(0, (0.65 - total outlet flux) / 0.65)`. The throughput term prevents near-zero flow from being rewarded for superficial uniformity. Note that these normalised thresholds are **illustrative**, not OIPT operating parameters.

## Technical validation / limitations

- Safety: parameter bounds and geometric feasibility validated **before** each simulation.
- Physics: bounded finite pressure field, mass balance checked; cases that fail are logged and excluded from ML training.
- Repeatability: pseudo-random seed and fixed numerical configuration supplied in CLI; tests check repeatability.
- Mesh dependence: a finer mesh should be assessed before interpreting trends; the reference best design at 60x36 gives approximately CV 0.0810 at 90x54 and 0.0748 at 120x72 (model-specific differences). This is a preliminary sensitivity illustration, not formal grid-convergence proof.
- Scientific limits: no turbulence, kinetic plasma chemistry, real gas mixtures, wafer surface reactions, full 3D geometry, actual pressure/flow calibration, manufacturing constraints, or experimental validation.
- External integration: no OpenFOAM binary or proprietary OIPT model was used for the provided results.

## Path from prototype to industrial KTP project

1. Agree physics model, objective metrics and safety limits with plasma/process engineers.
2. Replace `simulate(design, nx, ny)` with a verified adapter for an approved OpenFOAM or in-house CFD template; control mesh, boundary files and solver through a strict parameter whitelist.
3. Parse pressure/velocity/species outputs; enforce convergence, mesh quality, physical plausibility and mass-balance thresholds.
4. Fit/test the surrogate on validated CFD data, benchmarking baseline manual workflows, wall time, accuracy and repeated runs.
5. Add human sign-off for suggested engineering changes; maintain complete experiment/version traceability.

## Folder map

```text
run_demo.py                      CLI for complete demonstration
requirements.txt                 Python dependencies
src/agentic_cfd/flow_solver.py   Reduced-order finite-volume physics tool
src/agentic_cfd/agent.py         Autonomous constrained ML-guided optimisation
src/agentic_cfd/reporting.py     CSV/JSON/figures/report generation
tests/test_project.py            Physics/safety/agent reproducibility tests
example_outputs/                Completed demonstration, plots and logs
docs/PROJECT_MOTIVATION.md       Copy-ready reason and application description
docs/INTERVIEW_PREPARATION.md    Honest technical Q&A and demo script
docs/TWO_DAY_SUBMISSION_PLAN.md  Short plan up to 2 October deadline
```

**Integrity note:** Present this portfolio as something you can personally run, understand, explain and extend. Do not describe this as a production-ready Oxford Instruments system or claim deployment, proprietary-data use or verified plasma performance.
