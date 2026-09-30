"""Machine-readable traces, visual evidence and reproducible technical summary."""
from __future__ import annotations
from dataclasses import asdict
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .agent import FlowOptimisationAgent


def _plot_fields(agent: FlowOptimisationAgent, path: Path) -> None:
    baseline,best = agent.baseline,agent.best
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,(r,label) in enumerate([(baseline,'Baseline'),(best,'Optimised')]):
        im=axes[row,0].imshow(r.pressure.T,origin='lower',extent=(0,1,0,1),aspect='auto',vmin=0,vmax=1,cmap='viridis')
        axes[row,0].contour(r.permeability.T,levels=[.5],colors=['white'],linewidths=.8,
                            origin='lower',extent=(0,1,0,1))
        axes[row,0].set_title(f'{label}: pressure (white = resistant baffle)')
        axes[row,0].set_xlabel('Chamber length (normalised)')
        axes[row,0].set_ylabel('Chamber height (normalised)')
        q=np.asarray(r.flux_x[r.target_index,:]) / (1/r.ny)
        axes[row,1].plot((np.arange(r.ny)+.5)/r.ny,q,lw=2)
        axes[row,1].axvspan(.12,.88,color='gray',alpha=.12,label='wafer proxy sampling zone')
        axes[row,1].set_ylim(bottom=0)
        axes[row,1].set_title(f'{label}: superficial axial flux at x~0.67 (CV {r.uniformity_cv:.3f})')
        axes[row,1].set_xlabel('Normalised vertical position')
        axes[row,1].set_ylabel('Axial flux / face height (normalised)')
        axes[row,1].grid(alpha=.2)
    fig.colorbar(im,ax=[axes[0,0],axes[1,0]],location='left',shrink=.7,label='Normalised pressure',pad=.08)
    fig.suptitle('Reduced-order, pressure-driven gas-flow benchmark (not a full CFD/plasma reactor)',fontsize=12)
    fig.savefig(path,dpi=175,bbox_inches='tight')
    plt.close(fig)


def _plot_progress(agent: FlowOptimisationAgent, path: Path) -> None:
    ok=[t for t in agent.trace if t.status=='ok']
    scores=np.array([t.objective for t in ok])
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.plot([t.step for t in ok],scores,'o-',lw=1,alpha=.65,label='Each simulation')
    ax.plot([t.step for t in ok],np.minimum.accumulate(scores),'-',lw=2.5,label='Best-so-far objective')
    ax.axvline(8.5,ls='--',color='gray',label='Approx. surrogate-model start')
    ax.set(xlabel='Agent iteration (0 = baseline)',ylabel='Objective (lower is better)',
           title='Closed-loop autonomous design search')
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(path,dpi=175,bbox_inches='tight')
    plt.close(fig)


def write_outputs(agent: FlowOptimisationAgent, output: Path, runtime_s: float) -> dict:
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    baseline,best=agent.baseline,agent.best
    assert baseline is not None and best is not None
    with (output/'agent_trace.csv').open('w',newline='',encoding='utf-8') as fp:
        fieldnames=list(asdict(agent.trace[0]).keys())
        writer=csv.DictWriter(fp,fieldnames=fieldnames)
        writer.writeheader()
        for t in agent.trace:
            x=asdict(t)
            x['params']=json.dumps(x['params'])
            writer.writerow(x)
    improvement=100*(baseline.uniformity_cv-best.uniformity_cv)/baseline.uniformity_cv
    objective_reduction=100*(baseline.objective-best.objective)/baseline.objective
    summary={
        'project':'Agentic AI for automated gas-flow modelling and optimisation: proof of concept',
        'model':'2D steady pressure-driven Darcy/lubrication-type approximation; NOT validated Navier-Stokes/plasma CFD',
        'seed':agent.seed,'grid':[agent.nx,agent.ny], 'total_evaluations':len(agent.trace),
        'successful_evaluations':len(agent.results),
        'runtime_seconds':round(runtime_s,3),
        'uniformity_cv_reduction_percent':round(improvement,2),
        'objective_reduction_percent':round(objective_reduction,2),
        'baseline':baseline.metrics(), 'best':best.metrics(),
        'objective_definition':'CV of superficial axial flux in y=[0.12,0.88] at x~=0.67 + 0.8*max(0,(0.65-outlet_flux)/0.65)',
        'caveats':['normalised geometry and pressure; no gas chemistry, plasma kinetics, temperature, or Navier-Stokes inertia',
                   'results are proof-of-concept metrics only, not validated semiconductor process predictions',
                   'AI component is a local feedback-driven Random Forest surrogate planner; no LLM API is needed']
    }
    with (output/'metrics.json').open('w',encoding='utf-8') as fp:
        json.dump(summary,fp,indent=2)
    np.savez_compressed(output/'best_fields.npz',pressure=best.pressure,
                        permeability=best.permeability,flux_x=best.flux_x,flux_y=best.flux_y,
                        baseline_pressure=baseline.pressure,baseline_flux_x=baseline.flux_x)
    _plot_fields(agent,output/'simulation_comparison.png')
    _plot_progress(agent,output/'optimisation_progress.png')
    doc=f'''# Experiment report - Agentic gas-flow optimisation

**Status:** Executed local proof of concept. Not a validated OIPT or plasma-reactor model.

## Research question
Can a goal-driven AI workflow independently configure, run, evaluate and optimise a simplified engineering flow model while preserving user-defined geometric bounds and checking conservation?

## Modelling approach
2D steady pressure-driven Darcy/lubrication approximation, solved by cell-centred finite volumes using SciPy sparse linear algebra. Unit pressure at a user-defined inlet slot, zero pressure at the outlet, sealed remainder of boundaries, and a low-permeability baffle with variable aperture. It is a fast proxy used solely to demonstrate orchestration, not a full CFD/Navier-Stokes or plasma-etch/deposition model.

## Agent architecture
PLAN (feasible geometry candidate) -> VALIDATE (hard bounds) -> SIMULATE (sparse pressure solver) -> OBSERVE (uniformity, throughput, mass balance) -> LEARN (Random Forest surrogate) -> SELECT (uncertainty-aware acquisition) -> RECORD and REPORT.

## Reproducible conditions
- Seed: {agent.seed}; grid: {agent.nx} x {agent.ny}; evaluations including baseline: {len(agent.trace)}.
- Objective = CV of axial flux at x~0.67 over y in [0.12,0.88], plus a soft penalty for outlet flux below 0.65 (all units normalised).
- CV is standard deviation / mean; smaller indicates a more even profile.

## Measured outputs from this demonstration run

| Metric | Baseline | Selected design |
|---|---:|---:|
| Flux non-uniformity (CV) | {baseline.uniformity_cv:.4f} | {best.uniformity_cv:.4f} |
| Scalar objective (lower better) | {baseline.objective:.4f} | {best.objective:.4f} |
| Normalised outlet flux | {baseline.outlet_flux:.4f} | {best.outlet_flux:.4f} |
| Relative mass-balance error | {baseline.conservation_error:.2e} | {best.conservation_error:.2e} |

- Relative CV change against chosen starting geometry: **{improvement:.1f}% decrease**.
- Objective change against chosen starting geometry: **{objective_reduction:.1f}% decrease**.
- Runtime in this environment: **{runtime_s:.1f} seconds** (hardware and grid dependent).
- Best geometry: `{json.dumps(best.design.to_dict())}`.

## Limitations and recommended industrial extension
These numbers show optimisation **within this simplified model only**. They do not measure OpenFOAM speedup, plasma uniformity, industrial CFD accuracy or real machine performance. The next stage is to connect the validated agent interface to an approved OpenFOAM case, add 3D geometry and plasma/process physics where relevant, verify mesh independence and compare predictions against experiment or trusted solver results. An engineering supervisor must approve objective definitions, constraints and designs.

## Submission contents
`agent_trace.csv`, `metrics.json`, `best_fields.npz`, `simulation_comparison.png`, `optimisation_progress.png`.
'''
    (output/'EXPERIMENT_REPORT.md').write_text(doc,encoding='utf-8')
    return summary
