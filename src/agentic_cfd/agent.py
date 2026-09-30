"""Tool-using, feedback-driven optimisation agent with a learned surrogate planner.

The planner is LOCAL and auditable: after exploratory evaluations it fits a random
forest to simulator feedback and selects feasible designs via an uncertainty-aware
acquisition rule. No opaque LLM decisions or external API keys are required.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .flow_solver import Design, Result, BOUNDS, simulate

@dataclass
class Trace:
    step: int
    phase: str
    params: Design
    objective: float | None
    cv: float | None
    outlet_flux: float | None
    mass_balance_error: float | None
    status: str
    rationale: str
    elapsed_s: float

class FlowOptimisationAgent:
    """PLAN -> VALIDATE -> CALL SIMULATOR -> OBSERVE -> UPDATE PLAN -> REPORT."""
    def __init__(self, simulator: Callable[..., Result] = simulate,
                 seed: int = 42, nx: int = 60, ny: int = 36):
        self.simulator = simulator
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.nx, self.ny = nx, ny
        self.trace: list[Trace] = []
        self.results: list[Result] = []
        self.baseline: Result | None = None
        self.best: Result | None = None
        self.used: set[tuple[float, ...]] = set()

    def _new_valid_design(self) -> Design:
        # Constrained uniform design proposals. Individual widths are chosen
        # first; centres then remain within both the specified parameter bounds
        # and chamber edges. Never executes arbitrary strings/code.
        iw = float(self.rng.uniform(*BOUNDS['inlet_width']))
        gw = float(self.rng.uniform(*BOUNDS['gap_width']))
        ic_lo = max(BOUNDS['inlet_center'][0], iw/2)
        ic_hi = min(BOUNDS['inlet_center'][1], 1-iw/2)
        gc_lo = max(BOUNDS['gap_center'][0], gw/2)
        gc_hi = min(BOUNDS['gap_center'][1], 1-gw/2)
        d = Design(inlet_center=float(self.rng.uniform(ic_lo,ic_hi)),
                   inlet_width=iw,
                   gap_center=float(self.rng.uniform(gc_lo,gc_hi)),
                   gap_width=gw)
        d.validate()
        return d

    def _propose(self) -> tuple[Design,str,str]:
        if len(self.results) < 9:
            return self._new_valid_design(), 'exploration', 'Sample an admissible design to map objective landscape.'
        X = np.array([r.design.values() for r in self.results])
        y = np.array([r.objective for r in self.results])
        model = RandomForestRegressor(n_estimators=65, min_samples_leaf=1,
                                      random_state=self.seed, n_jobs=1)
        model.fit(X,y)
        candidates = [self._new_valid_design() for _ in range(256)]
        # Local adaptation near the current best, alongside global exploration.
        assert self.best is not None
        b = self.best.design
        for _ in range(64):
            for _attempt in range(12):
                values = [float(np.clip(v+self.rng.normal(0,.07*(hi-lo)),lo,hi))
                          for v,(lo,hi) in zip(b.values(), BOUNDS.values())]
                d = Design(*values)
                try:
                    d.validate()
                except ValueError:
                    continue
                candidates.append(d)
                break
        candidates = [c for c in candidates if tuple(round(v,5) for v in c.values()) not in self.used]
        if not candidates:
            return self._new_valid_design(),'exploration','Fallback because candidate pool was exhausted.'
        # Lower confidence bound: balance predicted quality with exploratory uncertainty.
        predictions = np.array([tree.predict(np.array([c.values() for c in candidates]))
                                for tree in model.estimators_])
        mu = predictions.mean(axis=0)
        sd = predictions.std(axis=0)
        acquisition = mu - .5*sd
        which = int(np.argmin(acquisition))
        return (candidates[which], 'surrogate-guided',
                f'Surrogate suggested objective {mu[which]:.4f} +/- {sd[which]:.4f}; '
                'selected by lower confidence bound while satisfying geometry bounds.')

    def _observe(self, design: Design, phase: str, rationale: str) -> None:
        step = len(self.trace)
        started = time.perf_counter()
        try:
            design.validate()
            res = self.simulator(design, nx=self.nx, ny=self.ny)
            self.results.append(res)
            if phase == 'baseline':
                self.baseline = res
            if self.best is None or res.objective < self.best.objective:
                previous = 'none' if self.best is None else f'{self.best.objective:.4f}'
                self.best = res
                rationale += f' New best objective (previous: {previous}).'
            self.trace.append(Trace(step,phase,design,res.objective,res.uniformity_cv,
                                    res.outlet_flux,res.conservation_error,'ok',rationale,
                                    time.perf_counter()-started))
        except (ValueError, RuntimeError, ArithmeticError) as ex:
            # Simulated tool failures are logged and excluded from model training.
            self.trace.append(Trace(step,phase,design,None,None,None,None,'failed',
                                    rationale + f' Simulator/validator rejected case: {ex}',
                                    time.perf_counter()-started))
        self.used.add(tuple(round(v,5) for v in design.values()))

    def run(self, iterations: int = 22, verbose: bool = True) -> 'FlowOptimisationAgent':
        if iterations < 8:
            raise ValueError('Use at least 8 optimisation iterations for a meaningful demonstration.')
        self._observe(Design(), 'baseline', 'Evaluate fixed unoptimised chamber geometry.')
        for _ in range(iterations):
            proposal,phase,reason = self._propose()
            self._observe(proposal,phase,reason)
            if verbose:
                last=self.trace[-1]
                print(f'[{last.step:02d}] {phase:16s} {last.status:6s} '
                      f'obj={last.objective if last.objective is not None else float("nan"):.4f} '
                      f'best={self.best.objective if self.best else float("nan"):.4f}')
        if self.baseline is None or self.best is None:
            raise RuntimeError('No successful simulations were completed')
        return self
