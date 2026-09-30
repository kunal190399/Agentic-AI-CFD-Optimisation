"""Basic physics, safety, repeatability, and agent feedback tests."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agentic_cfd.flow_solver import Design, simulate
from agentic_cfd.agent import FlowOptimisationAgent

def test_pressure_and_conservation():
    res=simulate(Design(),nx=36,ny=24)
    assert res.pressure.shape==(36,24)
    assert np.min(res.pressure)>=-1e-9 and np.max(res.pressure)<=1+1e-9
    assert res.conservation_error<1e-7
    assert res.outlet_flux>0
    assert res.uniformity_cv>0

def test_reject_nonphysical_design():
    with pytest.raises(ValueError):
        simulate(Design(inlet_width=.95))
    with pytest.raises(ValueError):
        simulate(Design(gap_center=.30,gap_width=.70))

def test_straightforward_better_than_intentionally_poor_baseline():
    baseline=simulate(Design(),nx=36,ny=24)
    improved=simulate(Design(.5,.6,.5,.6),nx=36,ny=24)
    assert improved.uniformity_cv<baseline.uniformity_cv
    assert improved.outlet_flux>baseline.outlet_flux

def test_agent_reproducible_and_within_bounds():
    a=FlowOptimisationAgent(seed=17,nx=26,ny=20).run(9,verbose=False)
    b=FlowOptimisationAgent(seed=17,nx=26,ny=20).run(9,verbose=False)
    assert len(a.trace)==10
    assert all(t.status=='ok' for t in a.trace)
    assert [t.params for t in a.trace]==[t.params for t in b.trace]
    assert [t.objective for t in a.trace]==pytest.approx([t.objective for t in b.trace])
    assert a.best.objective<=a.baseline.objective
    assert any(t.phase=='surrogate-guided' for t in a.trace)
