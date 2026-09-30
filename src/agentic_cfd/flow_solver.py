"""Transparent 2D *reduced-order* pressure-driven gas-flow demonstrator.

Physics: steady, single-phase, incompressible Darcy/lubrication-type model
    div(k grad(p)) = 0, u = -k grad(p).
It is NOT a Navier-Stokes solver, plasma-chemistry model, or validated OIPT reactor.
Finite-volume cell-centred discretisation, no-flux top/bottom/closed inlet,
unit-pressure specified inlet opening and zero-pressure outlet.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

BOUNDS = {
    'inlet_center': (0.25, 0.75),
    'inlet_width': (0.12, 0.70),
    'gap_center': (0.30, 0.70),
    'gap_width': (0.18, 0.70),
}

@dataclass(frozen=True)
class Design:
    inlet_center: float = .30
    inlet_width: float = .16
    gap_center: float = .62
    gap_width: float = .24

    def validate(self) -> None:
        for key, (lo, hi) in BOUNDS.items():
            value = getattr(self, key)
            if not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{key}={value} outside safe demonstration bounds [{lo}, {hi}]')
        if not 0 <= self.inlet_center-self.inlet_width/2 and self.inlet_center+self.inlet_width/2 <= 1:
            raise ValueError('Inlet aperture extends beyond domain')
        if not 0 <= self.gap_center-self.gap_width/2 and self.gap_center+self.gap_width/2 <= 1:
            raise ValueError('Baffle aperture extends beyond domain')

    def values(self) -> list[float]:
        return [float(getattr(self, k)) for k in BOUNDS]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

@dataclass
class Result:
    design: Design
    pressure: np.ndarray
    permeability: np.ndarray
    flux_x: np.ndarray  # flux across cell faces: shape (nx+1, ny)
    flux_y: np.ndarray  # shape (nx, ny+1)
    inlet_flux: float
    outlet_flux: float
    conservation_error: float
    uniformity_cv: float
    objective: float
    target_profile: np.ndarray
    target_y: np.ndarray
    target_index: int
    nx: int
    ny: int

    def metrics(self) -> dict[str, Any]:
        return {
            **self.design.to_dict(),
            'objective': float(self.objective),
            'uniformity_cv': float(self.uniformity_cv),
            'inlet_flux': float(self.inlet_flux),
            'outlet_flux': float(self.outlet_flux),
            'mass_balance_relative_error': float(self.conservation_error),
            'grid': f'{self.nx}x{self.ny}',
            'model': 'steady 2D pressure-driven Darcy/lubrication approximation',
        }

def _harmonic(a: float, b: float) -> float:
    return 2.0*a*b/(a+b)

def simulate(design: Design, nx: int = 60, ny: int = 36, required_flux: float = .65) -> Result:
    """Run one conservative 2D FVM case and return directly auditable metrics."""
    design.validate()
    if nx < 16 or ny < 16:
        raise ValueError('Grid must be at least 16 by 16')
    if required_flux <= 0:
        raise ValueError('required_flux must be positive')
    dx, dy = 1.0/nx, 1.0/ny
    x = (np.arange(nx)+.5)*dx
    y = (np.arange(ny)+.5)*dy
    permeability = np.ones((nx, ny), dtype=float)
    baffle_x = (x >= .46) & (x <= .51)
    outside_opening = np.abs(y-design.gap_center) > design.gap_width/2
    permeability[np.ix_(baffle_x, outside_opening)] = .015
    inlet = np.abs(y-design.inlet_center) <= design.inlet_width/2
    if inlet.sum() < 2:
        raise ValueError('Inlet opening smaller than two y-cells; refine the grid')
    if not baffle_x.any():
        raise ValueError('Baffle not resolved; refine x grid')
    rows, cols, vals = [], [], []
    rhs = np.zeros(nx*ny, dtype=float)

    def put(row: int, col: int, value: float) -> None:
        rows.append(row); cols.append(col); vals.append(value)

    def idx(i: int, j: int) -> int:
        return i*ny+j

    for i in range(nx):
        for j in range(ny):
            r = idx(i,j)
            k = permeability[i,j]
            diag = 0.0
            for di, dj, surface_ratio in [(1,0,dy/dx),(-1,0,dy/dx),(0,1,dx/dy),(0,-1,dx/dy)]:
                ni,nj = i+di, j+dj
                if 0 <= ni < nx and 0 <= nj < ny:
                    trans = _harmonic(k, permeability[ni,nj])*surface_ratio
                    diag += trans
                    put(r, idx(ni,nj), -trans)
                elif di == -1 and i == 0 and inlet[j]:
                    # Half-cell distance between centre and pressure-specified inlet.
                    trans = 2*k*dy/dx
                    diag += trans
                    rhs[r] += trans * 1.0
                elif di == 1 and i == nx-1:
                    trans = 2*k*dy/dx
                    diag += trans  # outlet p=0, no RHS contribution
                # all other boundaries are no-flux
            put(r, r, diag)

    A = coo_matrix((vals,(rows,cols)),shape=(nx*ny,nx*ny)).tocsr()
    p = spsolve(A,rhs).reshape(nx,ny)
    if not np.isfinite(p).all() or p.min() < -1e-7 or p.max() > 1+1e-7:
        raise RuntimeError('Pressure solver returned a nonphysical field')
    qx = np.zeros((nx+1,ny), dtype=float)
    qy = np.zeros((nx,ny+1), dtype=float)
    qx[0,inlet] = 2*permeability[0,inlet]*(1-p[0,inlet])*dy/dx
    qx[-1,:] = 2*permeability[-1,:]*p[-1,:]*dy/dx
    for i in range(1,nx):
        faces = 2*permeability[i-1,:]*permeability[i,:]/(permeability[i-1,:]+permeability[i,:])
        qx[i,:] = faces*(p[i-1,:]-p[i,:])*dy/dx
    for j in range(1,ny):
        faces = 2*permeability[:,j-1]*permeability[:,j]/(permeability[:,j-1]+permeability[:,j])
        qy[:,j] = faces*(p[:,j-1]-p[:,j])*dx/dy
    q_in, q_out = float(qx[0,:].sum()),float(qx[-1,:].sum())
    conservation = abs(q_in-q_out)/max(abs(q_in), abs(q_out),1e-12)
    if conservation > 1e-6:
        raise RuntimeError(f'Mass conservation check failed ({conservation:.2g})')
    target_idx = min(nx-1, int(nx*.67))
    sample = (y>=.12)&(y<=.88)
    target = qx[target_idx,sample]/dy  # approximate local superficial velocity
    cv = float(np.std(target)/max(abs(np.mean(target)),1e-12))
    # Goal: flow uniformity across nominal wafer region, subject to sufficient throughput.
    # This transparent scalar score (smaller is better) avoids trivial near-zero-flow wins.
    throughput_penalty = .8*max(0,(required_flux-q_out)/required_flux)
    score = cv + throughput_penalty
    return Result(design,p,permeability,qx,qy,q_in,q_out,conservation,cv,float(score),
                  target,y[sample],target_idx,nx,ny)
