#!/usr/bin/env python3
"""Run the offline autonomous gas-flow optimisation demonstration.
Usage: python run_demo.py --iterations 22 --output example_outputs
"""
import argparse, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from agentic_cfd.agent import FlowOptimisationAgent
from agentic_cfd.reporting import write_outputs

def main():
    parser=argparse.ArgumentParser(description='Agentic reduced-order gas-flow automation + ML-guided optimisation')
    parser.add_argument('--iterations',type=int,default=22,help='Additional simulations after baseline (>=8)')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--nx',type=int,default=60)
    parser.add_argument('--ny',type=int,default=36)
    parser.add_argument('--output',type=Path,default=ROOT/'example_outputs')
    args=parser.parse_args()
    start=time.perf_counter()
    agent=FlowOptimisationAgent(seed=args.seed,nx=args.nx,ny=args.ny).run(args.iterations)
    summary=write_outputs(agent,args.output,time.perf_counter()-start)
    print('\nDONE. Results in:',args.output.resolve())
    print(f"CV: {summary['baseline']['uniformity_cv']:.4f} -> {summary['best']['uniformity_cv']:.4f} "
          f"({summary['uniformity_cv_reduction_percent']:.1f}% decrease from chosen baseline)")
    print(f"Objective: {summary['baseline']['objective']:.4f} -> {summary['best']['objective']:.4f}")
    print('IMPORTANT: these are reduced-order proof-of-concept results, NOT verified industrial CFD data.')

if __name__=='__main__':
    main()
