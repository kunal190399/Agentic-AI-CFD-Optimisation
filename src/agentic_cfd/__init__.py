"""Agentic engineering simulation demonstrator (reduced-order proof of concept)."""
from .flow_solver import Design, simulate
from .agent import FlowOptimisationAgent
__all__ = ['Design','simulate','FlowOptimisationAgent']
