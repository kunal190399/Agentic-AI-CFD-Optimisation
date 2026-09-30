# Why I undertook this project

## Natural first-person explanation (application / interview)

With a Master's degree in Computer Science focused on AI and Machine Learning, I wanted to apply my expertise to a problem where AI could have a measurable engineering impact rather than build another standalone predictive model. The University of Exeter and Oxford Instruments Plasma Technology KTP challenge drew my attention to the time and expertise required to set up, run and interpret computational flow simulations during semiconductor process development.

I therefore initiated this project to explore whether an autonomous, auditable AI workflow could support a closed simulation-and-optimisation loop: define a design objective, propose safe parameters, call a numerical solver, assess performance and physical consistency, learn from the outcomes, and select the next experiment. I used a simplified two-dimensional pressure-driven gas-flow benchmark so I could validate the agent architecture, optimisation logic and reporting independently before introducing the much greater complexity of full CFD and plasma physics.

The motivation was to bridge my existing AI/ML engineering strengths with computational science, develop a better understanding of numerical modelling, and show how academic AI methods could be translated into practical tools for engineers. The prototype is intentionally a proof of concept; a natural next step would be integration with an approved OpenFOAM or commercial CFD workflow, supervised by engineers and tested against trusted results.

## 35-second interview answer

I chose this project because I wanted to move beyond AI models in isolation and apply my AI/ML background to a real engineering workflow. Semiconductor process modelling is an interesting challenge because repeated simulations can take considerable setup, analysis and optimisation effort. I built a small agentic proof of concept that automatically proposes flow-chamber configurations, validates the inputs, runs a reduced-order numerical model and learns from the results using a machine-learning surrogate. It helped me explore both the potential of agentic automation and the need for engineering checks and traceability. My next step would be to connect this architecture to a validated CFD solver and real process metrics.

## Honest application-ready project entry

**Agentic AI for Automated Gas-Flow Simulation and Optimisation | Independent Portfolio Project | September 2026**

- Built an autonomous Python workflow linking constrained design planning, a 2D finite-volume pressure-driven flow model, numerical validation and experiment tracking.
- Implemented a Random Forest surrogate to propose subsequent simulation settings based on observed objective values and model uncertainty.
- Applied geometric guardrails, flow-uniformity/throughput metrics and conservation checks; produced reproducible reports, visualisations and test coverage.
- In an illustrative 23-case reference run, reduced axial-flow non-uniformity (coefficient of variation) from 0.3384 to 0.0741 (78.1% relative reduction) **within the simplified benchmark**; identified integration with validated CFD/plasma models as the next research stage.

## Do not claim

- That this project uses real Oxford Instruments equipment, process recipes or industrial data.
- That it contains full OpenFOAM, Navier-Stokes, plasma etching or deposition simulations.
- That 78.1% means actual industrial productivity, plasma uniformity or production improvement.
- That an LLM generated physical conclusions; the implemented planning intelligence is local, learned surrogate-based selection under explicit constraints.
