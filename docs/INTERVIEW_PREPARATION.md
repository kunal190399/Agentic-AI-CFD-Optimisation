# Interview walkthrough and Q&A

## 90-second live demo script

1. Explain the problem: automate the loop of setting up, solving, evaluating and updating an engineering flow scenario.
2. Show `run_demo.py` and execute `python run_demo.py --iterations 10 --output quick_run`.
3. Open `src/agentic_cfd/agent.py`: baseline, bounded exploration, Random Forest fit, predicted score and uncertainty, selection of next case.
4. Open `flow_solver.py`: conservative finite-volume discretisation, pressure/flux fields and explicit mass balance.
5. Show `simulation_comparison.png` and `optimisation_progress.png`, then point to `agent_trace.csv` to demonstrate auditability.
6. Finish with limitations: this is a pressure-driven Darcy-type flow demonstrator, **not** a validated plasma or full Navier-Stokes CFD model. Explain the OpenFOAM integration next step.

## Common questions

**Why is this agentic rather than just a fixed optimisation loop?**
It has a goal, parameter-selection autonomy, strict tool-validation boundaries, observation/feedback, a learned surrogate planning model, iterative decisions and a recorded trace. It adapts candidate choices to actual simulator feedback. It does not require an LLM; agentic behaviour is not synonymous with chatbot behaviour.

**What is the AI/ML?**
A scikit-learn Random Forest regressor is retrained on actual run inputs and scalar objectives after exploratory simulations. It predicts scores for candidate designs; tree-level variation provides a practical heuristic uncertainty signal. A lower-confidence-bound acquisition score balances exploration and improvement. This is a heuristic, not a calibrated probabilistic uncertainty estimate.

**What physics do you simulate?**
A steady incompressible, pressure-driven Darcy/lubrication approximation solving div(k grad p)=0 via finite volume. Low-permeability cells mimic a baffle. It shows flow distribution under changing inlet and aperture geometry. It omits momentum inertia, Navier-Stokes boundary layers, species chemistry, temperature and plasma physics.

**How do you know the output is not nonsense?**
Geometry is range-checked; pressure is checked for finite bounded values; integrated inlet versus outlet flux is verified to a tight tolerance. A preliminary finer-grid comparison and deterministic repeatability tests are included. This is numerical self-consistency, **not** empirical validation.

**Why isn't OpenFOAM included?**
The near-term goal was to prove and independently demonstrate the autonomous orchestration and validation workflow with an installation-light solver. OpenFOAM (or OIPT's approved solver) would be integrated via a controlled adapter once mesh, geometry, process physics and verified baseline cases are agreed with engineering supervisors. Never imply an OpenFOAM benchmark happened.

**What are your benchmark numbers?**
For the documented seed-42, 60x36, 23-evaluation **simplified reference benchmark**: coefficient of variation reduced from 0.3384 for the chosen starting geometry to 0.0741 for the selected design. Mass-balance residual stayed extremely small. The selected opening geometry itself changed, so this demonstrates bounded geometry optimisation within the proxy model and is not proof of performance in a real machine.

**How would you deploy this at Oxford Instruments?**
Map a restricted design parameter schema to verified CFD case templates; integrate scheduler and post-processing; define acceptance tests and meaningful process objectives with the engineers; enforce solver convergence and physical checks; version the models/datasets; involve engineers in every sign-off and evaluate repeatability and accuracy against ground-truth or established workflows.

**What would you improve next?**
A validated 3D CFD adapter, mesh independence study, multi-objective optimisation, species transport, experimental calibration, human-in-the-loop approvals, and stronger uncertainty estimation such as cross-validation or Gaussian processes where appropriate.
