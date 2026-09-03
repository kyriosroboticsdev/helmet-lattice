"""ML pipeline package for the TPU lattice helmet-liner project (geometry-only study).

Modules
-------
schema            : the Data Handoff Contract (spec Section 2) as code.
make_dummy_data   : synthetic dataset matching the schema (run pipeline before real data).
data              : load + validate the master table against the contract.
features          : physics-informed engineered features; canonical design->feature matrix.
importance        : RF / gradient-boosting + SHAP feature importance (grouped CV).
surrogate_gp      : Gaussian-process surrogate with grouped CV + uncertainty calibration.
optimize          : multi-objective Bayesian optimization + active-learning loop.
"""
