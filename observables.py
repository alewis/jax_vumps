"""
These functions make computations upon an MPS based upon its interpretation
as a quantum state.
"""
import os
import importlib
import jax_vumps.contractions as ct

try:
    environ_name = os.environ["LINALG_BACKEND"]
except KeyError:
    print("While importing observables.py,")
    print("os.environ[\"LINALG_BACKEND\"] was undeclared; using NumPy.")
    environ_name = "numpy"

if environ_name == "jax":
    mps_linalg_name = "jax_vumps.jax_backend.mps_linalg"
elif environ_name == "numpy":
    mps_linalg_name = "jax_vumps.numpy_backend.mps_linalg"
else:
    raise ValueError("Invalid LINALG_BACKEND ", environ_name)

mps_linalg = importlib.import_module(mps_linalg_name)


