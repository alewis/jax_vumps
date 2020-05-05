"""
Interface functions for VUMPS.
"""
import sys
import os
import numpy as np

import jax_vumps.matrices as mat
import jax_vumps.vumps as vumps


def runvumps(H, bond_dimension: int, gradient_tol: float,
             max_iter: int, delta_0=0.1, checkpoint_every=500,
             out_directory="./vumps_output",
             heff_krylov_params=vumps.krylov_params(),
             env_solver_params=vumps.solver_params(),
             jax_linalg=False):

    """
    Performs a vumps simulation of some Hamiltonian H.

    PARAMETERS
    ----------
    H (array, dxdxdxd): The Hamiltonian to be simulated.
    bond_dimension (int): Bond dimension of the MPS.
    gradient_tol (float): VUMPS will terminate once the MPS gradient reaches
                          this tolerance.
    max_iter (int)       : VUMPS will terminate after this many iterations
                          even if tolerance has not been reached.
    delta_0 (float)        : Initial value for the gradient norm. The
                             convergence thresholds of the various solvers at
                             the initial step are proportional to this, via
                             coefficients in the Krylov and solver param dicts.
    checkpoint_every (int) : Simulation data is pickled at this periodicity.
    out_directory (string) : Output is saved here. The directory is created
                             if it doesn't exist.
    heff_krylov_params(dict):Hyperparameters for an eigensolve of certain
                             'effective Hamiltonians'. Formed by
                             'krylov_params()'.
    env_solver_params      : Hyperparameters for a linear solve that finds
                             the effective Hamiltonians. Formed by
                             'solver_params()'.
    jax_linalg (bool)   : Determines whether Jax or numpy code is used in
                          certain linear algebra calls.
    """

    if jax_linalg:
        os.environ["LINALG_BACKEND"] = "jax"
    else:
        os.environ["LINALG_BACKEND"] = "numpy"
    out = vumps.vumps(H, bond_dimension, gradient_tol, max_iter,
                      delta_0=delta_0, checkpoint_every=checkpoint_every,
                      out_directory=out_directory,
                      heff_krylov_params=heff_krylov_params,
                      env_solver_params=env_solver_params)
    return out


def vumpsXX(bond_dimension: int, gradient_tol: float,
            maxiter: int, delta_0=0.1, checkpoint_every=500,
            out_directory="./vumps",
            heff_krylov_params=vumps.krylov_params(),
            env_solver_params=vumps.solver_params(),
            jax_linalg=False,
            dtype=np.float32):
    """
    Performs a vumps simulation of the XX model,
    H = XX + YY. Parameters are the same as in runvumps.
    """
    H = mat.H_XX(jax=jax_linalg, dtype=dtype)
    out = runvumps(H, bond_dimension, gradient_tol, maxiter, delta_0=delta_0,
                   checkpoint_every=checkpoint_every,
                   out_directory=out_directory,
                   heff_krylov_params=heff_krylov_params,
                   env_solver_params=env_solver_params,
                   jax_linalg=jax_linalg)
    return out

def vumps_XX(bond_dimension, gradient_tol=1E-4, path="/testout/np_vumps_XX",
             maxiter=100, jax=True, dtype=np.float32):
    """
    Performs a vumps simulation of the XX model,
    H = XX + YY.

    PARAMETERS
    ----------
    path (string): Path to the directory where output is to be saved. It will
                   be created if it doesn't exist.
    bond_dimension (int): Bond dimension of the MPS.
    gradient_tol (float): VUMPS will terminate once the MPS gradient reaches
                          this tolerance.
    maxiter (int)       : VUMPS will terminate after this many iterations
                          even if tolerance has not been reached.
    jax (bool)   : Determines whether Jax or numpy code is used.
    dtype        : dtype of the output and internal computations.
    """
    H = mat.H_XX(jax=jax, dtype=dtype)
    out = runvumps(H, path, bond_dimension, gradient_tol, maxiter, jax=jax)
    return out
#  def vumps_XXZ(bond_dimension, gradient_tol=1E-4, maxiter=100,
#                path="/testout/np_vumps_xxz",
#                delta=1, ud=2, scale=1, jax=True, dtype=np.float32):
#      """
#      Performs a vumps simulation of the XXZ model,
#      H = (-1/(8scale))*[ud*[UD + DU] + delta*ZZ]

#      PARAMETERS
#      ----------
#      path (string): Path to the directory where output is to be saved. It will
#                     be created if it doesn't exist.
#      bond_dimension (int): Bond dimension of the MPS.
#      gradient_tol (float): VUMPS will terminate once the MPS gradient reaches
#                            this tolerance.
#      maxiter (int)       : VUMPS will terminate after this many iterations
#                            even if tolerance has not been reached.
#      jax (bool)   : Determines whether Jax or numpy code is used.
#      dtype        : dtype of the output and internal computations.
#      """
#      H = mat.H_XXZ(delta=delta, ud=ud, scale=scale, jax=jax, dtype=dtype)
#      out = runvumps(H, path, bond_dimension, gradient_tol, maxiter, jax=jax)
#      return out




#  def vumps_ising(J, h, bond_dimension, gradient_tol=1E-4,
#                  path="/testout/np_vumps_ising", maxiter=100, jax=True,
#                  dtype=np.float32):
#      """
#      Performs a vumps simulation of the trasverse-field Ising model,
#      H = J * XX + h * ZI

#      PARAMETERS
#      ----------
#      J, h (float) : Couplings.
#      path (string): Path to the directory where output is to be saved. It will
#                     be created if it doesn't exist.
#      bond_dimension (int): Bond dimension of the MPS.
#      gradient_tol (float): VUMPS will terminate once the MPS gradient reaches
#                            this tolerance.
#      maxiter (int)       : VUMPS will terminate after this many iterations
#                            even if tolerance has not been reached.
#      jax (bool)   : Determines whether Jax or numpy code is used.
#      dtype        : dtype of the output and internal computations.
#      """
#      H = mat.H_ising(J, h, jax=jax, dtype=dtype)
#      out = runvumps(H, path, bond_dimension, gradient_tol, maxiter, jax=jax)
#      return out
