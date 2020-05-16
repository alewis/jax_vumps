"""
Interface functions for VUMPS.
"""
import sys
import os
import numpy as np
from importlib import reload

import pickle as pkl

import jax_vumps.matrices as mat
import jax_vumps.vumps as vumps
import jax_vumps.params as params


def runvumps(H, bond_dimension: int, delta_0=0.1,
             out_directory="./vumps_output", jax_linalg=True,
             vumps_params=params.vumps_params(),
             heff_params=params.krylov_params(),
             env_params=params.gmres_params()):
    """
    Performs a vumps simulation of some Hamiltonian H.

    PARAMETERS
    ----------
    H (array, dxdxdxd): The Hamiltonian to be simulated.
    bond_dimension (int): Bond dimension of the MPS.
    delta_0 (float)        : Initial value for the gradient norm. The
                             convergence thresholds of the various solvers at
                             the initial step are proportional to this, via
                             coefficients in the Krylov and solver param dicts.
    out_directory (string) : Output is saved here. The directory is created
                             if it doesn't exist.
    jax_linalg (bool)   : Determines whether Jax or numpy code is used in
                          certain linear algebra calls.
    vumps_params (dict)    : Hyperparameters for the vumps solver. Formed
                             by 'vumps_params'.
    heff_params (dict)     : Hyperparameters for an eigensolve of certain
                             'effective Hamiltonians'. Formed by
                             'krylov_params()'.
    env_params (dict)      : Hyperparameters for a linear solve that finds
                             the effective Hamiltonians. Formed by
                             'solver_params()'.
    """
    if jax_linalg:
        os.environ["LINALG_BACKEND"] = "jax"
    else:
        os.environ["LINALG_BACKEND"] = "numpy"
    reload(vumps)
    out = vumps.vumps(H, bond_dimension, delta_0=0.1,
                      out_directory=out_directory,
                      vumps_params=vumps_params,
                      heff_params=heff_params,
                      env_params=env_params)
    return out


def vumps_XX(bond_dimension: int, delta_0=0.1,
             out_directory="./vumps", jax_linalg=True,
             dtype=np.float32,
             vumps_params=params.vumps_params(),
             heff_params=params.krylov_params(),
             env_params=params.gmres_params()):
    """
    Performs a vumps simulation of the XX model,
    H = XX + YY. Parameters are the same as in runvumps.
    """
    H = mat.H_XX(jax=jax_linalg, dtype=dtype)
    out = runvumps(H, bond_dimension, delta_0=delta_0,
                   out_directory=out_directory, jax_linalg=jax_linalg,
                   vumps_params=vumps_params, heff_params=heff_params,
                   env_params=env_params)
    return out


def vumps_ising(J, h, bond_dimension: int, delta_0=0.1,
                out_directory="./vumps", jax_linalg=True,
                dtype=np.float32,
                vumps_params=params.vumps_params(),
                heff_params=params.krylov_params(),
                env_params=params.gmres_params()):
    """
    Performs a vumps simulation of the XX model,
    H = XX + YY. Parameters are the same as in runvumps.
    """
    H = mat.H_ising(J, h, jax=jax_linalg, dtype=dtype)
    out = runvumps(H, bond_dimension, delta_0=delta_0,
                   out_directory=out_directory, jax_linalg=jax_linalg,
                   vumps_params=vumps_params, heff_params=heff_params,
                   env_params=env_params)
    return out


def vumps_from_checkpoint(checkpoint_path, out_directory="./vumps_load",
                          new_vumps_params=None, new_heff_params=None,
                          new_env_params=None, jax_linalg=True):
    """
    Find the ground state of a uniform two-site Hamiltonian
    using Variational Uniform Matrix Product States. This is a gradient
    descent method minimizing the distance between a given MPS and the
    best approximation to the physical ground state at its bond dimension.

    This interface function initializes vumps from checkpointed data.

    PARAMETERS
    ----------
    checkpoint_path (string): Path to the checkpoint .pkl file.
    """
    writer = vumps.make_writer(out_directory)
    if jax_linalg:
        os.environ["LINALG_BACKEND"] = "jax"
    else:
        os.environ["LINALG_BACKEND"] = "numpy"
    reload(vumps)
    with open(checkpoint_path, "rb") as f:
        chk = pkl.load(f)

    H, iter_data, vumps_params, heff_params, env_params, Niter = chk
    if new_vumps_params is not None:
        vumps_params = {**vumps_params, **new_vumps_params}

    if new_heff_params is not None:
        heff_params = {**heff_params, **new_heff_params}

    if new_env_params is not None:
        env_params = {**env_params, **new_env_params}

    out = vumps.vumps_work(H, iter_data, vumps_params, heff_params, env_params,
                           writer, Niter0=Niter)
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
