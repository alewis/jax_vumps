"""
Interface functions for VUMPS.
"""
import sys
import os
import numpy as np

import jax_vumps.matrices as mat
import jax_vumps.vumps as vumps


def runvumps(H, path: str, bond_dimension: int, gradient_tol: float,
             maxiter: int, jax_linalg=True):
    """
    Performs a vumps simulation of some Hamiltonian H.

    PARAMETERS
    ----------
    H (array, dxdxdxd): The Hamiltonian to be simulated.
    path (string): Path to the directory where output is to be saved. It will
                   be created if it doesn't exist.
    bond_dimension (int): Bond dimension of the MPS.
    gradient_tol (float): VUMPS will terminate once the MPS gradient reaches
                          this tolerance.
    maxiter (int)       : VUMPS will terminate after this many iterations
                          even if tolerance has not been reached.
    jax (bool)   : Determines whether Jax or numpy code is used.
    """
    here = sys.path[0]
    fullpath = here + path

    vumps_params = vumps.vumps_params(path=fullpath,
                                      chi=bond_dimension,
                                      vumps_tol=gradient_tol,
                                      maxiter=maxiter
                                      )
    if jax_linalg:
        os.environ["LINALG_BACKEND"] == "Jax"
    else:
        os.environ["LINALG_BACKEND"] == "NumPy"
    out = vumps.vumps(H, params=vumps_params)
    return out


def vumps_XXZ(bond_dimension, gradient_tol=1E-4, maxiter=100,
              path="/testout/np_vumps_xxz",
              delta=1, ud=2, scale=1, jax=True, dtype=np.float32):
    """
    Performs a vumps simulation of the XXZ model,
    H = (-1/(8scale))*[ud*[UD + DU] + delta*ZZ]

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
    H = mat.H_XXZ(delta=delta, ud=ud, scale=scale, jax=jax, dtype=dtype)
    out = runvumps(H, path, bond_dimension, gradient_tol, maxiter, jax=jax)
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


def vumps_ising(J, h, bond_dimension, gradient_tol=1E-4,
                path="/testout/np_vumps_ising", maxiter=100, jax=True,
                dtype=np.float32):
    """
    Performs a vumps simulation of the trasverse-field Ising model,
    H = J * XX + h * ZI

    PARAMETERS
    ----------
    J, h (float) : Couplings.
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
    H = mat.H_ising(J, h, jax=jax, dtype=dtype)
    out = runvumps(H, path, bond_dimension, gradient_tol, maxiter, jax=jax)
    return out
