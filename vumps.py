import copy
import os
import importlib

import numpy as np

from jax_vumps.writer import Writer
import jax_vumps.contractions as ct
import jax_vumps.observables as obs

import jax_vumps.jax_backend.environment as jax_environment
import jax_vumps.numpy_backend.environment as np_environment

import jax_vumps.jax_backend.heff as jax_heff
import jax_vumps.numpy_backend.heff as np_heff

try:
    environ_name = os.environ["LINALG_BACKEND"]
except KeyError:
    print("While importing vumps.py,")
    print("os.environ[\"LINALG_BACKEND\"] was undeclared; using NumPy.")
    environ_name = "numpy"

if environ_name == "jax":
    mps_linalg_name = "jax_vumps.jax_backend.mps_linalg"
elif environ_name == "numpy":
    mps_linalg_name = "jax_vumps.numpy_backend.mps_linalg"
else:
    raise ValueError("Invalid LINALG_BACKEND ", environ_name)

mps_linalg = importlib.import_module(mps_linalg_name)


##########################################################################
# Functions to handle output.
##########################################################################
def ostr(string):
    """
    Truncates to four decimal places.
    """
    return '{:1.4e}'.format(string)


def output(writer, Niter, delta, dE, E, norm, B2):
    """
    Does the actual outputting.
    """
    outstr = "N = " + str(Niter) + "| |B| = " + ostr(delta)
    outstr += "| E = " + '{0:1.16f}'.format(E)
    outstr += "| dE = " + ostr(np.abs(dE))
    outstr += "| |B2| = " + ostr(B2)
    writer.write(outstr)

    this_output = np.array([Niter, E, dE, delta, B2, norm])
    writer.data_write(this_output)


def make_writer(outdir=None):
    """
    Initialize the Writer. Creates a directory in the appropriate place, and
    an output file with headers hardcoded here as 'headers'. The Writer,
    defined in writer.py, remembers the directory and will append to this
    file as the simulation proceeds. It can also be used to pickle data,
    notably the final wavefunction.

    PARAMETERS
    ----------
    outdir (string): Path to the directory where output is to be saved. This
                     directory will be created if it does not yet exist.
                     Otherwise any contents with filename collisions during
                     the simulation will be overwritten.

    OUTPUT
    ------
    writer (writer.Writer): The Writer.
    """

    headers = ["N", "E", "dE", "|B|", "|B2|", "<psi>"]
    if outdir is None:
        return None
    writer = Writer(outdir, headers=headers)
    return writer

###############################################################################
# Effective environment.
###############################################################################


def solve_environment(mpslist, delta, fpoints, H, env_solver_params,
                      H_env=None):
    if H_env is None:
        H_env = [None, None]
    lh, rh = H_env  # lowercase means 'from previous iteration'

    A_L, C, A_R = mpslist
    rL, lR = fpoints

    if env_solver_params["use_jax"]:
        LH = jax_environment.solve_for_LH(A_L, H, lR, env_solver_params,
                                          delta,
                                          oldLH=lh)
        RH = jax_environment.solve_for_RH(A_R, H, rL, env_solver_params,
                                          delta,
                                          oldRH=rh)
    else:
        LH = np_environment.solve_for_LH(A_L, H, lR, env_solver_params,
                                         delta,
                                         oldLH=lh)
        RH = np_environment.solve_for_RH(A_R, H, rL, env_solver_params,
                                         delta,
                                         oldRH=rh)
    LHtest = np_environment.solve_for_LH(A_L, H, lR, env_solver_params, delta, oldLH=lh,
                          dense=True)
    print(np.allclose(LHtest, LH))
    print(mps_linalg.frobnorm(LHtest, LH))
    H_env = [LH, RH]
    return H_env

###############################################################################
# Main loop and friends.
###############################################################################
def vumps_approximate_tm_eigs(C):
    """
    Returns the approximate transfer matrix dominant eigenvectors,
    rL ~ C^\dag C, and lR ~ C C\dag = rL\dag, both trace-normalized.
    """
    rL = (C.T.conj()) @ C
    rL /= mps_linalg.trace(rL)
    lR = rL.T.conj()
    return (rL, lR)


def vumps_initialization(d: int, chi: int, dtype=np.float32):
    """
    Generate a random uMPS in mixed canonical forms, along with the left
    dominant eV L of A_L and right dominant eV R of A_R.

    PARAMETERS
    ----------
    d: Physical dimension.
    chi: Bond dimension.
    dtype: Data dtype of tensors.

    RETURNS
    -------
    mpslist = [A_L, C, A_R]: Arrays. A_L and A_R have shape (d, chi, chi),
                             and are respectively left and right orthogonal.
                             C is the (chi, chi) centre of orthogonality.
    A_C (array, (d, chi, chi)) : A_L @ C. One of the equations vumps minimizes
                                 is A_L @ C = C @ A_R = A_C.
    fpoints = [rL, lR] = C^dag @ C and C @ C^dag respectively. Will converge
                         to the left and right environment Hamiltonians.
                         Both are chi x chi.
    """
    A_1, A_2 = mps_linalg.random_tensors([(d, chi, chi), (d, chi, chi)],
                                         dtype=dtype)

    A_L, C_1 = mps_linalg.qrpos(A_1)
    C_2, A_R = mps_linalg.lqpos(A_2)
    C = C_1@C_2
    A_C = ct.rightmult(A_L, C)
    fpoints = vumps_approximate_tm_eigs(C)
    mpslist = [A_L, C, A_R]
    return (mpslist, A_C, fpoints)


def vumps_iteration(iter_data, delta, H, heff_krylov_params,
                    env_solver_params):
    """
    One main iteration of VUMPS.
    """
    mpslist, A_C, fpoints, H_env = iter_data
    if heff_krylov_params["use_jax"]:
        mpslist, A_C, delta = jax_heff.apply_gradient(iter_data, delta, H,
                                                      heff_krylov_params)
    else:
        mpslist, A_C, delta = np_heff.apply_gradient(iter_data, delta, H,
                                                     heff_krylov_params)

    fpoints = vumps_approximate_tm_eigs(mpslist[1])
    H_env = solve_environment(mpslist, delta, fpoints, H, env_solver_params,
                              H_env=H_env)
    return ((mpslist, A_C, fpoints, H_env), delta)


def _diagnostics(oldmpslist, Eold, H, iter_data):
    """
    Makes a few computations to output during a vumps run.
    """
    mpslist, A_C, fpoints, H_env = iter_data
    E = obs.twositeexpect(mpslist, H)
    dE = E - Eold
    norm = obs.norm(mpslist)
    B2 = obs.B2_variance(oldmpslist, mpslist)
    return dE, E, norm, B2


def krylov_params(n_krylov=40, max_restarts=10, tol_coef=0.1, use_jax=False):
    """
    Bundles parameters for the Lanczos eigensolver. These control
    the expense of finding the left and right environment tensors, and of
    minimizing the effective Hamiltonians.

    PARAMETERS
    ----------
    n_krylov (int): Size of the Krylov subspace.
    max_restarts (int): Maximum number of times to iterate the Krylov
                        space construction.
    tol_coef (float): This number times the MPS gradient will be the
                      convergence threshold of the eigensolve.
    """
    return {"n_krylov": n_krylov, "max_restarts": max_restarts,
            "tol_coef": tol_coef, "use_jax": use_jax}


def solver_params(inner_m=40, outer_k=3, maxiter=50, tol_coef=0.00001,
                  use_jax=False):
    """
    Bundles parameters for the (L)GMRES linear solver. These control
    the expense of finding the left and right environment Hamiltonians.

    For GMRES, these are in fact the same parameters as used in the Lanczos
    solve, but when/if we generalize to LGMRES they will not be.

    PARAMETERS
    ----------
    n_krylov (int): Size of the Krylov subspace.
    max_restarts (int): Maximum number of times to iterate the Krylov
                        space construction.
    tol_coef (float): This number times the MPS gradient will set the
                      convergence threshold of the linear solve.
    use_jax (bool) : Toggles whether the Jax backedn is used.
    """
    return {"inner_m": inner_m, "maxiter": maxiter, "outer_k": outer_k,
            "tol_coef": tol_coef, "use_jax": use_jax}

    #  return {"n_krylov": n_krylov, "maxter": max_restarts,
    #          "tol_coef": tol_coef, "use_jax": use_jax}


def vumps(H, chi: int, gradient_tol: float, max_iter: int,
          delta_0=0.1,
          checkpoint_every=500,
          out_directory="./vumps",
          heff_krylov_params=krylov_params(),
          env_solver_params=solver_params()):
    """
    Find the ground state of a uniform two-site Hamiltonian
    using Variational Uniform Matrix Product States. This is a gradient
    descent method minimizing the distance between a given MPS and the
    best approximation to the physical ground state at its bond dimension.

    PARAMETERS
    ----------
    H (array, (d, d, d, d)): The Hamiltonian whose ground state is to be found.
    chi (int)              : MPS bond dimension.
    gradient_tol (float)   : Convergence is declared once the gradient norm is
                             at least this small.
    max_iter (int)         : VUMPS ends after this many iterations even if
                             unconverged.
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

    RETURNS
    -------
    allout = mpslist, A_C, H_env.
            mpslist = [A_L, C, A_R] stores the MPS wavefunction.
            A_C and H_env are information needed to restart vumps.
    deltas (list of floats): List of the gradient values.
    """

    writer = make_writer(out_directory)
    delta = 0.1
    deltas = []
    d = H.shape[0]
    mpslist, A_C, fpoints = vumps_initialization(d, chi, H.dtype)

    writer.write("VUMPS! VUMPS! VUMPS/VUMPS/VUMPS/VUMPS! VUMPS!")
    H_env = solve_environment(mpslist, delta, fpoints, H, env_solver_params)
    E = obs.twositeexpect(mpslist, H)
    writer.write("Initial energy: ", E)
    writer.write("And so it begins...")
    iter_data = [mpslist, A_C, fpoints, H_env]
    for Niter in range(max_iter):
        Eold = E
        oldlist = copy.deepcopy(mpslist)
        iter_data, delta = vumps_iteration(iter_data, delta, H,
                                           heff_krylov_params,
                                           env_solver_params)
        dE, E, norm, B2 = _diagnostics(oldlist, Eold, H, iter_data)
        output(writer, Niter, delta, E, dE, norm, B2)
        deltas.append(delta)
        if delta <= gradient_tol:
            writer.write("Convergence achieved at iteration ", Niter)
            break

        if checkpoint_every is not None and Niter % checkpoint_every == 0:
            writer.write("Checkpointing...")
            writer.pickle(iter_data, Niter)

    if Niter == max_iter - 1:
        writer.write("Maximum iteration ", max_iter, "reached.")
    writer.write("Simulation finished. Pickling results.")
    writer.pickle(iter_data, Niter)
    return (iter_data, deltas)



