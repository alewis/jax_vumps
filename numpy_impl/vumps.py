import copy

import numpy as np
import scipy as sp

import jax
import jax.numpy as jnp

import tensornetwork as tn

import jax_vumps.writer.Writer as Writer
import jax_vumps.contractions as ct
import jax_vumps.mps_linalg as mps_linalg
import jax_vumps.observables as obs
import jax_vumps.utils as utils
import jax_vumps.environment
import jax_vumps.heff


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

    this_output = [Niter, E, dE, delta, B2, norm]
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


###############################################################################
# Main loop and friends.
###############################################################################
def vumps_approximate_tm_eigs(C):
    """
    Returns the approximate transfer matrix dominant eigenvectors,
    rL ~ C^\dag C, and lR ~ C C\dag = rL\dag, both trace-normalized.
    """
    rL = np.dot(np.conj(C.T), C)
    rL /= np.trace(rL)
    lR = rL.T.conj()
    return (rL, lR)


def vumps_initialization(d: int, chi: int, dtype=jnp.float32):
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
    A_1, A_2 = utils.random_tensors([(d, chi, chi), (d, chi, chi)],
                                    dtype=dtype)

    A_L, C_1 = mps_linalg.qrpos(A_1)
    C_2, A_R = mps_linalg.lqpos(A_2)
    C = C_1@C_2
    A_C = ct.rightmult(A_L, C)
    fpoints = vumps_approximate_tm_eigs(C)
    mpslist = [A_L, C, A_R]
    return (mpslist, A_C, fpoints)


def vumps_iteration(iter_data, delta, H, heff_krylov_params, tm_krylov_params,
                    env_solver_params):
    """
    One main iteration of VUMPS.
    """
    mpslist, A_C, fpoints, H_env = iter_data
    mpslist, A_C, delta = jax_vumps.heff.apply_gradient(iter_data, delta, H,
                                                        heff_krylov_params)
    fpoints = vumps_approximate_tm_eigs(mpslist[1])
    H_env = jax_vumps.environment.solve(mpslist, delta, fpoints, H,
                                        env_solver_params, H_env=H_env)
    return (mpslist, A_C, delta, fpoints, H_env)


def _diagnostics(oldmpslist, Eold, H, iter_data):
    """
    Makes a few computations to output during a vumps run.
    """
    mpslist, A_C, delta, fpoints, H_env = iter_data
    E = obs.twositeexpect(mpslist, H)
    dE = E - Eold
    norm = obs.norm(mpslist)
    B2 = obs.B2_variance(oldmpslist, mpslist)
    return dE, E, norm, B2


def krylov_params(n_krylov=40, max_restarts=10, tol_coef=0.1):
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
    return {"n_krylov: ": n_krylov, "max_restarts": max_restarts,
            "tol_coef: ": tol_coef}


def solver_params(n_krylov=40, max_restarts=10, tol_coef=0.1):
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
    """
    return {"n_krylov: ": n_krylov, "max_restarts": max_restarts,
            "tol_coef: ": tol_coef}


def vumps(H, chi: int, gradient_tol: float, max_iter: int,
          delta_0=0.1,
          checkpoint_every=500,
          out_directory="./vumps",
          tm_krylov_params=krylov_params(),
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
    tm_krylov_params (dict): Hyperparameters for an eigensolve of the MPS
                             transfer matrix. Formed by 'krylov_params()'.
    heff_krylov_params     : Hyperparameters for an eigensolve of certain
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
    H_env = jax_vumps.environment.solve(mpslist, fpoints, H, delta,
                                        env_solver_params)
    E = obs.twositeexpect(mpslist, H)
    writer.write("Initial energy: ", E)
    writer.write("And so it begins...")
    iter_data = [mpslist, A_C, fpoints, H_env]
    for Niter in range(max_iter):
        Eold = E
        oldlist = copy.deepcopy(mpslist)
        iter_data, delta = vumps_iteration(iter_data, delta, H,
                                           heff_krylov_params,
                                           tm_krylov_params, env_solver_params)
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



