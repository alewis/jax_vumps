import copy
import os
import importlib

import numpy as np
#import jax

from jax_vumps.writer import Writer
import jax_vumps.contractions as ct
import jax_vumps.observables as obs
import jax_vumps.benchmark as benchmark

import bhtools.tebd.vumps as old_vumps

try:
    ENVIRON_NAME = os.environ["LINALG_BACKEND"]
except KeyError:
    print("While importing vumps.py,")
    print("os.environ[\"LINALG_BACKEND\"] was undeclared; using NumPy.")
    ENVIRON_NAME = "numpy"

BACKEND_DIR_NAME = "jax_vumps." + ENVIRON_NAME + "_backend."
MPS_LINALG_NAME = BACKEND_DIR_NAME + "mps_linalg"
HEFF_NAME = BACKEND_DIR_NAME + "heff"
ENVIRONMENT_NAME = BACKEND_DIR_NAME + "environment"

mps_linalg = importlib.import_module(MPS_LINALG_NAME)
heff = importlib.import_module(HEFF_NAME)
environment = importlib.import_module(ENVIRONMENT_NAME)


#  if environ_name == "jax":
#      JAX_MODE = True
#  elif environ_name == "numpy":
#      JAX_MODE = False
#      mps_linalg_name = "jax_vumps.numpy_backend.mps_linalg"
#  else:
#    raise ValueError("Invalid LINALG_BACKEND ", environ_name)



#  def jit_toggle(f, static_argnums=None):
#      if JAX_MODE:
#          return jax.jit(f, static_argnums=static_argnums)
#      else:
#          return f


##########################################################################
# Functions to handle output.
##########################################################################
def ostr(string):
    """
    Truncates to four decimal places.  """
    return '{:1.4e}'.format(string)


def output(writer, Niter, delta, E, norm, B2, timing_data=None):
    """
    Does the actual outputting.
    """
    outstr = "N = " + str(Niter) + "| |B| = " + ostr(delta)
    outstr += "| E = " + '{0:1.16f}'.format(E)
    outstr += "| |B2| = " + ostr(B2)
    if timing_data is not None:
        outstr += "| dt= " + ostr(timing_data["Total"])
    writer.write(outstr)

    this_output = np.array([Niter, E, delta, B2, norm])
    writer.data_write(this_output)

    if timing_data is not None:
        writer.timing_write(Niter, timing_data)


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

    data_headers = ["N", "E", "dE", "|B|", "|B2|", "<psi>"]
    if outdir is None:
        return None
    timing_headers = ["N", "Total", "Diagnostics", "Iteration",
                      "Gradient", "HAc", "Hc", "Gauge Match", "Loss",
                      "Environment", "LH", "RH"]
    writer = Writer(outdir, data_headers=data_headers,
                    timing_headers=timing_headers)
    return writer


###############################################################################
# Effective environment.
###############################################################################
def solve_environment(mpslist, delta, fpoints, H, env_solver_params,
                      H_env=None):
    timing = {}
    timing["Environment"] = benchmark.tick()
    if H_env is None:
        H_env = [None, None]

    lh, rh = H_env  # lowercase means 'from previous iteration'

    A_L, C, A_R = mpslist
    rL, lR = fpoints

    timing["LH"] = benchmark.tick()
    LH = environment.solve_for_LH(A_L, H, lR, env_solver_params, delta,
                                  oldLH=lh)
    timing["LH"] = benchmark.tock(timing["LH"], dat=LH)

    timing["RH"] = benchmark.tick()
    RH = environment.solve_for_RH(A_R, H, rL, env_solver_params, delta,
                                  oldRH=rh)
    timing["RH"] = benchmark.tock(timing["RH"], dat=RH)

    H_env = [LH, RH]
    timing["Environment"] = benchmark.tock(timing["Environment"], dat=RH)
    return H_env, timing


###############################################################################
# Gradient.
###############################################################################
#@jit_toggle
def vumps_loss(A_L, A_C):
    """
    Norm of MPS gradient: see Appendix 4.
    """
    A_L_mat = mps_linalg.fuse_left(A_L)
    A_L_dag = A_L_mat.T.conj()
    N_L = mps_linalg.null_space(A_L_dag)
    N_L_dag = N_L.T.conj()
    A_C_mat = mps_linalg.fuse_left(A_C)
    B = N_L_dag @ A_C_mat
    Bnorm = mps_linalg.norm(B)
    return Bnorm


def apply_gradient(iter_data, delta, H, heff_krylov_params):
    """
    Apply the MPS gradient.
    """
    timing = {}
    timing["Gradient"] = benchmark.tick()
    mpslist, a_c, fpoints, H_env = iter_data
    a_l, c, a_r = mpslist
    rL, lR = fpoints
    LH, RH = H_env
    Hlist = [H, LH, RH]
    timing["HAc"] = benchmark.tick()
    A_C = heff.minimize_HAc(mpslist, a_c, Hlist, delta, heff_krylov_params)
    timing["HAc"] = benchmark.tock(timing["HAc"], dat=A_C)

    timing["Hc"] = benchmark.tick()
    C = heff.minimize_Hc(mpslist, Hlist, delta, heff_krylov_params)
    timing["Hc"] = benchmark.tock(timing["Hc"], dat=C)

    timing["Gauge Match"] = benchmark.tick()
    A_L, A_R = mps_linalg.gauge_match(A_C, C)
    timing["Gauge Match"] = benchmark.tock(timing["Gauge Match"], dat=A_L)

    timing["Loss"] = benchmark.tick()
    delta = vumps_loss(a_l, A_C)
    timing["Loss"] = benchmark.tock(timing["Loss"], dat=delta)

    newmpslist = [A_L, C, A_R]
    timing["Gradient"] = benchmark.tock(timing["Gradient"], dat=C)

    # SWAPS IN OLD CODE ##########################
    #  old_params = old_vumps.vumps_params()
    #  A_C = old_vumps.minimize_HAc(mpslist, a_c, Hlist, old_params)
    #  C = old_vumps.minimize_Hc(mpslist, Hlist, old_params)
    #  A_L, A_R = old_vumps.gauge_match(A_C, C)
    #  delta = vumps_loss(a_l, A_C)
    #  newmpslist = [A_L, C, A_R]
    # SWAPS IN OLD CODE ##########################
    return (newmpslist, A_C, delta, timing)


###############################################################################
# Main loop and friends.
###############################################################################
#@jit_toggle
def vumps_approximate_tm_eigs(C):
    """
    Returns the approximate transfer matrix dominant eigenvectors,
    rL ~ C^dag C, and lR ~ C Cdag = rLdag, both trace-normalized.
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
    A_1, = mps_linalg.random_tensors([(d, chi, chi)], dtype=dtype)

    A_L, _ = mps_linalg.qrpos(A_1)
    C, A_R = mps_linalg.lqpos(A_L)
    A_C = ct.rightmult(A_L, C)
    L0, R0 = vumps_approximate_tm_eigs(C)
    _, L = mps_linalg.tmeigs(A_R, direction="left", v0=L0)
    _, R = mps_linalg.tmeigs(A_L, direction="right", v0=R0)
    fpoints = (L0, R0)
    mpslist = [A_L, C, A_R]
    return (mpslist, A_C, fpoints)


def vumps_iteration(iter_data, delta, H, heff_krylov_params,
                    env_solver_params):
    """
    One main iteration of VUMPS.
    """
    timing = {}
    timing["Iteration"] = benchmark.tick()
    mpslist, A_C, fpoints, H_env = iter_data
    mpslist, A_C, delta, grad_time = apply_gradient(iter_data, delta, H,
                                                    heff_krylov_params)
    timing.update(grad_time)
    fpoints = vumps_approximate_tm_eigs(mpslist[1])
    H_env, env_time = solve_environment(mpslist, delta, fpoints, H,
                                        env_solver_params,
                                        H_env=H_env)
    timing.update(env_time)
    timing["Iteration"] = benchmark.tock(timing["Iteration"], dat=H_env[0])
    return ((mpslist, A_C, fpoints, H_env), delta, timing)


def diagnostics(oldmpslist, H, iter_data):
    """
    Makes a few computations to output during a vumps run.
    """
    t0 = benchmark.tick()
    mpslist, A_C, fpoints, H_env = iter_data
    E = obs.twositeexpect(mpslist, H)
    norm = obs.norm(mpslist)
    B2 = obs.B2_variance(oldmpslist, mpslist)
    tf = benchmark.tock(t0, dat=B2)
    return E, norm, B2, tf


def krylov_params(n_krylov=40, max_restarts=100, tol_coef=0.01):
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
            "tol_coef": tol_coef}


def solver_params(inner_m=30, outer_k=10, maxiter=100, tol_coef=0.01):
    """
    Bundles parameters for the (L)GMRES linear solver. These control the expense of finding the left and right environment Hamiltonians.  
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
    return {"inner_m": inner_m, "maxiter": maxiter, "outer_k": outer_k,
            "tol_coef": tol_coef}


def vumps(H, chi: int, gradient_tol: float, max_iter: int,
          delta_0=0.1,
          checkpoint_every=500,
          out_directory="./vumps",
          heff_krylov_params=krylov_params(),
          env_solver_params=solver_params()):
    """ Find the ground state of a uniform two-site Hamiltonian
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
    t_total = benchmark.tick()

    writer = make_writer(out_directory)
    delta = delta_0
    deltas = []
    d = H.shape[0]
    mpslist, A_C, fpoints = vumps_initialization(d, chi, H.dtype)
    writer.write("VUMPS! VUMPS! VUMPS/VUMPS/VUMPS/VUMPS! VUMPS!")
    writer.write("Linalg backend: " + MPS_LINALG_NAME)
    H_env, env_init_time = solve_environment(mpslist,
                                             delta, fpoints, H,
                                             env_solver_params)

    E = obs.twositeexpect(mpslist, H)
    writer.write("Initial energy: " + str(E))
    writer.write("Initial solve time: " + str(env_init_time["Environment"]))
    writer.write("And so it begins...")
    iter_data = [mpslist, A_C, fpoints, H_env]
    for Niter in range(max_iter):
        timing = {}
        timing["Total"] = benchmark.tick()
        oldlist = copy.deepcopy(mpslist)
        iter_data, delta, iter_time = vumps_iteration(iter_data, delta, H,
                                                      heff_krylov_params,
                                                      env_solver_params)
        timing.update(iter_time)

        E, norm, B2, tD = diagnostics(oldlist, H, iter_data)

        timing["Diagnostics"] = tD
        timing["Total"] = benchmark.tock(timing["Total"], dat=iter_data[1])
        output(writer, Niter, delta, E, norm, B2, timing)
        deltas.append(delta)
        if delta <= gradient_tol:
            writer.write("Convergence achieved at iteration " + str(Niter))
            break

        if checkpoint_every is not None and Niter % checkpoint_every == 0:
            writer.write("Checkpointing...")
            writer.pickle(iter_data, Niter)

    if Niter == max_iter - 1:
        writer.write("Maximum iteration " + str(max_iter) + " reached.")
    t_total = benchmark.tock(t_total, dat=mpslist[0])
    writer.write("The simulation took " + str(t_total) + " seconds.")
    writer.write("Simulation finished. Pickling results.")
    writer.pickle(iter_data, Niter)
    return (iter_data, deltas)
