import copy
import os
import importlib

import numpy as np

from jax_vumps.writer import Writer
import jax_vumps.benchmark as benchmark
import jax_vumps.params as params


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
CONTRACTIONS_NAME = BACKEND_DIR_NAME + "contractions"


mps_linalg = importlib.import_module(MPS_LINALG_NAME)
heff = importlib.import_module(HEFF_NAME)
environment = importlib.import_module(ENVIRONMENT_NAME)
ct = importlib.import_module(CONTRACTIONS_NAME)


##########################################################################
# Functions to handle output.
##########################################################################
def ostr(string):
    """
    Truncates to two decimal places.  """
    return '{:1.2e}'.format(string)


def output(writer, Niter, delta, E, dE, norm, timing_data=None):
    """
    Does the actual outputting.
    """
    outstr = "N = " + str(Niter) + "| eps = " + ostr(delta)
    outstr += "| E = " + '{0:1.16f}'.format(E)
    outstr += "| dE = " + ostr(dE)
    # outstr += "| |B2| = " + ostr(B2)
    if timing_data is not None:
        outstr += "| dt= " + ostr(timing_data["Total"])
    writer.write(outstr)

    this_output = np.array([Niter, E, delta, norm])
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

    data_headers = ["N", "E", "|B|", "<psi>"]
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
def solve_environment(mpslist, delta, fpoints, H, env_params, H_env=None):
    timing = {}
    timing["Environment"] = benchmark.tick()
    if H_env is None:
        H_env = [None, None]

    lh, rh = H_env  # lowercase means 'from previous iteration'

    A_L, C, A_R = mpslist
    rL, lR = fpoints

    timing["LH"] = benchmark.tick()
    LH = environment.solve_for_LH(A_L, H, lR, env_params, delta, oldLH=lh)
    timing["LH"] = benchmark.tock(timing["LH"], dat=LH)

    timing["RH"] = benchmark.tick()
    RH = environment.solve_for_RH(A_R, H, rL, env_params, delta, oldRH=rh)
    timing["RH"] = benchmark.tock(timing["RH"], dat=RH)

    H_env = [LH, RH]
    timing["Environment"] = benchmark.tock(timing["Environment"], dat=RH)
    return (H_env, timing)


###############################################################################
# Gradient.
###############################################################################


def apply_gradient(iter_data, H, heff_krylov_params, gauge_via_svd):
    """
    Apply the MPS gradient.
    """
    timing = {}
    timing["Gradient"] = benchmark.tick()
    mpslist, a_c, fpoints, H_env, delta = iter_data
    a_l, c, a_r = mpslist
    rL, lR = fpoints
    LH, RH = H_env
    Hlist = [H, LH, RH]
    timing["HAc"] = benchmark.tick()
    _, A_C = heff.minimize_HAc(mpslist, a_c, Hlist, delta, heff_krylov_params)
    timing["HAc"] = benchmark.tock(timing["HAc"], dat=A_C)

    timing["Hc"] = benchmark.tick()
    _, C = heff.minimize_Hc(mpslist, Hlist, delta, heff_krylov_params)
    timing["Hc"] = benchmark.tock(timing["Hc"], dat=C)

    timing["Gauge Match"] = benchmark.tick()
    A_L, A_R = mps_linalg.gauge_match(A_C, C, svd=gauge_via_svd)
    timing["Gauge Match"] = benchmark.tock(timing["Gauge Match"], dat=A_L)

    timing["Loss"] = benchmark.tick()
    eL = mps_linalg.norm(A_C - ct.rightmult(A_L, C))
    eR = mps_linalg.norm(A_C - ct.leftmult(C, A_R))
    delta = max(eL, eR)
    timing["Loss"] = benchmark.tock(timing["Loss"], dat=delta)

    newmpslist = [A_L, C, A_R]
    timing["Gradient"] = benchmark.tock(timing["Gradient"], dat=C)
    return (newmpslist, A_C, delta, timing)


###############################################################################
# Main loop and friends.
###############################################################################
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
    fpoints = (L0, R0)
    mpslist = [A_L, C, A_R]
    return (mpslist, A_C, fpoints)


def vumps_iteration(iter_data, H, heff_params, env_params, gauge_via_svd):
    """
    One main iteration of VUMPS.
    """
    timing = {}
    timing["Iteration"] = benchmark.tick()
    mpslist, A_C, delta, grad_time = apply_gradient(iter_data, H, heff_params,
                                                    gauge_via_svd)
    timing.update(grad_time)
    fpoints = vumps_approximate_tm_eigs(mpslist[1])
    _, _, _, H_env, _ = iter_data
    H_env, env_time = solve_environment(mpslist, delta, fpoints, H,
                                        env_params, H_env=H_env)
    iter_data = [mpslist, A_C, fpoints, H_env, delta]
    timing.update(env_time)
    timing["Iteration"] = benchmark.tock(timing["Iteration"], dat=H_env[0])
    return (iter_data, timing)


def diagnostics(oldmpslist, mpslist, H, oldE):
    """
    Makes a few computations to output during a vumps run.
    """
    t0 = benchmark.tick()
    E = mps_linalg.twositeexpect(mpslist, H)
    dE = abs(E - oldE)
    norm = mps_linalg.mpsnorm(mpslist)
    tf = benchmark.tock(t0, dat=norm)
    return E, dE, norm, tf


def vumps(H, chi: int, delta_0=0.1,
          out_directory="./vumps",
          vumps_params=params.vumps_params(),
          heff_params=params.krylov_params(),
          env_params=params.gmres_params()
          ):
    """
    Find the ground state of a uniform two-site Hamiltonian
    using Variational Uniform Matrix Product States. This is a gradient
    descent method minimizing the distance between a given MPS and the
    best approximation to the physical ground state at its bond dimension.

    This interface function initializes vumps from a random initial state.

    PARAMETERS
    ----------
    H (array, (d, d, d, d)): The Hamiltonian whose ground state is to be found.
    chi (int)              : MPS bond dimension.
    delta_0 (float)        : Initial value for the gradient norm. The
                             convergence thresholds of the various solvers at
                             the initial step are proportional to this, via
                             coefficients in the Krylov and solver param dicts.

    The following arguments are bundled together by initialization functions
    in jax_vumps.params.

    vumps_params (dict)    : Hyperparameters for the vumps solver. Formed
                             by 'vumps_params'.
    heff_params (dict)     : Hyperparameters for an eigensolve of certain
                             'effective Hamiltonians'. Formed by
                             'krylov_params()'.
    env_params (dict)      : Hyperparameters for a linear solve that finds
                             the effective Hamiltonians. Formed by
                             'solver_params()'.

    RETURNS
    -------
    """

    writer = make_writer(out_directory)
    d = H.shape[0]
    mpslist, A_C, fpoints = vumps_initialization(d, chi, H.dtype)
    H_env, env_init_time = solve_environment(mpslist, delta_0,
                                             fpoints, H, env_params)
    iter_data = [mpslist, A_C, fpoints, H_env, delta_0]
    writer.write("Initial solve time: " + str(env_init_time["Environment"]))
    out = vumps_work(H, iter_data, vumps_params, heff_params,
                     env_params, writer)
    return out


def vumps_work(H, iter_data, vumps_params, heff_params, env_params, writer,
               Niter0=1):
    """
    Main work loop for vumps. Should be accessed via one of the interface
    functions above.

    PARAMETERS
    ----------
    H

    """
    checkpoint_every = vumps_params["checkpoint_every"]
    max_iter = vumps_params["max_iter"]

    t_total = benchmark.tick()
    mpslist, A_C, fpoints, H_env, delta = iter_data
    E = mps_linalg.twositeexpect(mpslist, H)
    writer.write("VUMPS! VUMPS! VUMPS/VUMPS/VUMPS/VUMPS! VUMPS!")
    writer.write("Initial energy: " + str(E))
    writer.write("Linalg backend: " + MPS_LINALG_NAME)
    writer.write("And so it begins...")
    for Niter in range(Niter0, vumps_params["max_iter"]+Niter0):
        dT = benchmark.tick()
        timing = {}
        oldE = E
        oldlist = copy.deepcopy(mpslist)
        iter_data, iter_time = vumps_iteration(iter_data, H, heff_params,
                                               env_params,
                                               vumps_params["gauge_via_svd"])
        mpslist, A_C, fpoints, H_env, delta = iter_data
        timing.update(iter_time)

        E, dE, norm, tD = diagnostics(oldlist, mpslist, H, oldE)
        timing["Diagnostics"] = tD
        timing["Total"] = benchmark.tock(dT, dat=iter_data[1])
        output(writer, Niter, delta, E, dE, norm, timing)

        if delta <= vumps_params["gradient_tol"]:
            writer.write("Convergence achieved at iteration " + str(Niter))
            break

        if checkpoint_every is not None and (Niter+1) % checkpoint_every == 0:
            writer.write("Checkpointing...")
            to_pickle = [H, iter_data, vumps_params, heff_params, env_params]
            to_pickle.append(Niter)
            writer.pickle(to_pickle, Niter)

    if Niter == max_iter - 1:
        writer.write("Maximum iteration " + str(max_iter) + " reached.")
    t_total = benchmark.tock(t_total, dat=mpslist[0])
    writer.write("The main loops took " + str(t_total) + " seconds.")
    writer.write("Simulation finished. Pickling results.")
    to_pickle = [H, iter_data, vumps_params, heff_params, env_params, Niter]
    writer.pickle(to_pickle, Niter)
    return (iter_data, timing)
