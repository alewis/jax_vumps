import jax
import jax.numpy as jnp

import jax_vumps.jax_backend.contractions as ct
import jax_vumps.jax_backend.mps_linalg as mps_linalg
import jax_vumps.jax_backend.gmres as gmres


def call_solver(matvec, matvec_args, hI, params, x0, tol):
    """
    Code used by both solve_for_RH and solve_for_LH to call the
    sparse solver. Solves matvec(*matvec_args, x) = hI for x.

    PARAMETERS
    ----------
    matvec (jax.tree_util.Partial): A function implementing the linear
                                    transform.
    matvec_args (list)            : List of constant arguments to matvec.
    hI (array)                    : The right hand side of the equation.
    params (dict)                 : Parameters for the solver.
    x0 (array)                    : Initial guess vector.
    tol (float)                   : Convergence threshold.

    RETURNS
    ------
    """
    if x0 is None:
        x0, = mps_linalg.random_tensors([hI.shape, ], dtype=hI.dtype)
    x0 = x0.flatten()

    if params["solver"] == "gmres":
        solver_out = gmres.gmres_m(matvec,
                                   matvec_args,
                                   hI.flatten(),
                                   x0,
                                   tol=tol,
                                   n_kry=params["n_krylov"],
                                   max_restarts=params["max_restarts"],
                                   verbose=True)
    elif params["solver"] == "lgmres":
        raise NotImplementedError("lgmres not implemented in Jax.")
    x, err_rel, n_iter, converged = solver_out
    #  x, info = lgmres(op,
    #                  hI.flatten(),
    #                  tol=tol,#params["env_tol"],
    #                  maxiter=params["maxiter"],
    #                  inner_m=params["inner_m"],
    #                  outer_k=params["outer_k"],
    #                  x0=x0)
    solution = x.reshape(hI.shape)
    return solution


###############################################################################
# LH
###############################################################################
@jax.tree_util.Partial
@jax.jit
def LH_matvec(lR, A_L, v):
    chi = A_L.shape[2]
    v = v.reshape((chi, chi))
    Th_v = ct.XopL_X(A_L, v)
    vR = ct.proj(v, lR)*jnp.eye(chi, dtype=A_L.dtype)
    v = v - Th_v + vR
    v = v.flatten()
    return v


@jax.jit
def prepare_for_LH_solve(A_L, H, lR):
    """
    Computes A and b in the A x = B to be solved for the left environment
    Hamiltonian. Separates things that can be Jitted from the rest of
    solve_for_LH.
    """
    hL_bare = ct.compute_hL(A_L, H)
    hL_div = ct.proj(hL_bare, lR)*jnp.eye(hL_bare.shape[0], dtype=A_L.dtype)
    hL = hL_bare - hL_div
    return hL


def solve_for_LH(A_L, H, lR, params, delta, oldLH=None,
                 dense=False):
    """
    Find the renormalized left environment Hamiltonian using a sparse
    solver.
    """
    tol = params["tol_coef"]*delta
    hL = prepare_for_LH_solve(A_L, H, lR)
    matvec_args = [lR, A_L]
    LH = call_solver(LH_matvec, matvec_args, hL, params, oldLH, tol)
    return LH


###############################################################################
# RH
###############################################################################
@jax.tree_util.Partial
@jax.jit
def RH_matvec(rL, A_R, v):
    chi = A_R.shape[2]
    v = v.reshape((chi, chi))
    Th_v = ct.XopR_X(A_R, v)
    Lv = ct.proj(rL, v)*jnp.eye(chi, dtype=A_R.dtype)
    v = v - Th_v + Lv
    v = v.flatten()
    return v


@jax.jit
def prepare_for_RH_solve(A_R, H, rL):
    """
    Computes A and b in the A x = B to be solved for the right environment
    Hamiltonian. Separates things that can be Jitted from the rest of
    solve_for_RH.
    """
    hR_bare = ct.compute_hR(A_R, H)
    hR_div = ct.proj(rL, hR_bare)*jnp.eye(hR_bare.shape[0], dtype=A_R.dtype)
    hR = hR_bare - hR_div
    return hR


def solve_for_RH(A_R, H, rL, params, delta, oldRH=None):
    """
    Find the renormalized right environment Hamiltonian using a sparse
    solver.
    """
    tol = params["tol_coef"]*delta
    hR = prepare_for_RH_solve(A_R, H, rL)
    matvec_args = [rL, A_R]
    RH = call_solver(RH_matvec, matvec_args, hR, params, oldRH, tol)
    # RHL = np.abs(ct.proj(rL, RH))
    # if RHL > 1E-6:
        # print("Warning: large <L|RH> = ", str(RHL))
    return RH
