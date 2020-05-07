import jax

import jax_vumps.jax_backend.contractions as ct
import jax_vumps.jax_backend.lanczos as lz

###############################################################################
# Effective Hamiltonians for A_C.
###############################################################################
@jax.tree_util.Partial
@jax.jit
def apply_HAc_for_solver(A_L, A_R, Hlist, v):
    A_C = v.reshape(A_L.shape)
    newA_C = ct.apply_HAc(A_C, A_L, A_R, Hlist)
    newv = newA_C.flatten()
    return newv


def minimize_HAc(mpslist, A_C, Hlist, delta, params):
    """
    The dominant (most negative) eigenvector of HAc.
    """
    A_L, C, A_R = mpslist
    tol = params["tol_coef"]*delta
    lzout = lz.minimum_eigenpair(apply_HAc_for_solver, [A_L, A_R, Hlist],
                                 params["n_krylov"],
                                 maxiter=params["max_restarts"], tol=tol,
                                 v0=A_C.flatten())
    ev, newA_C, err = lzout
    newA_C = newA_C.reshape((A_C.shape))
    return newA_C.real


###############################################################################
# Effective Hamiltonians for C.
###############################################################################
@jax.tree_util.Partial
@jax.jit
def apply_Hc_for_solver(A_L, A_R, Hlist, v):
    chi = A_L.shape[2]
    C = v.reshape((chi, chi))
    newC = ct.apply_Hc(C, A_L, A_R, Hlist)
    newv = newC.flatten()
    return newv


def minimize_Hc(mpslist, Hlist, delta, params):
    """
    The dominant (most negative) eigenvector of Hc.
    """
    A_L, C, A_R = mpslist
    tol = params["tol_coef"]*delta
    lzout = lz.minimum_eigenpair(apply_Hc_for_solver, [A_L, A_R, Hlist],
                                 params["n_krylov"],
                                 maxiter=params["max_restarts"], tol=tol,
                                 v0=C.flatten())
    ev, newC, err = lzout
    newC = newC.reshape((C.shape))
    return newC.real
