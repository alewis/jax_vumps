import jax
import jax.numpy as jnp

import jax_vumps.jax_backend.contractions as ct
import jax_vumps.jax_backend.lanczos as lz
import jax_vumps.jax_backend.arnoldi as arn 
from tensornetwork.backends.jax.jax_backend import JaxBackend

###############################################################################
# Effective Hamiltonians for A_C.
###############################################################################
#  @jax.tree_util.Partial
#  @jax.jit
#  def apply_HAc_for_solver(A_L, A_R, Hlist, A_C):
#      chi = A_L.shape[2]
#      C = v.reshape
#      newA_C = ct.apply_HAc(A_C, A_L, A_R, Hlist)
#      return newA_C
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
    return ev, newA_C


#  def minimize_HAc(mpslist, A_C, Hlist, delta, params):
#      """
#      The dominant (most negative) eigenvector of HAc.
#      """
#      A_L, C, A_R = mpslist
#      tol = params["tol_coef"]*delta
#      mv_args = [A_L, A_R, Hlist]
#      ev, newA_C = JaxBackend().eigsh_lanczos(apply_HAc_for_solver, mv_args,
#                                              initial_state=A_C,
#                                              numeig=1,
#                                              num_krylov_vecs=params["n_krylov"],
#                                              ndiag=params["max_restarts"],
#                                              tol=tol,
#                                              reorthogonalize=False)
#      ev = ev[0]
#      newA_C = newA_C[0]
#      newA_C = newA_C.reshape((A_C.shape))
#      return ev, newA_C


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
    return ev, newC


@jax.tree_util.Partial
@jax.jit
def apply_Hc_for_tn_solver(A_L, A_R, Hlist, C):
    newC = ct.apply_Hc(C, A_L, A_R, Hlist)
    return newC


def minimize_Hc_tensornetwork(mpslist, Hlist, delta, params):
    A_L, C, A_R = mpslist
    tol = params["tol_coef"]*delta
    mv_args = [A_L, A_R, Hlist]
    ev, newC = JaxBackend().eigsh_lanczos(apply_Hc_for_tn_solver, mv_args,
                                          initial_state=C,
                                          numeig=1,
                                          num_krylov_vecs=params["n_krylov"],
                                          ndiag=params["max_restarts"],
                                          tol=tol,
                                          reorthogonalize=False)
    ev = ev[0]
    newC = newC[0]
    newC = newC.reshape((C.shape))
    return ev, newC
