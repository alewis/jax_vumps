"""
Low-level tensor network manipulations for single-site finite chain DMRG.

Adam GM Lewis
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax_dmrg.errors as errors
import jax_dmrg.utils as utils
import tensornetwork#.ncon
from scipy.sparse.linalg import LinearOperator



###############################################################################
# CONTRACTIONS
###############################################################################



def energy(L, R, mpo_chain, mps_chain):
    for mps, mpo in zip(mps_chain, mpo_chain):
        L = XopL(L, mpo, mps)
    E = jnp.einsum("abc, abc", L, R)
    return E


def norm(mps_chain):
    L = XnoL(mps_chain[0])
    for mps in mps_chain[1:]:
        L = XL(L, mps)
    n = jnp.einsum("aa", L)
    return n





"""
matvecs
"""
def matmat(mv, mv_args, X):
    """
    Does A@X for X a matrix, given mv implementing A@x for x a vector.
    """
    return jnp.hstack([mv(*mv_args, col.reshape(-1, 1)) for col in X.T])


@jax.tree_util.Partial
@jax.jit
def matrix_matvec(A, x):
    return A@x


def numpy_matrix_matvec(A, x):
    return A@x


def numpy_matrix_linop(A):
    op = LinearOperator(shape=A.shape,
                        matvec=lambda x: numpy_matrix_matvec(A, x))
    return op
