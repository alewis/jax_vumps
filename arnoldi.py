from typing import Sequence, Callable
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from jax.ops import index, index_update


@partial(jax.jit, static_argnums=(2,))
def arnoldi_krylov_jit(A_mv: Callable,
                       A_args: Sequence,
                       n_kry: int,
                       v0):
    """
    Given an (m x m) matrix A, a vector v0, and a dimension n_kry, finds
    an orthonormal basis on the order-(n_kry+1) Krylov space defined by
    (A, v0) as span{v0, A@v0, A^2@v0, .... A^(n_kry)@v0}.

    This orthonormal basis is found by the Arnoldi method: v0 is repeatedly
    multiplied by A, and each result orthogonalized against its ancestors
    using the stabilized Gram-Schmidt procedure. The basis vectors are
    stored in the columns of an (m x n_kry+1) matrix V.

    The Arnoldi process also generates an (n_kry+1, n_kry) upper-Hessenberg
    matrix H. The submatrix H[:-1, :] is the projection of A onto the
    first n_kry columns of V, H[:-1, :] = (V[:, :-1])^T @ A @ V[:, :-1].
    The last row of H is zero except for H[-1, -1] = norm(V[:, -1]).

    A is represented as a function y=A(*A_args, x) implementing a linear map
    y = A@x. This is possible because A is never modified during the procedure.

    *IMPORTANT*: we must have 1 <= n_kry < v0.size, but this is NOT ENFORCED
                 because doing so is incompatible with Jit.

    PARAMETERS
    ----------
    A_mv, Callable: Function A_mv(*A_args, x) returning y = A@x. This must
                    be a Jax type, which can be achieved by wrapping it in
                    tree_utils.Partial.
    A_args, List  : List containing any arguments to A_mv besides x. Only
                    positional arguments are allowed, and each must be a
                    Jax type.
    n_kry, Int    : The dimensions of the Krylov subspace. We must have
                    1 < n_kry < v0.size -1, but this is 
                    *NOT ENFORCED*
                    since it can't be in a Jitted function.
    v0 (N,) array : Vector defining the Krylov subspace.


    RETURNS
    -------
    V (m x n_kry+1) : Columns are an orthonormal basis of the Krylov
                      subspace.
    H (n_kry+1, n_kry:): Upper Hessenberg projection of A onto H, plus:w
                         a diagonal entry ||V[:, -1]||.
    """
    dtype = v0.dtype
    m = v0.shape[0]
    V = jnp.zeros((m, n_kry + 1), dtype=dtype)

    v = v0 / jnp.linalg.norm(v0)  # Normalize the input vector.
    V = index_update(V, index[:, 0], v)  # Use it as the first Krylov vector.

    def this_arnoldi_scan_function(carry, x):
        return _arnoldi_scan_function(carry, x, A_mv, A_args)
    carry, stack = jax.lax.scan(this_arnoldi_scan_function,
                                (V, v),
                                jnp.arange(n_kry))
    V, _ = carry
    H, h_off = stack
    H = H.T + jnp.diag(h_off, -1)[:n_kry+1, :n_kry]
    return (V, H)


@jax.jit
def _arnoldi_scan_function(carry, k, A_mv, A_args):
    """
    Main loop of arnoldi_krylov in a jax.lax.scan - friendly format.
    k is the current iteration index. carry is V, v; the second being
    the latest value of v. stack is hs, v_norm; v_norm is the first
    lower off diagonal of the eventual H, and hs is the upper triangle.
    """
    eps = 1E-8
    V, v_old = carry
    r = A_mv(*A_args, v_old)
    v_new, hs = gs_orthogonalize(V, r)
    v_norm = jnp.linalg.norm(v_new)
    switch = v_norm > eps

    #  Normalize v unless it is the zero vector.
    v_new = jax.lax.cond(switch,
                         (v_new, v_norm), lambda x: x[0] / x[1],
                         v_new, lambda x: jnp.zeros(x.size, dtype=x.dtype),
                         )
    V = index_update(V, index[:, k+1], v_new)
    newcarry = (V, v_new)
    stack = (hs, v_norm)
    return newcarry, stack


@jax.jit
def gs_orthogonalize(V, r):
    """
    Orthonormalize r against the vectors in the columns of V using
    the stablized Gram-Schmidt procedure. More specifically, given
    V whose columns form an orthonormal basis {v0, v1, ...} and some
    other vector r, return r_new so that {v0, v1, ..., r_new} is an
    orthonormal basis on span{v0, v1, ..., r}.

    PARAMETERS
    ----------
    V, array-like (N, n): Columns are the basis vectors to be orthonormalized
                          against. They are assumed already orthonormal.
    r, array-like (N,)  : The vector to orthonormalized against V.


    RETURNS
    -------
    r_new, array-like (N,) : Orthonormal to the columns of  V, such that
                             {V, r_new} spans
                             the same space as {V, r}.
    hs, array-like (n,)  : Projections of the {v} onto successive r_new during
                           the procedure.
    """
    hs = jnp.zeros(r.size, r.dtype)
    r, hs = jax.lax.scan(_gs_step, r, V.T)
    return (r, hs)


@jax.jit
def _gs_step(r, v_i):
    """
    Performs one iteration of the stabilized Gram-Schmidt procedure, with
    r to be orthonormalized against {v} = {v_0, v_1, ...}.
    """
    h_i = jnp.vdot(v_i, r)
    r_i = r - h_i * v_i
    return r_i, h_i


def arnoldi_krylov_numpy(A, b, n: int):
    """
    This is the Wikipedia implementation of the Arnoldi process, for testing.
    It has been slightly modified to make A a linear operator and to
    account for varying dtypes.

    Computes a basis of the (n + 1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^n b}.

    Arguments
      A: m × m LinearOperator
      b: initial vector (length m)
      n: dimension of Krylov subspace, must be >= 1

    Returns
      Q: m x (n + 1) array, the columns are an orthonormal basis of the
        Krylov subspace.
      h: (n + 1) x n array, A on basis Q. It is upper Hessenberg. 
    """
    m = A.shape[0]
    dtype = b.dtype
    h = np.zeros((n + 1, n), dtype=dtype)
    Q = np.zeros((m, n + 1), dtype=dtype)
    q = b / np.linalg.norm(b)  # Normalize the input vector
    Q[:, 0] = q  # Use it as the first Krylov vector

    for k in range(n):
        #v = A.dot(q)  # Generate a new candidate vector
        v = A(q)
        for j in range(k + 1):  # Subtract the projections on previous vectors
            h[j, k] = np.dot(Q[:, j].conj(), v)
            v = v - h[j, k] * Q[:, j]

        h[k + 1, k] = np.linalg.norm(v)
        eps = 1e-7  # If v is shorter than this threshold it is the zero vector
        if h[k + 1, k] > eps:  # Add the produced vector to the list, unless
            q = v / h[k + 1, k]  # the zero vector is produced.
            Q[:, k + 1] = q
        else:  # If that happens, stop iterating.
            return Q, h
    return Q, h

#  def arnoldi_numpy(A_mv: Callable,
#                    A_args: Sequence,
#                    n_kry: int,
#                    v0):
#      """
#      Basic implementation of the Arnoldi process on numpy arrays for testing.
#      """





#  def arnoldi_krylov_orig(A_mv: Callable,
#                     A_args: Sequence,
#                     n_kry: int,
#                     v0):
#      """
#      Given an (m x m) matrix A, a vector v0, and a dimension n_kry, finds
#      an orthonormal basis on the order-(n_kry+1) Krylov space defined by
#      (A, v0) as span{v0, A@v0, A^2@v0, .... A^(n_kry)@v0}.

#      This orthonormal basis is found by the Arnoldi method: v0 is repeatedly
#      multiplied by A, and each result orthogonalized against its ancestors
#      using the stabilized Gram-Schmidt procedure. The basis vectors are
#      stored in the columns of an (m x n_kry+1) matrix V.

#      The Arnoldi process also generates an (n_kry+1, n_kry) upper-Hessenberg
#      matrix H. The submatrix H[:-1, :] is the projection of A onto the
#      first n_kry columns of V, H[:-1, :] = (V[:, :-1])^T @ A @ V[:, :-1].
#      The last row of H is zero except for H[-1, -1] = norm(V[:, -1]).

#      A is represented as a function y=A(*A_args, x) implementing a linear map
#      y = A@x. This is possible because A is never modified during the procedure.


#      PARAMETERS
#      ----------
#      A_mv, Callable: Function A_mv(*A_args, x) returning y = A@x. This must
#                      be a Jax type, which can be achieved by wrapping it in
#                      tree_utils.Partial.
#      A_args, List  : List containing any arguments to A_mv besides x. Only
#                      positional arguments are allowed, and each must be a
#                      Jax type.
#      n_kry, Int    : The dimensions of the Krylov subspace, must be >= 1.
#      v0 (m,) array : Vector defining the Krylov subspace.


#      RETURNS
#      -------
#      V (m x n_kry+1) : Columns are an orthonormal basis of the Krylov
#                        subspace.
#      H (n_kry+1, n_kry:): Upper Hessenberg projection of A onto H, plus:w
#                           a diagonal entry ||V[:, -1]||.
#      """
#      dtype = v0.dtype
#      m = v0.shape[0]
#      H = jnp.zeros((n_kry + 1, n_kry), dtype=dtype)
#      V = jnp.zeros((m, n_kry + 1), dtype=dtype)

#      v = v0 / jnp.linalg.norm(v0)  # Normalize the input vector.
#      V = index_update(V, index[:, 0], v)  # Use it as the first Krylov vector.
#      n_eff = n_kry
#      for k in range(n_kry):
#          r = A_mv(*A_args, v)  # Generate a new candidate vector.
#          for i in range(k+1):
#              hik = jnp.vdot(V[:, i], r)
#              r = r - hik @ V[:, i]
#              H = index_update(H, index[i, k], hik)
#          r_norm = jnp.linalg.norm(r)
#          H = index_update(H, index[k+1, k], r_norm)

#          #  The conditional that follows is equivalent to:
#          #  if r_norm > eps:
#          #      v = r / r_norm
#          #  else:
#          #      v = jnp.zeros(v.size)
#          switch = r_norm > 1E-8
#          v, n_eff = jax.lax.cond(switch,
#                                  r, lambda x: x[0] / x[1],
#                                  r, lambda x: jnp.zeros(v.size, dtype=dtype))
#          V = index_update(V, index[:, k+1], v)
#      return (V, H)
