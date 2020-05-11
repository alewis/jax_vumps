from functools import partial

import jax
import jax.numpy as jnp
from jax.ops import index_update, index


SOFTNORMTHRESH = 1E-5


@jax.jit
def softnorm(v):
    return jnp.amax([jnp.linalg.norm(v), SOFTNORMTHRESH])


def minimum_eigenpair(matvec, matvec_args, n_krylov, tol=1E-6, maxiter=10,
                      v0=None, verbose=False):
    """
    Find the algebraically minimum eigenpair of the Hermitian operator A_op
    using explicitly restarted Lanczos iteration.

    PARAMETERS
    ----------
    matvec: Hermitian operator.
    matvec_args: Fixed arguments to matvec.
    n_krylov: Size of Krylov subspace.
    tol: Error tolerance.
    maxiter: The program ends after this many iterations even if unconverged.
    v0: Guess vector.
    """
    if v0 is None:
        raise NotImplementedError("Must supply v0.")
    v = v0
    for it in range(maxiter):
        E, v, err = eigenpair_iteration(matvec, matvec_args, v, n_krylov)
        #  print("LZ Iteration: ", it)
        #  print("\t E=", E, "err= ", err)
        if err < tol:
            return (E, v, err)
    if verbose:
        print("Warning: Lanczos solve exited without converging, err=", err)
    return (E, v, err)


@partial(jax.jit, static_argnums=(3,))
def eigenpair_iteration(A_matvec, A_data, v, n_krylov):
    """
    Performs one iteration of the explicitly restarted Lanczos method.
    """
    K, T = tridiagonalize(A_matvec, A_data, v, n_krylov)
    Es, eVsT = jnp.linalg.eigh(T)
    E = Es[0]
    min_eVT = eVsT[:, 0]
    psi = K @ min_eVT
    psi = psi / softnorm(psi)
    Apsi = A_matvec(*A_data, psi)
    err = jnp.linalg.norm(jnp.abs(E*psi - Apsi))
    return (E, psi, err)


@jax.jit
def body_modified_gram_schmidt(i, vals):
    vector, krylov_vectors = vals
    v = krylov_vectors[i, :]
    vector -= jax.numpy.vdot(v, vector).reshape(vector.shape)
    return [vector, krylov_vectors]



def trid_iter(tup, k, A_matvec, A_data):
    V, this_v = tup
    this_norm = jnp.linalg.norm(this_v)
    normalized_v = this_v / this_norm
    # reorth here
    Av = A_matvec(*A_data, normalized_v)
    alpha = jnp.vdot(this_v, Av)
    new_v = Av - normalized_v * alpha - V[:, k-1] * this_norm
    V = index_update(V, index[:, k], normalized_v)
    carry = (V, new_v)
    stack = (alpha, this_norm)
    return (carry, stack)


#  @jax.jit
#  def trid_iter(tup, k, A_matvec, A_data):  # A_matvec, A_data, tup
#      V, v_k, beta_k = tup
#      v_km1 = V[:, k-1]

#      v_temp = A_matvec(*A_data, v_k)
#      alpha = v_k.conj() @ v_temp
#      v = v_temp - beta_k * v_km1 - alpha * v_k
#      beta = softnorm(v)
#      v = v / beta
#      V = index_update(V, index[:, k], v_k)

#      #carry = (V, v_k, v, beta)
#      carry = (V, v, beta)
#      stack = (alpha, beta)
#      return (carry, stack)


#@partial(jax.jit, static_argnums=(3,))
def tridiagonalize(A_matvec, A_data, v0, n_krylov):
    """
    Lanczos tridiagonalization. A_matvec and A_data collectively represent
    a Hermitian
    linear map of a length-n vector with initial value v0 onto its vector space
    v = A_matvec(A_data, v0). Returns an n x n_krylov vector V that
    orthonormally spans the Krylov space of A, and an symmetric, real,
    and tridiagonal matrix T = V^dag A V of size n_krylov x n_krylov.

    PARAMETERS
    ----------
    A_matvec, A_data: represent the linear operator.
    v0              : a length-n vector.
    n_krylov        : size of the krylov space.

    RETURNS
    -------
    K (n, n_krylov) : basis of the Krylov space.
    T (n_krylov, n_krylov) : Tridiagonal projection of A onto V.
    """

    v = v0/softnorm(v0)

    #@jax.jit
    def this_iter(tup, x):
        return trid_iter(tup, x, A_matvec, A_data)

    #v0_km1 = jnp.zeros(v0.size, dtype=v0.dtype)
    V = jnp.zeros((v0.size, n_krylov+1), dtype=v0.dtype)
    #V = index_update(V, index[:, 0], v)

    carry, stack = jax.lax.scan(this_iter, (V, v),
                                xs=jnp.arange(n_krylov)+1)
    #V, alphas, betas = tup
    Vs, _ = carry
    print(Vs)
    alphas, betas = stack

    K = Vs[:, 1:]#.T
    T = jnp.diag(alphas) + jnp.diag(betas[:-1], 1) + jnp.diag(betas[:-1], -1)
    return (K, T)
