from functools import partial
import numpy as np

import jax
import jax.numpy as jnp

import jax_dmrg.utils as utils
import jax_dmrg.map
import jax_dmrg.operations as ops

SOFTNORMTHRESH = 1E-10


def lz_params(
        ncv=4,
        lz_tol=1E-12,
        lz_maxiter=2
        ):
    params = {"ncv": ncv,
              "lz_tol": lz_tol,
              "lz_maxiter": lz_maxiter}
    return params


@jax.jit
def softnorm(v):
    return jnp.amax([jnp.linalg.norm(v), SOFTNORMTHRESH])


def softnormnp(v):
    return np.amax([np.linalg.norm(v), SOFTNORMTHRESH])


def dmrg_solve_hard(A, L, R, mpo, lz_params=None):
    """
    The local ground state step of single-site DMRG. matvec is hardcoded 
    in.
    """
    if lz_params is None:
        lz_params = lz_params()
    keys = ["ncv", "lz_tol", "lz_maxiter"]
    n_krylov, tol, maxiter = [lz_params[key] for key in keys]

    A_vec = jnp.ravel(A)
    E, eV = dmrg_minimum_eigenpair(mpo, L, R, A_vec)
    newA = eV.reshape(A.shape)
    return (E, newA)


@jax.jit
def dmrg_minimum_eigenpair(mpo, L, R, v0):
    psi = v0
    for it in range(N_ITER):
        E, psi = dmrg_eigenpair_iteration(mpo, L, R, psi)
    return (E, psi)


@jax.jit
def dmrg_eigenpair_iteration(mpo, L, R, psi):
    K, T = dmrg_tridiagonalize(mpo, L, R, psi)
    Es, eVsT = jnp.linalg.eigh(T)
    E = Es[0]
    min_eVT = eVsT[:, 0]
    psi = K @ min_eVT
    psi = psi / softnorm(psi)
    return (E, psi)


@jax.jit
def dmrg_matvec(mpo, L, R, psi):
    ML, chiL, _ = L.shape
    MR, chiR, _ = R.shape
    d = mpo.shape[3]
    A = psi.reshape((chiL, d, chiR))
    newA = ops.single_mpo_heff(mpo, L, R, A)
    #  LA = jnp.einsum('abe, ecd', L, A)
    #  LAM = jnp.einsum('eafd, ecbf', LA, mpo)
    #  newA = jnp.einsum('abde, dce', LAM, R)
    newv = newA.reshape((chiL*d*chiR))
    return newv


#@jax.tree_util.partial(jax.jit, static_argnums=(3,))
@jax.jit
def dmrg_tridiagonalize(mpo, L, R, v0):
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

    vs = []  # Krylov vectors.
    alphas = []  # Diagonal entries of T.
    betas = []  # Off-diagonal entries of T.
    betas.append(0.)
    vs.append(0)
    v = v0/softnorm(v0) + 0.j
    vs.append(v)
    for k in range(1, N_KRYLOV + 1):
    #for k in range(1, n_krylov + 1):
        v = dmrg_matvec(mpo, L, R, v)
        #v = A_matvec(A_data, vs[k])
        alpha = (vs[k] @ v).real
        alphas.append(alpha)
        v = v - betas[k - 1] * vs[k - 1] - alpha * vs[k]
        beta = softnorm(v).real
        betas.append(beta)
        vs.append(v / beta)

    K = jnp.array(vs[1:-1]).T
    alpha_arr = jnp.array(alphas)
    beta_arr = jnp.array(betas[1:-1])
    T = jnp.diag(alpha_arr) + jnp.diag(beta_arr, 1) + jnp.diag(beta_arr, -1)
    return (K, T)



def dmrg_solve(A, L, R, mpo, n_krylov, tol, maxiter):
    """
    The local ground state step of single-site DMRG.
    """
    mpo_map = jax_dmrg.map.SingleMPOHeffMap(mpo, L, R)
    A_vec = jnp.ravel(A)

    #  E, eV, err = minimum_eigenpair_jit(mpo_map.matvec, mpo_map.data,
    #                                     n_krylov, maxiter, tol,
    #                                     A_vec)
    E, eV, err = minimum_eigenpair(mpo_map, n_krylov, maxiter=maxiter, tol=tol,
                                   v0=A_vec)
    #print(err)
    newA = eV.reshape(A.shape)
    return (E, newA, err)

def matrix_optimize(A, n_krylov=32, tol=1E-6, rtol=1E-8, maxiter=10, v0=None,
                    verbose=False):
    """
    The minimum eigenpair of a dense Hermitian matrix A.
    """
    A_map = jax_dmrg.map.MatrixMap(A, hermitian=True)
    E, eV, err = minimum_eigenpair(A_map, n_krylov, tol=tol, rtol=rtol, v0=v0,
                                   maxiter=maxiter, verbose=verbose)

    return (E, eV, err)


@partial(jax.jit, static_argnums=(2, 3, 4))
def minimum_eigenpair_jit(A_mv, A_data, n_krylov, maxiter, tol, v0):
    """
    Find the algebraically minimum eigenpair of the Hermitian operator A_op
    using explicitly restarted Lanczos iteration.

    PARAMETERS
    ----------
    A_op: Hermitian operator.
    n_krylov: Size of Krylov subspace.
    tol: convergence threshold.
    maxiter: The program ends after this many iterations even if unconverged.
    v0: An optional initial vector.


    """
    v = v0
    for it in range(maxiter):
        E, v, _ = eigenpair_iteration(A_mv, A_data, v, n_krylov)
    return (E, v, 0)



def minimum_eigenpair(A_op, n_krylov, tol=1E-6, rtol=1E-9, maxiter=10,
                      v0=None, verbose=False):
    """
    Find the algebraically minimum eigenpair of the Hermitian operator A_op
    using explicitly restarted Lanczos iteration.

    PARAMETERS
    ----------
    A_op: Hermitian operator.
    n_krylov: Size of Krylov subspace.
    tol, rtol: Absolute and relative error tolerance at which convergence is
               declared.
    maxiter: The program ends after this many iterations even if unconverged.
    v0: An optional initial vector.


    """
    m, n = A_op.shape
    #verbose = True
    if m != n:
        raise ValueError("Lanczos requires a Hermitian matrix; your shape was",
                         A_op.shape)
    if v0 is None:
        v, = utils.random_tensors([(n,)], dtype=A_op.dtype)
    else:
        v = v0
    #verbose=True
    olderr = 0.
    A_mv = A_op.matvec
    A_data = A_op.data
    for it in range(maxiter):
        E, v, err = eigenpair_iteration(A_mv, A_data, v, n_krylov)
        #  if verbose:
        #      print("LZ Iteration: ", it)
        #      print("\t E=", E, "err= ", err)
        #  if jnp.abs(err - olderr) < rtol or err < tol:
        #      return (E, v, err)
        #  if not it % 3:
        #      olderr = err
    #if verbose:
        
    #print("Warning: Lanczos solve exited without converging.")
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
    err = 10
    #  Apsi = A_matvec(A_data, psi)
    #  err = jnp.linalg.norm(jnp.abs(E*psi - Apsi))
    return (E, psi, err)


@jax.jit
def trid_iter(tup, x, A_matvec, A_data):  # A_matvec, A_data, tup
    v_km1, v_k, beta_k = tup
    v_temp = A_matvec(A_data, v_k)
    alpha = v_k @ v_temp
    v = v_temp - beta_k * v_km1 - alpha * v_k
    beta = softnorm(v)
    v = v / beta
    carry = (v_k, v, beta)
    stack = (v_k, alpha, beta)
    return (carry, stack)

@partial(jax.jit, static_argnums=(3,))
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

    @jax.jit
    def this_iter(tup, x):
        return trid_iter(tup, x, A_matvec, A_data)

    #  def trid_iter(tup, x):  # A_matvec, A_data, tup
    #      v_km1, v_k, beta_k = tup
    #      v_temp = A_matvec(A_data, v_k)
    #      alpha = v_k @ v_temp
    #      v = v_temp - beta_k * v_km1 - alpha * v_k
    #      beta = jnp.linalg.norm(v)
    #      v = v / beta
    #      carry = (v_k, v, beta)
    #      stack = (v_k, alpha, beta)
    #      return (carry, stack)

    v0_km1 = jnp.zeros(v0.size, dtype=v0.dtype)
    _, tup = jax.lax.scan(this_iter, (v0_km1, v, 0.), None, length=n_krylov)
    vs, alphas, betas = tup

    K = vs.T
    # alpha_arr = jnp.array(alphas)
    # beta_arr = jnp.array(betas[1:-1])
    # T = jnp.diag(alpha_arr) + jnp.diag(beta_arr, 1) + jnp.diag(beta_arr, -1)
    T = jnp.diag(alphas) + jnp.diag(betas[:-1], 1) + jnp.diag(betas[:-1], -1)
    return (K, T)


def numpy_tridiagonalize(A_matvec, A_data, v0, n_krylov):
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

    vs = []  # Krylov vectors.
    alphas = []  # Diagonal entries of T.
    betas = []  # Off-diagonal entries of T.
    betas.append(0.)
    vs.append(0)
    v = v0/np.linalg.norm(v0) #softnormnp(v0)# + 0.j
    vs.append(v)

    for k in range(1, n_krylov + 1):
        v = A_matvec(A_data, vs[k])
        alpha = (vs[k] @ v) #.real
        v = v - betas[k - 1] * vs[k - 1] - alpha * vs[k]
        beta = np.linalg.norm(v)
        vs.append(v/beta)
        alphas.append(alpha)
        betas.append(beta)
    K = np.array(vs[1:-1]).T
    alpha_arr = np.array(alphas)
    beta_arr = np.array(betas[1:-1])
    T = np.diag(alpha_arr) + np.diag(beta_arr, 1) + np.diag(beta_arr, -1)

    return (K, T)
