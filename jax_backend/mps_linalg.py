"""
Functions that operate upon MPS tensors as if they were matrices.
These functions are not necessarily backend-agnostic.
"""
import numpy as np
import time
from functools import partial

import jax
import jax.numpy as jnp

import jax_vumps.jax_backend.contractions as ct


def random_tensors(shapes, seed=None, dtype=np.float32):
    """
    Returns a list of Gaussian random tensors, one for (and with) each shape
    in shapes. A random seed may optionally be specified; otherwise the
    system time is used.

    PARAMETERS
    ----------
    shapes: A list of input shapes.
    seed (default time.time()) : The random seed.
    dtype : dtype of tensors.

    RETURNS
    -------
    tensors : A list of random tensors of the given dtype, one respectively
              for each shape.
    """
    if seed is None:
        seed = int(time.time())
    key = jax.random.PRNGKey(seed)
    tensors = []
    for shape in shapes:
        key, subkey = jax.random.split(key)
        tensor = jax.random.normal(key, shape=shape, dtype=dtype)

        if (dtype == np.complex64 or dtype == np.complex128 or
           dtype == jnp.complex64 or dtype == jnp.complex128):

            tensor_i = jax.random.normal(key, shape=shape, dtype=dtype)
            tensor = tensor + 1.0j*tensor_i
        tensors.append(tensor)
    return tensors


def frobnorm(A, B=None):
    """
    The Frobenius norm of the difference between A and B, divided by the
    number of entries in A.
    """
    if B is None:
        B = np.zeros(A.shape)
    ans = (1./A.size)*norm(jnp.abs(A.ravel()-B.ravel()))
    return ans


@partial(jax.jit, static_argnums=(2,))
def sortby(es, vecs, mode="LM"):
    """
    The vector 'es' is sorted,
    and the i's in 'vecs[:, i]' are sorted in the same way. This is done
    by returning new, sorted arrays (not in place). 'Mode' may be 'LM' (sorts
    from largest to smallest magnitude) or 'SR' (sorts from most negative
    to most positive real part).
    """
    sortidx = jax.lax.cond(mode == "LM",
                           es, lambda x: jnp.abs(es).argsort()[::-1],
                           es, lambda x: es.real.argsort()
                           )
    #  if mode == "LM":
    #      sortidx = jnp.abs(es).argsort()[::-1]
    #  elif mode == "SR":
    #      sortidx = (es.real).argsort()
    essorted = es[sortidx]
    vecsorted = vecs[:, sortidx]
    return essorted, vecsorted


@jax.jit
def sortbyLM(es, vecs):
    sortidx = jnp.abs(es).argsort()[::-1]
    essorted = es[sortidx]
    vecsorted = vecs[:, sortidx]
    return essorted, vecsorted


@jax.jit
def fuse_left(A):
    """
    Joins the left bond with the physical index.
    """
    oldshp = A.shape
    d, chiL, chiR = oldshp
    A = A.reshape(d*chiL, chiR)
    return A


@partial(jax.jit, static_argnums=(1,))
def unfuse_left(A, shp):
    """
    Reverses fuse_left.
    """
    return A.reshape(shp)


@jax.jit
def fuse_right(A):
    """
    Joins the right bond with the physical index.
    """
    oldshp = A.shape
    d, chiL, chiR = oldshp
    A = A.transpose((1, 0, 2)).reshape((chiL, d*chiR))
    return A


@partial(jax.jit, static_argnums=(1,))
def unfuse_right(A, shp):
    """
    Reverses fuse_right.
    """
    d, chiL, chiR = shp
    A = A.reshape((chiL, d, chiR)).transpose((1, 0, 2))
    return A


@jax.jit
def norm(A):
    return jnp.linalg.norm(A)


@jax.jit
def trace(A):
    return jnp.trace(A)



###############################################################################
# QR
###############################################################################
@jax.jit
def qrpos(mps):
    """
    Reshapes the (d, chiL, chiR) MPS tensor into a (d*chiL, chiR) matrix,
    and computes its QR decomposition, with the phase of R fixed so as to
    have a non-negative main diagonal. A new left-orthogonal
    (chiL, d, chiR) MPS tensor (reshaped from Q) is returned along with
    R.

    In addition to being phase-adjusted, R is normalized by division with
    its L2 norm.

    PARAMETERS
    ----------
    mps (array-like): The (d, chiL, chiR) MPS tensor.

    RETURNS
    -------
    mps_L, R: A left-orthogonal (d, chiL, chiR) MPS tensor, and an upper
              triangular (chiR x chiR) matrix with a non-negative main
              diagonal such that mps = mps_L @ R.
    """
    d, chiL, chiR = mps.shape
    mps_mat = fuse_left(mps)
    Q, R = jnp.linalg.qr(mps_mat)
    phases = jnp.sign(jnp.diag(R))
    Q = Q*phases
    R = phases.conj()[:, None] * R
    R = R / norm(R)
    mps_L = unfuse_left(Q, mps.shape)
    return (mps_L, R)


@jax.jit
def lqpos(mps):
    """
    Reshapes the (d, chiL, chiR) MPS tensor into a (chiL, d*chiR) matrix,
    and computes its LQ decomposition, with the phase of L fixed so as to
    have a non-negative main diagonal. A new right-orthogonal
    (d, chiL, chiR) MPS tensor (reshaped from Q) is returned along with
    L.
    In addition to being phase-adjusted, L is normalized by division with
    its L2 norm.

    PARAMETERS
    ----------
    mps (array-like): The (d, chiL, chiR) MPS tensor.

    RETURNS
    -------
    L, mps_R:  A lower-triangular (chiL x chiL) matrix with a non-negative
               main-diagonal, and a right-orthogonal (d, chiL, chiR) MPS
               tensor such that mps = L @ mps_R.
    """
    d, chiL, chiR = mps.shape
    mpsT = mps.transpose((0, 2, 1))
    Qdag, Ldag = qrpos(mpsT)
    Q = Qdag.T.conj()
    L = Ldag.T.conj()
    mps_R = unfuse_right(Q, mps.shape)
    return (L, mps_R)


@jax.jit
def null_space(A):
    """
    The scipy code to compute the null space of a matrix.
    """
    u, s, vh = jnp.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    rcond = jnp.finfo(s.dtype).eps * max(M, N)
    tol = jnp.amax(s) * rcond
    num = jnp.sum(s > tol, dtype=int)
    Q = vh[num:, :].T.conj()
    return Q


@jax.jit
def mps_null_spaces(mpslist):
    """
    Return matrices spanning the null spaces of A_L and A_R, and
    the hermitian conjugates of these, reshaped into rank
    3 tensors.
    """
    AL, C, AR = mpslist
    d, chi, _ = AL.shape
    NLshp = (d, chi, (d-1)*chi)
    ALdag = fuse_left(AL).T.conj()
    NLm = null_space(ALdag)
    NL = NLm.reshape(NLshp)

    ARmat = fuse_right(AR)
    NRm_dag = null_space(ARmat)
    NRm = NRm_dag.conj()
    NR = NRm.reshape((d, chi, (d-1)*chi))
    NR = NR.transpose((0, 2, 1))
    return (NL, NR)


#  def gauge_match_svd(A_C, C):
#      AcC = ct.rightmult(A_C, C.T.conj())
#      AcC = fuse_left(AcC)
#      Ul, Sl, Vldag = jnp.linalg.svd(AcC, full_matrices=False)
#      A_L = Ul @ Vldag
#      A_L = unfuse_left(A_L, A_C.shape).astype(A_C.dtype)

#      CAc = ct.leftmult(C.T.conj(), A_C)
#      CAc = fuse_right(CAc)
#      Ur, Sr, Vrdag = jnp.linalg.svd(CAc, full_matrices=False)
#      A_R = Ur @ Vrdag
#      A_R = unfuse_right(A_R, A_C.shape).astype(A_C.dtype)
#      return (A_L, A_R)


@jax.jit
def gauge_match(A_C, C):
    """
    Return approximately gauge-matched A_L and A_R from A_C and C
    using a polar decomposition.
    """
    Ashape = A_C.shape
    UC = polarU(C)

    AC_mat_l = fuse_left(A_C)
    UAc_l = polarU(AC_mat_l)
    A_L = UAc_l @ UC.T.conj()
    A_L = unfuse_left(A_L, Ashape)

    AC_mat_r = fuse_right(A_C)
    UAc_r = polarU(AC_mat_r)
    A_R = UC.T.conj() @  UAc_r
    A_R = unfuse_right(A_R, Ashape)
    return (A_L, A_R)


@jax.jit
def polarU(a):

    """
    Compute the unitary part of the polar decomposition.
    """
    a = jnp.asarray(a)
    w, _, vh = jnp.linalg.svd(a, full_matrices=False)
    u = w @ vh
    return u


@jax.jit
def B2_variance(oldlist, newlist):
    """
    Given two MPS tensors in mixed canonical form, estimate the gradient
    variance.

    PARAMETERS
    ----------
    oldlist, newlist: Both lists [A_L, C, A_R] representing two MPS in
                      mixed canonical form.

    RETURNS
    ------
    B2 (float) : The gradient variance.
    """
    NL, NR = mps_null_spaces(oldlist)
    AL, C, AR = newlist
    AC = ct.rightmult(AL, C)
    L = ct.XopL(AC, B=NL)
    R = ct.XopR(AR, B=NR)
    B2_tensor = L @ R.T
    B2 = norm(B2_tensor)
    return B2


@jax.jit
def twositeexpect(mpslist, H):
    """
    The expectation value of the operator H in the state represented
    by A_L, C, A_R in mpslist.

    RETURNS
    -------
    out: The expectation value.
    """
    A_L, C, A_R = mpslist
    A_CR = ct.leftmult(C, A_R)
    expect = ct.twositeexpect(A_L, A_CR, H)
    return expect


@jax.jit
def mpsnorm(mpslist):
    A_L, C, A_R = mpslist
    A_CR = ct.leftmult(C, A_R)
    rho = ct.rholoc(A_L, A_CR)
    the_norm = trace(rho)
    return the_norm.real
