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


#@partial(jax.jit, static_argnums=(1,))
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


#@partial(jax.jit, static_argnums=(1,))
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


#@jax.jit
#  def null_space(A):
#      """
#      The scipy code to compute the null space of a matrix.
#      """
#      m, n = A
#      Q, R = jnp.linalg.qr(A, mode="complete")
#      eps = jnp.finfo(A.dtype).eps
#      s = 0.
#      r = m
#      for j in range(m-1, -1, -1):
#          s += jnp.sum(R[j, :])
#          eps *= n
#          if s <= eps:
#              r = j
#      Qnull = Q[r:, :].T.conj()
#      return Qnull
    #  u, s, vh = jnp.linalg.svd(A, full_matrices=True)
    #  M, N = u.shape[0], vh.shape[1]
    #  rcond = jnp.finfo(s.dtype).eps * max(M, N)
    #  tol = jnp.amax(s) * rcond
    #  num = jnp.sum(s > tol, dtype=int)
    #  Q = vh[num:, :].T.conj()
    #return Q


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


#  @jax.jit
#  def gauge_match(A_C, C):
#      """
#      Return approximately gauge-matched A_L and A_R from A_C and C
#      using a polar decomposition.
#      """
#      Ashape = A_C.shape
#      UC = polarU(C)

#      AC_mat_l = fuse_left(A_C)
#      UAc_l = polarU(AC_mat_l)
#      A_L = UAc_l @ UC.T.conj()
#      A_L = unfuse_left(A_L, Ashape)

#      AC_mat_r = fuse_right(A_C)
#      UAc_r = polarU(AC_mat_r)
#      A_R = UC.T.conj() @  UAc_r
#      A_R = unfuse_right(A_R, Ashape)
#      return (A_L, A_R)


@jax.jit
def polarjit(A, svd):
    U = jax.lax.cond(svd, A, polarU_SVD,
                          A, polarU_QDWH)
    return U


@jax.jit
def gauge_match(A_C, C, svd=True):
    """
    Return approximately gauge-matched A_L and A_R from A_C and C
    using a polar decomposition.

    A_L and A_R are chosen to minimize ||A_C - A_L C|| and ||A_C - C A_R||.
    The respective solutions are the isometric factors in the
    polar decompositions of A_C C\dag and C\dag A_C.

    PARAMETERS
    ----------
    A_C (d, chi, chi)
    C (chi, chi)     : MPS tensors.
    svd (bool)      :  Toggles whether the SVD or QDWH method is used for the 
                       polar decomposition. In general, this should be set
                       False on the GPU and True otherwise.

    RETURNS
    -------
    A_L, A_R (d, chi, chi): Such that A_L C A_R minimizes ||A_C - A_L C||
                            and ||A_C - C A_R||, with A_L and A_R
                            left (right) isometric.
    """
    Ashape = A_C.shape
    UC = polarjit(C, svd)

    AC_mat_l = fuse_left(A_C)
    UAc_l = polarjit(AC_mat_l, svd)
    A_L = UAc_l @ UC.T.conj()
    A_L = unfuse_left(A_L, Ashape)

    AC_mat_r = fuse_right(A_C)
    UAc_r = polarjit(AC_mat_r, svd)
    A_R = UC.T.conj() @  UAc_r
    A_R = unfuse_right(A_R, Ashape)
    return (A_L, A_R)


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


#############################################################################
# Polar decomposition
#############################################################################
@jax.jit
def polarU_SVD(A):

    """
    Compute the unitary part of the polar decomposition explitly as
    U = u @ vH where A = u @ S @ vh is the SVD of A. This is about twice
    as fast as polarU_QDWH on the
    CPU or TPU but around an order of magnitude slower on the GPU.
    """
    a = jnp.asarray(A)
    w, _, vh = jnp.linalg.svd(A, full_matrices=False)
    u = w @ vh
    return u


def polarU_QDWH(A, Niter=4):
    """
    Computes the isometric factor U in the polar decomposition, U = u @ vh
    where u and vh are the singular vector matrices in the SVD.

    This algorithm computes this factor using the "QDWH" iterative algorithm,
    (explained for example at https://sci-hub.tw/10.1137/120876605), which
    is based on an iterative procedure called "dynamically weighted Halley
    iterations". Each iteration is essentially performed by
    weighted_iterationQR. Eventually (usually after 2 iterations) we switch to
    the cheaper weighted_iterationChol, which is mathematically equivalent
    but only stable when the input is well-conditioned; the iterations
    improve the condition number.

    Compared to directly computing u @ vh via the SVD, this algorithm is
    considerably (~7-20 X) faster on the GPU, but perhaps 2X slower on the
    CPU or TPU.


    PARAMETERS
    ----------
    A: The matrix to be decomposed.
    Niter: The number of QDWH iterations.


    RETURNS
    -------
    U: The approximate polar factor.
    """
    m, n = A.shape
    if n > m:
        U = polar_qdwh_transpose(A, Niter)
    else:
        U = polar_qdwh(A, Niter)
    return U


@partial(jax.jit, static_argnums=(1,))
def polar_qdwh_transpose(A, Niter):
    """
    Handles the polar decomposition when n > m by transposing the input and
    then the output.
    """
    A = A.T.conj()
    Ud = polar_qdwh(A, Niter)
    U = Ud.T.conj()
    return U


@partial(jax.jit, static_argnums=(1,))
def polar_qdwh(A, Niter):
    """
    Implements the QDWH polar decomposition. 
    """
    m, n = A.shape
    alpha = jnp.linalg.norm(A)
    lcond = 1E-6
    k = 0
    X = A / alpha
    for k in range(Niter):
        a = hl(lcond)
        b = (a - 1)**2 / 4
        c = a + b - 1
        X = jax.lax.cond(c < 100, (X, a, b, c), weighted_iterationChol,
                                  (X, a, b, c), weighted_iterationQR)
        # if c < 100:
        #   X = weighted_iterationChol(X, a, b, c)
        # else:
        #   X = weighted_iterationQR(X, a, b, c)
        lcond *= (a + b * lcond**2)/(1 + c * lcond**2)
    return X


@jax.jit
def hl(l):
    d = (4*(1 - l**2)/(l**4))**(1/3)
    f = 8*(2 - l**2)/(l**2 * (1 + d)**(1/2))
    h = (1 + d)**(1/2) + 0.5 * (8 - 4*d + f)**0.5
    return h


@jax.jit
def weighted_iterationChol(args):
    """
    One iteration of the QDWH polar decomposition, using the cheaper but
    less stable Cholesky factorization method.
    """
    X, a, b, c = args
    m, n = X.shape
    eye = jnp.eye(n)
    Z = eye + c * X.T.conj() @ X
    W = jax.scipy.linalg.cholesky(Z)
    Winv = jax.scipy.linalg.solve_triangular(W, eye, overwrite_b=True)
    XWinv = X@Winv
    X *= (b / c)
    X += (a - (b/c))*(XWinv)@(Winv.T.conj())
    return X


@jax.jit
def weighted_iterationQR(args):
    """
    One iteration of the QDWH polar decomposition, using the more expensive
    but more stable QR factorization method.
    """
    X, a, b, c = args
    m, n = X.shape
    eye = jnp.eye(n)
    cX = jnp.sqrt(c)*X
    XI = jnp.vstack((cX, eye))
    Q, R = jnp.linalg.qr(XI)
    Q1 = Q[:m, :]
    Q2 = Q[m:, :]
    X *= (b/c)
    X += (1/jnp.sqrt(c)) * (a - b/c) * Q1 @ Q2.T.conj()
    return X

#  def vumps_shim(H, params):
#      """
#      Find the ground state of a uniform two-site Hamiltonian.
#      This is the function to call from outside.
#      """

#      print("VUMPS! VUMPS! VUMPS/VUMPS/VUMPS/VUMPS! VUMPS!")
#      chi = params["chi"]
#      delta = params["delta_0"]
#      tol = params["vumps_tol"]
#      max_iter = params["maxiter"]
#      d = H.shape[0]
#      mpslist, A_C, fpoints = old_vumps.vumps_initial_tensor(d, chi, params,
#                                                             H.dtype)

#      print("It begins...")
#      vumps_state = [False]
#      H_env = old_vumps.vumps_environment(mpslist, fpoints, H, delta, params)
#      Niter = 0
#      while delta >= tol and Niter < max_iter:
#          print("delta = ", delta)
#          mpslist, A_C, fpoints, H_env, delta, vumps_state = old_vumps.vumps_iter(
#                  mpslist, A_C, fpoints, H, H_env, delta, params,
#                  vumps_state)
#      return (None, None)


#  def vumps_params(path="vumpsout/",
#          #dynamic_chi = False,
#          chi=64,
#          checkpoint_every = 500,
#          #chimax=200,
#          #dchi=8,
#          #checkchi=30,
#          #B2_thresh = 1E-14,
#          delta_0=1E-1,
#          vumps_tol=1E-14,
#          maxiter = 1000,
#          outdir = None,
#          #maxchi=128,
#          #minlam = 1E-13,
#          svd_switch = 1,
#          dom_ev_approx = True,
#          #dom_ev_approx = False,
#          adaptive_tm_tol = True,
#          TM_neigs = 1,
#          TM_ncv = 40,
#          TM_tol = 0.1,
#          TM_tol_initial = 1E-12,
#          adaptive_env_tol = True,
#          env_tol = 0.01,
#          env_maxiter = 100,
#          outer_k_lgmres = 10,
#          inner_m_lgmres = 30,
#          adaptive_Heff_tol = True,
#          Heff_tol = 0.01,
#          Heff_ncv = 40,
#          Heff_neigs = 1
#          ):
#      """
#      Default arguments for vumps. Also documents the parameters.

#      Nothing here should change during a VUMPS iteration.
#      """
#      params = dict()
#      #params["dynamic_chi"] = dynamic_chi #If true, increases chi from chi to chimax
#          #by increments of dchi every checkchi iterations of vumps_tol
#          #has not yet been reached. Otherwise, chi is maintained throughout.
#      #params["dchi"] = dchi
#      #params["chimax"] = chimax
#      #params["checkchi"] = checkchi
#      #params["B2_thresh"] = B2_thresh

#          #bond dimension with an SVD at this periodicity. The truncation
#          #error will also be measured.
#      #params["maxchi"] = maxchi #Maximum bond dimension.
#      params["chi"] = chi
#      params["checkpoint_every"] = checkpoint_every
#      #params["minlam"] = minlam #Truncate singular values smaller than this.

#      params["delta_0"] = delta_0 #Initial value for the loss function.

#                               #Must be larger than tol.
#      params["vumps_tol"] = vumps_tol  #Converge to this tolerance.
#                           #None means machine precision.
#      params["maxiter"] = maxiter #Maximum iterations allowed to VUMPS.
#      params["outdir"] = path #Where to save output.
#      #params["svd_switch"] = svd_switch #Do regauging with a polar decomposition
#                                   #instead of an SVD when the singular
#                                   #values become this small.

#      params["dom_ev_approx"] = dom_ev_approx
#                      #If True, approximate e.g. lR as C^dag * C instead
#                      #of solving for it explictly.
#      params["adaptive_tm_tol"] = adaptive_tm_tol
#      params["TM_neigs"] = TM_neigs #Number of eigenvectors found when diagonalizing
#                             #the transfer matrix (only 1 is returned).
#      params["TM_ncv"] = TM_ncv #Number of Krylov vectors used to diagonalize
#                            # the transfer matrix.
#      params["TM_tol"] = TM_tol #Tolerance when diagonalizing the transfer
#                                #matrix.
#      params["TM_tol_initial"] = TM_tol_initial
#      params["adaptive_env_tol"] = adaptive_env_tol #See env_tol.
#      params["env_tol"] = env_tol #Solver tolerance for finding the
#                                #reduced environments. If adaptive_env_tol
#                                #is True, the solver tolerance is the
#                                #gradient norm multiplied by env_tol.
#      params["env_maxiter"] = env_maxiter #The maximum number of steps used to solve
#                      #for the reduced environments will be
#                      #(env_maxiter+1)*innermlgmres
#      params["outer_k_lgmres"] = outer_k_lgmres #Number of vectors to carry between
#                              #inner GMRES iterations (when finding the
#                              #reduced environments).
#      params["inner_m_lgmres"] = inner_m_lgmres #Number of inner GMRES iterations per
#                              #each outer iteration (when finding the
#                              #reduced environments).
#      params["adaptive_Heff_tol"] = adaptive_Heff_tol #See Heff_tol.
#      params["Heff_tol"] = Heff_tol #Solver tolerance for finding the
#                           #effective Hamiltonians. If adaptive_Heff_tol is
#                           #True, the solver tolerance is the gradient
#                           #norm multiplied by Heff_tol.
#      params["Heff_ncv"] = Heff_ncv #Number of Krylov vectors used to diagonalize
#                             #the effective Hamiltonians.
#      params["Heff_neigs"] = Heff_neigs #Number of eigenvectors found when solving
#                               #the effective Hamiltonians (only 1 is
#                               #returned).
#      return params
