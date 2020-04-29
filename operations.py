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
# INITIALIZERS
###############################################################################


def makechis(d, N, maxchi):
    """Create the vector of chis for the chain.
       This is a length-N+1 list of exponents of d. The exponents are of
       the form
       [0, 1, 2, 1, 0] (if N+1 is odd)
       [0, 1, 2, 2, 1, 0] (if N+1 is even)
       Any numbers in the above exceeding ln(maxchi) / ln(d) are replaced
       with maxchi.
    """
    last = N
    chis = np.zeros(N+1, dtype=np.int)
    trunc = False
    middle = -(-(N+1) // 2)
    for k in range(middle):
        if not trunc:
            chi = d**k
            if chi > maxchi:
                trunc = True
                chi = maxchi
        else:
            chi = maxchi
        chis[k] = chi

    chirs = []
    trunc = False
    for k in range(middle):#range(N-1, middle-1, -1):
        if not trunc:
            chi = d**k
            if chi > maxchi:
                trunc = True
                chi = maxchi
        else:
            chi = maxchi
        chis[N - k] = chi
    return chis


def random_finite_mps(d, n, maxchi, dtype=jnp.float32):
    chis = makechis(d, n, maxchi)
    mps_shapes = []
    for i in range(len(chis)-1):
        mps_shapes.append((chis[i], d, chis[i+1]))
    mps_chain = utils.random_tensors(mps_shapes, dtype=dtype)
    return mps_chain


def left_boundary_eye(chiM, dtype=jnp.float32):
    """
    'Identity' left boundary Hamiltonian for a finite chain. chiM
    is the mpo bond dimensions.
    """
    errflag, errstr = errors.check_natural(chiM, "chiM")
    if errflag:
        raise ValueError(errstr)

    L = np.zeros(chiM)
    L[0] = 1.
    L = jnp.array(L, dtype=dtype).reshape((chiM, 1, 1))
    return L


def right_boundary_eye(chiM, dtype=jnp.float32):
    """
    'Identity' right boundary Hamiltonian for a finite chain. chiM
    is the mpo bond dimensions.
    """
    errflag, errstr = errors.check_natural(chiM, "chiM")
    if errflag:
        raise ValueError(errstr)

    R = np.zeros(chiM)
    R[-1] = 1.
    R = jnp.array(R, dtype=dtype).reshape((chiM, 1, 1))
    return R


def xx_mpo():
    """
    The XX Hamiltonian in MPO form.
    """
    d = 2
    sP = np.sqrt(2)*np.array([[0, 0], [1, 0]])
    sM = np.sqrt(2)*np.array([[0, 1], [0, 0]])
    sI = np.array([[1, 0], [0, 1]])
    M = np.zeros([4, 4, d, d])
    M[0, 0, :, :] = sI
    M[3, 3, :, :] = sI
    M[0, 1, :, :] = sM
    M[1, 3, :, :] = sP
    M[0, 2, :, :] = sP
    M[2, 3, :, :] = sM
    M = jnp.array(M)
    return M


###############################################################################
# CONTRACTIONS
###############################################################################
@jax.jit
def leftcontract(C, mps):
    """
    --0C1----0mps2-- -> --0mps2--
               1            1
               |            |
    """
    mps = jnp.einsum("ab, bcd", C, mps)
    return mps


@jax.jit
def rightcontract(mps, C):
    """
    --0mps2----0C1-- -> --0mps2--
        1                   1
        |                   |
    """
    mps = jnp.dot(mps, C)
    return mps


@jax.jit
def XopL(L, mpo, mps):
    """
    **FIX**
    ----0mps2--      ---
    |     1          |
    2     3          2
    L0--0mpo1--  ->  L0-
    1     2          1
    |     |          |
    -----mps*--      ---
    """
    mps_d = jnp.conj(mps)
    L = jnp.einsum("egd, eahf, dfc, ghb", L, mpo, mps, mps_d)
    return L

@jax.jit
def XL(L, mps):
    """
    ----0mps2--      ---
    |     1          |
    1     |          1
    L     |          L
    0     |          0
    |     1          |
    ----0mps*2-      ---
    """
    mps_d = jnp.conj(mps)
    L = jnp.einsum("cd, deb, cea", L, mps, mps_d)
    return L

@jax.jit
def XnoL(mps):
    """
    ----0mps2--      ---
    |     1          |
    |     |          1
    |     |          L
    |     |          0
    |     1          |
    ----0mps*2-      ---
    """
    mps_d = jnp.conj(mps)
    L = jnp.einsum("cdb, cda", mps, mps_d)
    return L


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


@jax.jit
def XopR(R, mpo, mps):
    """
    ---0mps2--       --
         1    |       |
         3    2       2
    ---0mpo1-0R  ->  0R
         2    1       1
         |    |       |
    ----mps*--|      --
    """
    mps_d = jnp.conj(mps)
    R = jnp.einsum("fed, afgh, cgd, bhe", R, mpo, mps, mps_d)
    return R


@jax.jit
def joinL(L, C):
    """
    --0C1-     ---
    |          |
    2          2
    L0--   ->  L0-
    1          1
    |          |
    ----       ---
    """
    res = jnp.einsum("abd, dc", L, C)
    return res


@jax.jit
def joinR(C, R):
    """
      --          --
       |           |
       2           2
     -0R   ->    -0R
       1           1
       |           |
    -0C1          --
    """
    res = jnp.einsum("adc, bd", R, C)
    return res


#@jax.tree_util.Partial
@jax.jit
def single_mpo_heff(mpo, L, R, A):
    tensors = [L, A, mpo, R]
    idxs = [[2, -1, 1],
            [1, 3, 4],
            [2, 5, -2, 3],
            [5, -3, 4]]
    newA = tensornetwork.ncon(tensors, idxs, backend="jax")
    return newA


def single_mpo_heff_np(mpo, L, R, A):
    tensors = [L, A, mpo, R]
    idxs = [[2, -1, 1],
            [1, 3, 4],
            [2, 5, -2, 3],
            [5, -3, 4]]
    newA = tensornetwork.ncon(tensors, idxs)
    return newA


@jax.tree_util.Partial
@jax.jit
def single_mpo_heff1(mpo, L, R, A):
    newA = jnp.einsum('fad, dhe, fgbh, gce', L, A, mpo, R)
    return newA


@jax.tree_util.Partial
@jax.jit
def single_mpo_heff2(mpo, L, R, A):
    tensors = [L, A, mpo, R]
    idxs = [[3, -1, 1],
            [1, 5, 2],
            [3, 4, -2, 5],
            [4, -3, 2]]
    newA = tensornetwork.ncon(tensors, idxs, backend="jax")
    return newA



@jax.jit
def single_mpo_heff5(mpo, L, R, A):
    newA = jnp.einsum('ead, dfg, ehbf, hcg', L, A, mpo, R)
    return newA

@jax.jit
def single_mpo_heff4(mpo, L, R, A):
    chiM, _, _, _ = mpo.shape
    chiL, d, chiR = A.shape
    #LA = jnp.einsum("abe, ecd", L, A)
    LA = jnp.matmul(L.reshape(chiM*chiL, chiL), A.reshape(chiL, d*chiR))
    LA = LA.reshape(chiM, chiL, d, chiR)
    LAM1 = jnp.einsum("eafd, ecbf", LA, mpo) #chiL, d, chiM, chiR



    #  LA = LA.transpose((3, 0, 2, 1)).reshape((chiL*chiR, d*chiM))
    #  mpo = mpo.transpose((0, 2, 3, 1)).reshape((d*chiM, d*chiM))
    #  LAM = jnp.matmul(LA, mpo).reshape((chiL, chiR, d, chiM))
    #  print(LAM.shape)
    #  LAM = LAM.transpose((0, 3, 1, 2))

    #  LAM = LAM.transpose((0, 3, 1, 2)).reshape((chiL*d, chiM*chiR))
    #LAM1 = LAM1.transpose((0, 3, 1, 2)).reshape((chiL*d, chiM*chiR))
    #LAM1 = LAM1.reshape((chiL*d, chiM*chiR))
    LAM1 = LAM1.reshape((chiL*d, chiM*chiR))
    # print(LAM.shape)
    Rt = R.transpose((0, 2, 1)).reshape((chiR*chiM, chiR))
    LAMR = jnp.matmul(LAM1, Rt)
    newA = LAMR.reshape((chiL, d, chiR))
    return newA

###############################################################################
# QR
###############################################################################
@jax.jit
def qrpos(mps):
    """
    Reshapes the (chiL, d, chiR) MPS tensor into a (chiL*d, chiR) matrix,
    and computes its QR decomposition, with the phase of R fixed so as to
    have a non-negative main diagonal. A new left-orthogonal
    (chiL, d, chiR) MPS tensor (reshaped from Q) is returned along with
    R.

    In addition to being phase-adjusted, R is normalized by division with
    its L2 norm.

    PARAMETERS
    ----------
    mps (array-like): The (chiL, d, chiR) MPS tensor.

    RETURNS
    -------
    mps_L, R: A left-orthogonal (chiL, d, chiR) MPS tensor, and an upper
              triangular (chiR x chiR) matrix with a non-negative main
              diagonal such that mps = mps_L @ R.
    """
    chiL, d, chiR = mps.shape
    mps_mat = jnp.reshape(mps, (chiL*d, chiR))
    Q, R = jnp.linalg.qr(mps_mat)
    phases = jnp.sign(jnp.diag(R))
    Q = Q*phases
    R = jnp.conj(phases)[:, None] * R
    R = R / jnp.linalg.norm(R)
    mps_L = Q.reshape(mps.shape)
    return (mps_L, R)


@jax.jit
def lqpos(mps):
    """
    Reshapes the (chiL, d, chiR) MPS tensor into a (chiL, d*chiR) matrix,
    and computes its LQ decomposition, with the phase of L fixed so as to
    have a non-negative main diagonal. A new right-orthogonal
    (chiL, d, chiR) MPS tensor (reshaped from Q) is returned along with
    L.
    In addition to being phase-adjusted, L is normalized by division with
    its L2 norm.

    PARAMETERS
    ----------
    mps (array-like): The (chiL, d, chiR) MPS tensor.

    RETURNS
    -------
    L, mps_R:  A lower-triangular (chiL x chiL) matrix with a non-negative
               main-diagonal, and a right-orthogonal (chiL, d, chiR) MPS
               tensor such that mps = L @ mps_R.
    """
    chiL, d, chiR = mps.shape
    mps_mat = jnp.reshape(mps, (chiL, chiR*d))
    mps_mat = jnp.conj(mps_mat.T)
    Qdag, Ldag = jnp.linalg.qr(mps_mat)
    Q = jnp.conj(Qdag.T)
    L = jnp.conj(Ldag.T)
    phases = jnp.sign(jnp.diag(L))
    L = L*phases
    L = L / jnp.linalg.norm(L)
    Q = jnp.conj(phases)[:, None] * Q
    mps_R = Q.reshape(mps.shape)
    return (L, mps_R)


"""
matvecs
"""
def matmat(mv, mv_args, X):
    """
    Does A@X for X a matrix, given mv implementing A@x for x a vector.
    """
    return jnp.hstack([mv(*mv_args, col.reshape(-1, 1)) for col in X.T])


@jax.jit
def matrix_matvec(A, x):
    return A@x


def numpy_matrix_matvec(A, x):
    return A@x


def numpy_matrix_linop(A):
    op = LinearOperator(shape=A.shape,
                        matvec=lambda x: numpy_matrix_matvec(A, x))
    return op
