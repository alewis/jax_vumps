"""
Low level tensor network manipulations.

Conventions
      2                 3 4
      |                 | |
      O                  U
      |                 | |
      1                 1 2

  2---A---3            1
      |                |
      1             2--A--3
"""
import os
import numpy as np
import jax.numpy as jnp
import tensornetwork as tn
from mps_flrw.bhtools.tebd.scon import scon


def default_backend():
    try:
        backend = os.environ["LINALG_BACKEND"]
    except KeyError:
        backend = "numpy"
    if (backend != "jax" and backend != "numpy"):
        raise ValueError("Invalid backend ", backend, "was selected.")
    return backend


#  def Bop(O, B, backend=None):
#      """
#         1
#         |
#         O
#         |
#      2--B--3
#      """
#      if backend is None:
#          backend = default_backend()
#      return tn.ncon([O, B],
#                     [(1, -1),
#                      (1, -2, -3)], backend=backend)


def leftmult(lam, gam, backend=None):
    """
    2--lam--gam--3
            |
            1
            |
    where lam is stored 1--lam--2
    """
    if backend is None:
        backend = default_backend()
    out = tn.ncon([lam, gam],
                  [[-2, 1],
                  [-1, 1, -3]], backend=backend)
    return out


def rightmult(gam, lam, backend=None):
    """
    2--gam--lam--3
       |
       1
       |
    """
    if backend is None:
        backend = default_backend()
    out = tn.ncon([gam, lam],
                  [[-1, -2, 1],
                  [1, -3]], backend=backend)
    return out


def gauge_transform(gl, A, gr, backend=None):
    """
            |
            1
     2--gl--A--gr--3
    """
    if backend is None:
        backend = default_backend()
    glA = leftmult(gl, A, backend=backend)
    out = rightmult(glA, gr, backend=backend)
    return out

###############################################################################
# Chain contractors - MPS.
###############################################################################


#  def leftchain(As, backend=None):
#      """
#        |---A1---A2---...-AN-2
#        |   |    |         |
#        |   |    |   ...   |
#        |   |    |         |
#        |---B1---B2---...-BN-1
#        (note: AN and BN will be contracted into a single matrix)

#        As is a list of MPS tensors.
#      """
#      Bs = [A.conj() for A in As]
#      for A, B in zip(As, Bs, backend=None):
#          X = XopL(A, B=B, X=X)
#      return X


#  def chainnorm(As, Bs=None, X=None, Y=None, backend=None):
#      """
#        |---A1---A2---...-AN---
#        |   |    |        |   |
#        X   |    |   ...  |   Y
#        |   |    |        |   |
#        |---B1---B2---...-BN---
#      """
#      X = leftchain(As, Bs=Bs, X=X)
#      if Y is not None:
#          X = np.dot(X, Y.T)
#      return np.trace(X)


def proj(A, B, backend=None):
    """
    2   2
    |---|
    |   |
    A   B
    |   |
    |---|
    1   1
    Contract A with B to find <A|B>.
    """
    if backend is None:
        backend = default_backend()
    idxs = [[1, 2], [1, 2]]
    contract = [A, B]
    ans = tn.ncon(contract, idxs, backend=backend)
    return ans


# *****************************************************************************
# Single site to open legs.
# *****************************************************************************
def XopL(A, B=None, X=None, backend=None):
    """
      |---A---2
      |   |
      X   |
      |   |
      |---B---1
    """
    if backend is None:
        backend = default_backend()
    if B is None:
        B = A.conj()
    if X is not None:
        A = leftmult(X, A, backend=backend)
    idx = [(2, 1, -2),
           (2, 1, -1)]
    return tn.ncon([A, B], idx, backend=backend)


def XopR(A, B=None, X=None, backend=None):
    """
      2---A---|
          |   |
          |   X
          |   |
      1---B---|
    """
    if backend is None:
        backend = default_backend()
    if B is None:
        B = A.conj()
    if X is not None:
        B = rightmult(B, X, backend=backend)
    idx = [(2, -2, 1),
           (2, -1, 1)]
    return tn.ncon([A, B], idx, backend=backend)


# ***************************************************************************
# TWO SITE OPERATORS
# ***************************************************************************
#  def normalize_fixed_points(l, r, lam=None, verbose=True):
#      """
#         a) l and r are both Hermitian positive semi-definite.
#         b) l.T * lam * r * lam = 1
#      """
#      l /= npla.norm(np.ravel(l))
#      r /= npla.norm(np.ravel(r))
#      # Divide out the appropriate phase to make l and r Hermitian pos. semi-def
#      r_tr = np.trace(r)
#      phase_r = r_tr/np.abs(r_tr)
#      r/= phase_r
#      l_tr = np.trace(l)
#      phase_l = l_tr/np.abs(l_tr)
#      l /= phase_l

#      normright = ct.gauge_transform(lam, r, lam)
#      n = np.vdot(np.ravel(l), np.ravel(normright))
#      abs_n = np.abs(n)
#      l /= np.sqrt(abs_n)
#      r /= np.sqrt(abs_n)
#      tol = 1E-12
#      if verbose and abs_n < tol:
#          print("Warning: l and r are orthogonal; their dot product is", abs_n)
#          #raise AssertionError()

#      lh = 0.5*(l+np.conj(l).T)
#      rh = 0.5*(r+np.conj(r).T)
#      if verbose:
#          if npla.norm(lh-l) > tol:
#              print("Warning: l was not made Hermitian")
#              #raise AssertionError()
#          if npla.norm(rh-r) > tol:
#              print("Warning: r was not made Hermitian")
#              #raise AssertionError()
#      return lh, rh


def rholoc(A1, A2, backend=None):
    """
    -----A1-----A2-----
    |    |(3)   |(4)   |
    |                  |
    |                  |
    |    |(1)   |(2)   |
    -----A1-----A2------
    returned as a (1:2)x(3:4) matrix.
    Assuming the appropriate Schmidt vectors have been contracted into the As,
    np.trace(np.dot(op, rholoc.T)) is the expectation value of the two-site
    operator op coupling A1 to A2.
    """
    if backend is None:
        backend = default_backend()
    B1 = A1.conj()
    B2 = A2.conj()
    d = A1.shape[0]
    to_contract = [A1, A2, B1, B2]
    idxs = [(-3, 1, 2),
            (-4, 2, 3),
            (-1, 1, 4),
            (-2, 4, 3)]
    rholoc = tn.ncon(to_contract, idxs, backend=backend).reshape((d**2, d**2))
    return rholoc


def twositecontract(left, right, U, backend=None):
    """
       2--left-right--4
            |__|
            |U |
            ----
            |  |
            1  3
    """
    if backend is None:
        backend = default_backend()
    to_contract = (left, right, U)
    idxs = [(2, -2, 1),
            (3, 1, -4),
            (-1, -3, 2, 3)]
    return tn.ncon(to_contract, idxs, backend=backend)


def twositeexpect(left, right, U, backend=None):
    if backend is None:
        backend = default_backend()
    d = U.shape[0]
    rho = rholoc(left, right, backend=backend).reshape((d, d, d, d))
    idxs = [(1, 2, 3, 4), (1, 2, 3, 4)]
    expect = tn.ncon([rho, U], idxs, backend=backend).real
    return expect


def tmdense(A, backend=None):
    """
    2-A-4
      |
      |
    1-A-3
    """
    if backend is None:
        backend = default_backend()
    idxs = [[1, -2, -4], [1, -1, -3]]
    out = tn.ncon([A, A.conj()], idxs, backend=backend)
    return out


##############################################################################
# VUMPS environment
##############################################################################
def compute_hL(A_L, htilde, backend=None):
    """
    --A_L--A_L--
    |  |____|
    |  | h  |
    |  |    |
    |-A_L*-A_L*-
    """
    if backend is None:
        backend = default_backend()
    A_L_d = A_L.conj()
    to_contract = [A_L, A_L, A_L_d, A_L_d, htilde]
    idxs = [(2, 4, 1),
            (3, 1, -2),
            (5, 4, 7),
            (6, 7, -1),
            (5, 6, 2, 3)]
    h_L = tn.ncon(to_contract, idxs, backend=backend)
    return h_L


#  def compute_hLgen(A_L1, A_L2, A_L3, A_L4, htilde, backend=None):
#      """
#      --A_L1--A_L2--
#      |  |____|
#      |  | h  |
#      |  |    |
#      |-A_L3-A_L4-
#      """
#      to_contract = [A_L1, A_L2, A_L3, A_L4, htilde]
#      idxs = [(2, 4, 1),
#              (3, 1, -2),
#              (5, 4, 7),
#              (6, 7, -1),
#              (5, 6, 2, 3)]
#      h_L = tn.ncon(to_contract, idxs)
#      return h_L


def compute_hR(A_R, htilde, backend=None):
    """
     --A_R--A_R--
        |____|  |
        | h  |  |
        |    |  |
     --A_R*-A_R*-
    """
    if backend is None:
        backend = default_backend()
    A_R_d = A_R.conj()
    to_contract = [A_R, A_R, A_R_d, A_R_d, htilde]
    idxs = [(2, -2, 1),
            (3, 1, 4),
            (5, -1, 7),
            (6, 7, 4),
            (5, 6, 2, 3)]
    h_R = tn.ncon(to_contract, idxs, backend=backend)
    return h_R


##############################################################################
# VUMPS heff 
##############################################################################
def apply_HAc(A_C, A_L, A_R, Hlist, backend=None):
    """
    Compute A'C via eq 11 of vumps paper (131 of tangent space methods).
    """
    if backend is None:
        backend = default_backend()
    H, LH, RH = Hlist
    to_contract_1 = [A_L, A_L.conj(), A_C, H]
    idxs_1 = [(2, 1, 4),
              (3, 1, -2),
              (5, 4, -3),
              (3, -1, 2, 5)]
    term1 = tn.ncon(to_contract_1, idxs_1, backend=backend)

    to_contract_2 = [A_C, A_R, A_R.conj(), H]
    idxs_2 = [(5, -2, 4),
              (2, 4, 1),
              (3, -3, 1),
              (-1, 3, 5, 2)]
    term2 = tn.ncon(to_contract_2, idxs_2, backend=backend)

    term3 = leftmult(LH, A_C, backend=backend)
    term4 = rightmult(A_C, RH.T, backend=backend)
    A_C_prime = term1 + term2 + term3 + term4
    return A_C_prime


def HAc_dense(A_L, A_R, Hlist, backend=None):
    """
    Construct the dense effective Hamiltonian HAc.
    """
    if backend is None:
        backend = default_backend()
    d, chi, _ = A_L.shape
    H, LH, RH = Hlist
    if backend == "jax":
        I_chi = jnp.eye(chi, dtype=H.dtype)
        I_d = jnp.eye(d, dtype=H.dtype)
    else:
        I_chi = np.eye(chi, dtype=H.dtype)
        I_d = np.eye(d, dtype=H.dtype)

    contract_1 = [A_L, A_L.conj(), H, I_chi]
    idx_1 = [(2, 1, -5),
             (3, 1, -2),
             (3, -1, 2, -4),
             (-3, -6)
             ]
    term1 = tn.ncon(contract_1, idx_1, backend=backend)

    contract_2 = [I_chi, A_R, A_R.conj(), H]
    idx_2 = [(-2, -5),
             (2, -6, 1),
             (4, -3, 1),
             (-1, 4, -4, 2)
             ]
    term2 = tn.ncon(contract_2, idx_2, backend=backend)

    contract_3 = [LH, I_d, I_chi]
    idx_3 = [(-2, -5),
             (-1, -4),
             (-3, -6)
             ]
    term3 = tn.ncon(contract_3, idx_3, backend=backend)

    contract_4 = [I_chi, I_d, RH]
    idx_4 = [(-2, -5),
             (-1, -4),
             (-6, -3)
             ]
    term4 = tn.ncon(contract_4, idx_4, backend=backend)

    HAc = term1 + term2 + term3 + term4
    return HAc


def apply_HAc_dense(A_C, A_L, A_R, Hlist):
    """
    Construct the dense effective Hamiltonian HAc and apply it to A_C.
    For testing.
    """
    d, chi, _ = A_C.shape
    HAc = HAc_dense(A_L, A_R, Hlist)
    HAc_mat = HAc.reshape((d*chi*chi, d*chi*chi))
    A_Cvec = A_C.flatten()
    A_C_p = HAc_mat @ A_Cvec
    A_C_p = A_C_p.reshape(A_C.shape)
    return A_C_p


def apply_Hc(C, A_L, A_R, Hlist, backend=None):
    """
    Compute C' via eq 16 of vumps paper (132 of tangent space methods).
    """
    if backend is None:
        backend = default_backend()
    H, LH, RH = Hlist
    A_Lstar = A_L.conj()
    A_C = rightmult(A_L, C, backend=backend)
    to_contract = [A_C, A_Lstar, A_R, A_R.conj(), H]
    idxs = [(4, 1, 3),
            (6, 1, -1),
            (5, 3, 2),
            (7, -2, 2),
            (6, 7, 4, 5)]
    term1 = tn.ncon(to_contract, idxs, backend=backend)
    term2 = LH @ C
    term3 = C @ RH.T
    C_prime = term1 + term2 + term3
    return C_prime


def Hc_dense(A_L, A_R, Hlist, backend=None):
    """
    Construct Hc as a dense matrix.
    """
    if backend is None:
        backend = default_backend()
    H, LH, RH = Hlist
    chi = LH.shape[0]
    Id = np.eye(chi, dtype=LH.dtype)
    to_contract_1 = [A_L, A_R, np.conj(A_L), np.conj(A_R), H]
    idx_1 = [[2, 1, -4],
             [4, -3, 6],
             [3, 1, -2],
             [5, -1, 6],
             [3, 5, 2, 4]
             ]
    term1 = tn.ncon(to_contract_1, idx_1, backend=backend)
    if backend == "jax":
        term2 = jnp.kron(Id, LH).reshape((chi, chi, chi, chi))
        term3 = jnp.kron(RH.T, Id).reshape((chi, chi, chi, chi))
    else:
        term2 = np.kron(Id, LH).reshape((chi, chi, chi, chi))
        term3 = np.kron(RH.T, Id).reshape((chi, chi, chi, chi))
    H_C = term1 + term2 + term3
    H_C = H_C.transpose((1, 0, 3, 2))
    return H_C
