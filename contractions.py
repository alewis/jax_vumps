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
import tensornetwork as tn

try:
    environ_name = os.environ["LINALG_BACKEND"]
except KeyError:
    print("While importing contractions.py,")
    print("os.environ[\"LINALG_BACKEND\"] was undeclared; using NumPy.")
    environ_name = "numpy"


if environ_name == "jax":
    JAX_MODE = True
elif environ_name == "numpy":
    JAX_MODE = False
else:
    raise ValueError("Invalid LINALG_BACKEND ", environ_name)


def default_backend():
    try:
        backend = os.environ["LINALG_BACKEND"]
    except KeyError:
        backend = "numpy"
    if (backend != "jax" and backend != "numpy"):
        raise ValueError("Invalid backend ", backend, "was selected.")
    return backend


def jit_toggle(f, static_argnums=None, **kwargs):
    if JAX_MODE:
        return jax.jit(f, static_argnums=static_argnums, **kwargs)
    else:
        return f


def leftmult(lam, gam, backend=None):
    """
    2--lam--gam--3
            |
            1
            |
    where lam is stored 1--lam--2
    """
    #  if len(lam.shape) != 2:
    #      raise ValueError("Bad shape!")
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
    #  if len(lam.shape) != 2:
    #      raise ValueError("Bad shape!")
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
