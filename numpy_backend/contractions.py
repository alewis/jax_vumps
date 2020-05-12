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
import tensornetwork as tn


def leftmult(lam, gam):
    """
    2--lam--gam--3
            |
            1
            |
    where lam is stored 1--lam--2
    """
    #  if len(lam.shape) != 2:
    #      raise ValueError("Bad shape!")
    out = tn.ncon([lam, gam],
                  [[-2, 1],
                  [-1, 1, -3]], backend="numpy")
    return out


def rightmult(gam, lam):
    """
    2--gam--lam--3
       |
       1
       |
    """
    #  if len(lam.shape) != 2:
    #      raise ValueError("Bad shape!")
    out = tn.ncon([gam, lam],
                  [[-1, -2, 1],
                   [1, -3]], backend="numpy")
    return out


def gauge_transform(gl, A, gr):
    """
            |
            1
     2--gl--A--gr--3
    """
    glA = leftmult(gl, A, backend="numpy")
    out = rightmult(glA, gr, backend="numpy")
    return out

###############################################################################
# Chain contractors - MPS.
###############################################################################
def proj(A, B):
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
    ans = tn.ncon(contract, idxs, backend="numpy")
    return ans


# *****************************************************************************
# Single site to open legs.
# *****************************************************************************
def XopL(A, B=None, X=None):
    """
      |---A---2
      |   |
      X   |
      |   |
      |---B---1
    """
    if B is None:
        B = A.conj()
    if X is not None:
        A = leftmult(X, A)
    idx = [(2, 1, -2),
           (2, 1, -1)]
    return tn.ncon([A, B], idx, backend="numpy")


def XopR(A, B=None, X=None):
    """
      2---A---|
          |   |
          |   X
          |   |
      1---B---|
    """
    if B is None:
        B = A.conj()
    if X is not None:
        B = rightmult(B, X)
    idx = [(2, -2, 1),
           (2, -1, 1)]
    return tn.ncon([A, B], idx, backend="numpy")


# ***************************************************************************
# TWO SITE OPERATORS
# ***************************************************************************
def rholoc(A1, A2):
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
    B1 = A1.conj()
    B2 = A2.conj()
    d = A1.shape[0]
    to_contract = [A1, A2, B1, B2]
    idxs = [(-3, 1, 2),
            (-4, 2, 3),
            (-1, 1, 4),
            (-2, 4, 3)]
    rholoc = tn.ncon(to_contract, idxs, backend="numpy").reshape((d**2, d**2))
    return rholoc


def twositecontract(left, right, U):
    """
       2--left-right--4
            |__|
            |U |
            ----
            |  |
            1  3
    """
    to_contract = (left, right, U)
    idxs = [(2, -2, 1),
            (3, 1, -4),
            (-1, -3, 2, 3)]
    return tn.ncon(to_contract, idxs, backend="numpy")


def twositeexpect(left, right, U):
    d = U.shape[0]
    rho = rholoc(left, right).reshape((d, d, d, d))
    idxs = [(1, 2, 3, 4), (1, 2, 3, 4)]
    expect = tn.ncon([rho, U], idxs, backend="numpy").real
    return expect


def tmdense(A):
    """
    2-A-4
      |
      |
    1-A-3
    """
    idxs = [[1, -2, -4], [1, -1, -3]]
    out = tn.ncon([A, A.conj()], idxs, backend="numpy")
    return out


##############################################################################
# VUMPS environment
##############################################################################
def compute_hL(A_L, htilde):
    """
    --A_L--A_L--
    |  |____|
    |  | h  |
    |  |    |
    |-A_L*-A_L*-
    """
    A_L_d = A_L.conj()
    to_contract = [A_L, A_L, A_L_d, A_L_d, htilde]
    idxs = [(2, 4, 1),
            (3, 1, -2),
            (5, 4, 7),
            (6, 7, -1),
            (5, 6, 2, 3)]
    h_L = tn.ncon(to_contract, idxs, backend="numpy")
    return h_L


def compute_hR(A_R, htilde):
    """
     --A_R--A_R--
        |____|  |
        | h  |  |
        |    |  |
     --A_R*-A_R*-
    """
    A_R_d = A_R.conj()
    to_contract = [A_R, A_R, A_R_d, A_R_d, htilde]
    idxs = [(2, -2, 1),
            (3, 1, 4),
            (5, -1, 7),
            (6, 7, 4),
            (5, 6, 2, 3)]
    h_R = tn.ncon(to_contract, idxs, backend="numpy")
    return h_R


##############################################################################
# VUMPS heff 
##############################################################################
def apply_HAc(A_C, A_L, A_R, Hlist):
    """
    Compute A'C via eq 11 of vumps paper (131 of tangent space methods).
    """
    H, LH, RH = Hlist
    to_contract_1 = [A_L, A_L.conj(), A_C, H]
    idxs_1 = [(2, 1, 4),
              (3, 1, -2),
              (5, 4, -3),
              (3, -1, 2, 5)]
    term1 = tn.ncon(to_contract_1, idxs_1, backend="numpy")

    to_contract_2 = [A_C, A_R, A_R.conj(), H]
    idxs_2 = [(5, -2, 4),
              (2, 4, 1),
              (3, -3, 1),
              (-1, 3, 5, 2)]
    term2 = tn.ncon(to_contract_2, idxs_2, backend="numpy")

    term3 = leftmult(LH, A_C)
    term4 = rightmult(A_C, RH.T)
    A_C_prime = term1 + term2 + term3 + term4
    return A_C_prime


def apply_Hc(C, A_L, A_R, Hlist):
    """
    Compute C' via eq 16 of vumps paper (132 of tangent space methods).
    """
    H, LH, RH = Hlist
    A_Lstar = A_L.conj()
    A_C = rightmult(A_L, C)
    to_contract = [A_C, A_Lstar, A_R, A_R.conj(), H]
    idxs = [(2, 4, 1),
            (5, 4, -1),
            (3, 1, 6),
            (7, -2, 6),
            (5, 7, 2, 3)]
    term1 = tn.ncon(to_contract, idxs, backend="numpy")
    term2 = LH @ C
    term3 = C @ RH.T
    C_prime = term1 + term2 + term3
    return C_prime

#  def apply_Hc(C, A_L, A_R, Hlist):
#      """
#      Compute C' via eq 16 of vumps paper (132 of tangent space methods).
#      """
#      H, LH, RH = Hlist
#      A_Lstar = A_L.conj()
#      A_C = rightmult(A_L, C)
#      to_contract = [A_C, A_Lstar, A_R, A_R.conj(), H]
#      idxs = [(4, 1, 3),
#              (6, 1, -1),
#              (5, 3, 2),
#              (7, -2, 2),
#              (6, 7, 4, 5)]
#      term1 = tn.ncon(to_contract, idxs, backend="numpy")
#      term2 = LH @ C
#      term3 = C @ RH.T
#      C_prime = term1 + term2 + term3
#      return C_prime
