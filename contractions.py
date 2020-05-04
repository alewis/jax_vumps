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

import numpy as np
import scipy as sp

import jax.numpy as jnp

import tensornetwork as tn


def Bop(O, B):
    """
       1
       |
       O
       |
    2--B--3
    """
    return tn.ncon([O, B],
                   [(1, -1),
                    (1, -2, -3)])


def leftmult(lam, gam):
    """
    2--lam--gam--3
            |
            1
            |
    where lam is stored 1--lam--2
    """
    out = tn.ncon([lam, gam],
                  [-2, 1],
                  [1, -1, -3])
    return out


def rightmult(gam, lam):
    """
    2--gam--lam--3
       |
       1
       |
    """
    out = tn.ncon([gam, lam],
                  [-1, -2, 1],
                  [1, -3])
    return out


def gauge_transform(gl, A, gr):
    """
            |
            1
     2--gl--A--gr--3
    """
    return rightmult(leftmult(gl, A), gr)


###############################################################################
# Chain contractors - MPS.
###############################################################################


#  def leftchain(As):
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
#      for A, B in zip(As, Bs):
#          X = XopL(A, B=B, X=X)
#      return X


#  def chainnorm(As, Bs=None, X=None, Y=None):
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
    idxs = [[0, 1], [0, 1]]
    contract = [A, B]
    ans = tn.ncon(contract, idxs)
    return ans

# *****************************************************************************
# Single site to open legs.
# *****************************************************************************
def XL(A, X):
    """
      |---A---2
      |   |
      X   |
      |   |
      |---A---1
    """
    B = A.conj()
    A = leftmult(X, A)
    idx = [(2, 1, -2),
           (2, 1, -1)]
    return tn.ncon([A, B], idx)


def XR(A, X):
    """
      2---A---|
          |   |
          |   X
          |   |
      1---A---|
    """
    B = A.conj()
    B = rightmult(B, X)
    idx = [(2, -2, 1),
           (2, -1, 1)]
    return tn.ncon([A, B], idx)

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
    rholoc = tn.ncon(to_contract, idxs).reshape((d**2, d**2))
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
    return tn.ncon(to_contract, idxs)


def twositeexpect(left, right, U):
    rho = rholoc(left, right)
    idxs = [(1, 2, 3, 4), (1, 2, 3, 4)]
    expect = tn.ncon([rho, U], idxs).real
    return expect


def tmdense(A):
    """
    2-A-4
      |
      |
    1-A-3
    """
    idxs = [[1, -2, -4], [1, -1, -3]]
    out = tn.ncon([A. A.conj()], idxs)
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
    h_L = tn.ncon(to_contract, idxs)
    return h_L


#  def compute_hLgen(A_L1, A_L2, A_L3, A_L4, htilde):
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
    h_R = tn.ncon(to_contract, idxs)
    return h_R
