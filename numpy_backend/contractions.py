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
    return np.dot(lam, gam)


def rightmult(gam, lam):
    """
    2--gam--lam--3
       |
       1
       |
    """
    return np.dot(gam, lam)


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


def leftchain(As):
    """
      |---A1---A2---...-AN-2
      |   |    |         |
      |   |    |   ...   |
      |   |    |         |
      |---B1---B2---...-BN-1
      (note: AN and BN will be contracted into a single matrix)

      As is a list of MPS tensors.
    """
    Bs = [jnp.conj(A) for A in As]
    for A, B in zip(As, Bs):
        X = XopL(A, B=B, X=X)
    return X


def chainnorm(As, Bs=None, X=None, Y=None):
    """
      |---A1---A2---...-AN---
      |   |    |        |   |
      X   |    |   ...  |   Y
      |   |    |        |   |
      |---B1---B2---...-BN---
    """
    X = leftchain(As, Bs=Bs, X=X)
    if Y is not None:
        X = np.dot(X, Y.T)
    return np.trace(X)


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
def XopL(A, O=None, X=None, B=None):
    """
      |---A---2
      |   |
      X   O
      |   |
      |---B---1
    """
    if B is None:
        B = np.conj(A)
    A = leftmult(X, A)
    if O is not None:
        B = Bop(O, B)
    idx = [(2, 1, -2),
           (2, 1, -1)]
    return tn.ncon([A, B], idx)


def XopR(A, O=None, X=None, B=None):
    """
      2---A---|
          |   |
          O   X
          |   |
      1---B---|
    """
    if B is None:
        B = np.conj(A)
    B = rightmult(B, X)
    if O is not None:
        B = Bop(O, B)
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
    return np.einsum("ibd, iac", A, A.conj())
