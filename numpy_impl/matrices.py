"""
This module contains functions that return some particular matrix or operator,
notably including Pauli matrices and Hamiltonians.
"""

import numpy as np
import jax.numpy as jnp

###############################################################################
# PAULI MATRICES
###############################################################################


def sigX(jax=True, dtype=np.float32):
    """
    Pauli X matrix.

    PARAMETERS
    ----------
    jax (bool, default True): Toggles whether a Jax or np array is returned.
    dtype: the data type of the return value.
    """
    vals = [[0, 1],
            [1, 0]]
    X = np.array(vals, dtype=dtype)
    if jax:
        X = jnp.array(X)
    return X


def sigY(jax=True, dtype=np.complex64):
    """
    Pauli Y matrix.

    PARAMETERS
    ----------
    jax (bool, default True): Toggles whether a Jax or np array is returned.
    dtype: the data type of the return value.
    """
    vals = [[0, -1],
            [1, 0]]
    Y = 1.0j*np.array(vals, dtype=dtype)
    if jax:
        Y = jnp.array(Y)
    return Y


def sigZ(jax=True, dtype=np.float32):
    """
    Pauli Z matrix.

    PARAMETERS
    ----------
    jax (bool, default True): Toggles whether a Jax or np array is returned.
    dtype: the data type of the return value.
    """
    vals = [[1, 0],
            [0, -1]]
    Z = np.array(vals, dtype=dtype)
    if jax:
        Z = jnp.array(Z)
    return Z


def sigU(jax=True, dtype=np.float32):
    """
    Pauli 'up' matrix.

    PARAMETERS
    ----------
    jax (bool, default True): Toggles whether a Jax or np array is returned.
    dtype: the data type of the return value.
    """
    vals = [[0, 1],
            [0, 0]]
    U = np.array(vals, dtype=dtype)
    if jax:
        U = jnp.array(U)
    return U


def sigD(jax=True, dtype=np.float32):
    """
    Pauli 'down' matrix.

    PARAMETERS
    ----------
    jax (bool, default True): Toggles whether a Jax or np array is returned.
    dtype: the data type of the return value.
    """
    vals = [[0, 0],
            [1, 0]]
    D = np.array(vals, dtype=dtype)
    if jax:
        D = jnp.array(D)
    return D


###############################################################################
# HAMILTONIANS
###############################################################################

def H_ising(J, h, jax=True, dtype=np.float32):
    """
    The famous and beloved transverse-field Ising model,
    H = J * XX + h * ZI

    PARAMETERS
    ----------
    J, h : couplings
    jax (bool, default True): Toggles whether a Jax or np array is returned.
    dtype: the data type of the return value.
    """
    X = sigX(jax=False, dtype=dtype)
    Z = sigZ(jax=False, dtype=dtype)
    ham = J*np.kron(X, X) + h*np.kron(Z, np.eye(2, dtype=dtype))
    ham = ham.reshape(2, 2, 2, 2)
    if jax:
        ham = jnp.array(ham)
    return ham


def H_XXZ(delta=1, ud=2, scale=1, jax=True, dtype=np.float32):
    """
    H = (-1/(8scale))*[ud*[UD + DU] + delta*ZZ]

    PARAMETERS
    ----------
    delta, ud, scale : couplings, default 1, 2, 1
    jax (bool, default True): Toggles whether a Jax or np array is returned.
    dtype: the data type of the return value.
    """
    U = sigU(jax=False, dtype=dtype)
    D = sigD(jax=False, dtype=dtype)
    Z = sigZ(jax=False, dtype=dtype)
    UD = ud * (np.kron(U, D) + np.kron(D, U))
    H = UD + delta * np.kron(Z, Z)
    H *= -(1/(8*scale))
    H = H.reshape(2, 2, 2, 2)
    if jax:
        H = jnp.array(H)
    return H


def H_XX(jax=True, dtype=np.float32):
    """
    The XX model, H = XX + YY

    PARAMETERS
    ----------
    jax (bool, default True): Toggles whether a Jax or np array is returned.
    dtype: the data type of the return value.
    """
    X = sigX(jax=False, dtype=dtype)
    Y = sigZ(jax=False, dtype=dtype)
    H = np.kron(X, X) + np.kron(Y, Y).real
    H = H.reshape(2, 2, 2, 2)
    if jax:
        H = jnp.array(H)
    return H
