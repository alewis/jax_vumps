import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs, eigsh

import tensornetwork as tn

import jax_vumps.contractions as ct
import jax_vumps.numpy_backend.mps_linalg as mps_linalg


###############################################################################
# Sparse solver functions.
###############################################################################
def sparse_solver_op(func, arr_shape, *args, dtype=np.complex128, **kwargs):
    """
    A LinearOperator is returned that applies func(arr), in
    preparation for interface with a sparse solver.

    The solver will input a flattened arr, but func will usually expect
    a higher-rank object. The necessary shape is given as arr_shape.

    *args and **kwargs are passed to func.
    """
    flat_shape = np.prod(np.array(arr_shape))
    op_shape = (flat_shape, flat_shape)

    def solver_interface(x):
        x = x.reshape(arr_shape)
        new_x = func(x, *args, **kwargs)
        new_x = new_x.flatten()
        return new_x
    op = LinearOperator(op_shape, matvec=solver_interface, dtype=dtype)
    return op


def sparse_eigensolve(func, tol, arr_shape, guess, params,
                      *args,
                      hermitian=True, which="LM",
                      **kwargs):
    """
    Returns the dominant eigenvector of the linear map f(x) implicitly
    represented by func(x, *args, **kwargs).

    Internally within func, x has the shape arr_shape. It, along with the
    guess, will be reshaped appropriately.

    The eigenvector is computed by Arnoldi iteration to a tolerance tol.
    If tol is None, machine precision of guess's data type is used.

    The param 'sigma' is passed to eigs. Eigenvalues are found 'around it'.
    This can be used to find the most negative eigenvalue instead of the
    one with the largest magnitude, for example.
    """
    op = sparse_solver_op(func, arr_shape, *args, dtype=guess.dtype,
                          **kwargs)
    ncv = params["n_krylov"]
    maxiter = params["max_restarts"]
    neigs = 1
    # print("guess: ", guess)
    # print(guess.flatten())
    if hermitian:
        w, v = eigsh(op, k=neigs, which=which, tol=tol,
                     v0=guess.flatten(), ncv=ncv, maxiter=maxiter)
    else:
        w, v = eigs(op, k=neigs, which=which, tol=tol,
                    v0=guess.flatten(), ncv=ncv, maxiter=maxiter)

    w, v = mps_linalg.sortby(w.real, v, mode="SR")
    eV = v[:, 0]
    eV = eV.reshape(arr_shape)
    return eV


###############################################################################
# Effective Hamiltonians for A_C.
###############################################################################
def minimize_HAc(mpslist, A_C, Hlist, delta, params):
    """
    The dominant (most negative) eigenvector of HAc.
    """
    A_L, C, A_R = mpslist
    tol = params["tol_coef"]*delta
    A_C_prime = sparse_eigensolve(ct.apply_HAc, tol, A_C.shape, A_C,
                                  params, A_L, A_R, Hlist, hermitian=True,
                                  which="SA")
    return A_C_prime.real


def HAc_dense_eigs(mpslist, Hlist, hermitian=True):
    """
    Construct the dense effective Hamiltonian HAc and find its dominant
    eigenvector.
    """
    A_L, C, A_R = mpslist
    HAc = ct.HAc_dense(A_L, A_R, Hlist)
    d, chi, _ = A_L.shape
    HAc_mat = HAc.reshape((d*chi*chi, d*chi*chi))
    w, v = np.linalg.eigh(HAc_mat)
    A_C = v[:, 0].reshape(A_L.shape)
    return A_C


###############################################################################
# Effective Hamiltonians for C.
###############################################################################
def minimize_Hc(mpslist, Hlist, delta, params):
    """
    The dominant (most negative) eigenvector of Hc.
    """
    A_L, C, A_R = mpslist
    if len(C.shape) == 1:
        bigC = np.diag(C)
    else:
        bigC = C
    tol = params["tol_coef"]*delta
    C_prime = sparse_eigensolve(ct.apply_Hc, tol, bigC.shape, bigC,
                                params, A_L, A_R, Hlist, hermitian=True,
                                which="SA")
    return C_prime.real




def apply_Hc_dense(C, A_L, A_R, Hlist):
    """
    Applies the dense Hc to C. Use the vec trick to reason through this for
    the dot product terms. Then match the indices in the big contraction with
    the results.
    """
    Hc = ct.Hc_dense(A_L, A_R, Hlist)
    Cvec = C.flatten()
    chi = C.shape[0]
    Hc_mat = Hc.reshape((chi**2, chi**2))
    Cp = np.dot(Hc_mat, Cvec)
    Cp = Cp.reshape((chi, chi))
    return Cp


def Hc_dense_eigs(A_L, A_R, Hlist, hermitian=True):
    """
    Find the dominant eigenvector of the dense matrix Hc.
    """
    Hc = Hc_dense(A_L, A_R, Hlist)
    chi = A_L.shape[1]
    Hc = Hc.reshape((chi*chi, chi*chi))
    print("Hc Herm: ", mps_linalg.norm(Hc - Hc.T.conj()))
    if hermitian:
        w, v = np.linalg.eigh(Hc)
    else:
        w, v = np.linalg.eig(Hc)
    print("Dense Hc w: ", w[0])
    C = v[:, 0]
    C /= mps_linalg.norm(C)
    C = C.reshape((chi, chi))
    return C



