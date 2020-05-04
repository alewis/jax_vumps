import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs, eigsh

import jax
import jax.numpy as jnp

import tensornetwork as tn

import jax_vumps.mps_linalg as mps_linalg
import jax_vumps.utils as utils
import jax_vumps.contractions as ct


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


def sparse_eigensolve(func, arr_shape, guess, params,
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
    tol = params["Heff_tol"]
    ncv = params["Heff_ncv"]
    neigs = params["Heff_neigs"]
    # print("guess: ", guess)
    # print(guess.flatten())
    if hermitian:
        w, v = eigsh(op, k=neigs, which=which, tol=tol,
                     v0=guess.flatten(), ncv=ncv)
    else:
        w, v = eigs(op, k=neigs, which=which, tol=tol,
                    v0=guess.flatten(), ncv=ncv)

    w, v = utils.sortby(w.real, v, mode="SR")
    eV = v[:, 0]
    eV = eV.reshape(arr_shape)
    return eV

###############################################################################
# Effective Hamiltonians for A_C.
###############################################################################


def apply_HAc(A_C, A_L, A_R, Hlist):
    """
    Compute A'C via eq 11 of vumps paper (131 of tangent space methods).
    """
    H, LH, RH = Hlist
    to_contract_1 = [A_L, np.conj(A_L), A_C, H]
    idxs_1 = [(2, 1, 4),
              (3, 1, -2),
              (5, 4, -3),
              (3, -1, 2, 5)]
    term1 = tn.ncon(to_contract_1, idxs_1)

    to_contract_2 = [A_C, A_R, np.conj(A_R), H]
    idxs_2 = [(5, -2, 4),
              (2, 4, 1),
              (3, -3, 1),
              (-1, 3, 5, 2)]
    term2 = tn.ncon(to_contract_2, idxs_2)

    term3 = ct.leftmult(LH, A_C)
    term4 = ct.rightmult(A_C, RH.T)
    A_C_prime = term1 + term2 + term3 + term4
    return A_C_prime


def minimize_HAc(mpslist, A_C, Hlist, params):
    """
    The dominant (most negative) eigenvector of HAc.
    """
    A_L, C, A_R = mpslist
    A_C_prime = sparse_eigensolve(apply_HAc, A_C.shape, A_C,
                                  params, A_L, A_R, Hlist, hermitian=True,
                                  which="SA")
    return A_C_prime.real


def HAc_dense(A_L, A_R, Hlist):
    """
    Construct the dense effective Hamiltonian HAc.
    """
    d, chi, _ = A_L.shape
    H, LH, RH = Hlist
    I_chi = np.eye(chi, dtype=H.dtype)
    I_d = np.eye(d, dtype=H.dtype)

    contract_1 = [A_L, np.conj(A_L), H, I_chi]
    idx_1 = [(2, 1, -5),
             (3, 1, -2),
             (3, -1, 2, -4),
             (-3, -6)
             ]
    term1 = tn.ncon(contract_1, idx_1)

    contract_2 = [I_chi, A_R, np.conj(A_R), H]
    idx_2 = [(-2, -5),
             (2, -6, 1),
             (4, -3, 1),
             (-1, 4, -4, 2)
             ]
    term2 = tn.ncon(contract_2, idx_2)

    contract_3 = [LH, I_d, I_chi]
    idx_3 = [(-2, -5),
             (-1, -4),
             (-3, -6)
             ]
    term3 = tn.ncon(contract_3, idx_3)

    contract_4 = [I_chi, I_d, RH]
    idx_4 = [(-2, -5),
             (-1, -4),
             (-6, -3)
             ]
    term4 = tn.ncon(contract_4, idx_4)

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
    A_C_p = np.dot(HAc_mat, A_Cvec).reshape(A_C.shape)
    return A_C_p


def HAc_dense_eigs(mpslist, Hlist, hermitian=True):
    """
    Construct the dense effective Hamiltonian HAc and find its dominant
    eigenvector.
    """
    A_L, C, A_R = mpslist
    HAc = HAc_dense(A_L, A_R, Hlist)
    d, chi, _ = A_L.shape
    HAc_mat = HAc.reshape((d*chi*chi, d*chi*chi))
    print("HAc Herm: ", np.linalg.norm(HAc_mat - np.conj(HAc_mat.T)))
    if hermitian:
        w, v = np.linalg.eigh(HAc_mat)
    else:
        w, v = np.linalg.eig(HAc_mat)

    #print("HAc evs: ", w[0])
    A_C = v[:, 0]
    A_C /= np.linalg.norm(A_C)
    A_C = A_C.reshape(A_L.shape)
    return A_C


###############################################################################
# Effective Hamiltonians for C.
###############################################################################
def minimize_Hc(mpslist, Hlist, params):
    """
    The dominant (most negative) eigenvector of Hc.
    """
    A_L, C, A_R = mpslist
    if len(C.shape) == 1:
        bigC = np.diag(C)
    else:
        bigC = C
    C_prime = sparse_eigensolve(apply_Hc, bigC.shape, bigC,
                                params, A_L, A_R, Hlist, hermitian=True,
                                which="SA")
    return C_prime.real


def apply_Hc(C, A_L, A_R, Hlist):
    """
    Compute C' via eq 16 of vumps paper (132 of tangent space methods).
    """
    H, LH, RH = Hlist
    A_Lstar = np.conj(A_L)
    A_C = ct.rightmult(A_L, C)
    to_contract = [A_C, A_Lstar, A_R, np.conj(A_R), H]
    idxs = [(4, 1, 3),
            (6, 1, -1),
            (5, 3, 2),
            (7, -2, 2),
            (6, 7, 4, 5)]
    term1 = tn.ncon(to_contract, idxs)
    term2 = np.dot(LH, C)
    term3 = np.dot(C, RH.T)
    C_prime = term1 + term2 + term3
    return C_prime


def Hc_dense(A_L, A_R, Hlist):
    """
    Construct Hc as a dense matrix.
    """
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
    term1 = tn.ncon(to_contract_1, idx_1)
    term2 = np.kron(Id, LH).reshape((chi, chi, chi, chi))
    term3 = np.kron(RH.T, Id).reshape((chi, chi, chi, chi))

    H_C = term1 + term2 + term3
    H_C = H_C.transpose((1, 0, 3, 2))
    return H_C


def apply_Hc_dense(C, A_L, A_R, Hlist):
    """
    Applies the dense Hc to C. Use the vec trick to reason through this for
    the dot product terms. Then match the indices in the big contraction with
    the results.
    """
    Hc = Hc_dense(A_L, A_R, Hlist)
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
    print("Hc Herm: ", np.linalg.norm(Hc - np.conj(Hc.T)))
    if hermitian:
        w, v = np.linalg.eigh(Hc)
    else:
        w, v = np.linalg.eig(Hc)
    print("Dense Hc w: ", w[0])
    C = v[:, 0]
    C /= np.linalg.norm(C)
    C = C.reshape((chi, chi))
    return C


def vumps_loss(A_L, A_C):
    """
    Norm of MPS gradient: see Appendix 4.
    """
    A_L_mat = ct.fuse_left(A_L)
    A_L_dag = np.conj(A_L_mat.T)
    N_L = mps_linalg.null_space(A_L_dag)
    N_L_dag = np.conj(N_L.T)
    A_C_mat = ct.fuse_left(A_C)
    B = np.dot(N_L_dag, A_C_mat)
    Bnorm = np.linalg.norm(B)
    return Bnorm


def apply_gradient(iter_data, delta, H, heff_krylov_params):
    """
    Work loop for vumps.
    """
    mpslist, A_C, fpoints, H_env, delta = iter_data
    a_l, c, a_r = mpslist
    rL, lR = fpoints
    LH, RH = H_env
    Hlist = [H, LH, RH]
    A_C = minimize_HAc(mpslist, A_C, Hlist, delta, heff_krylov_params)
    C = minimize_Hc(mpslist, Hlist, delta, heff_krylov_params)
    A_L, A_R = mps_linalg.gauge_match(A_C, C)
    newmpslist = [A_L, C, A_R]
    delta = vumps_loss(a_l, A_C)
    return (newmpslist, A_C, delta)
