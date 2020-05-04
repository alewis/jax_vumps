import numpy as np
import scipy as sp

from scipy.sparse.linalg import LinearOperator, lgmres

import tensornetwork as tn
import jax_vumps.contractions as ct
#  import jax_vumps.numpy_backend.mps_linalg as mps_linalg


def LH_linear_operator(A_L, lR):
    """
    Return, as a LinearOperator, the LHS of the equation found by
    summing the geometric series for
    the left environment Hamiltonian.
    """
    chi = A_L.shape[1]
    Id = np.eye(chi, dtype=A_L.dtype)

    def matvec(v):
        v = v.reshape((chi, chi))
        Th_v = ct.XL(A_L, v)
        vR = ct.proj(v, lR)*Id
        v = v - Th_v + vR
        v = v.flatten()
        return v

    op = LinearOperator((chi**2, chi**2), matvec=matvec, dtype=A_L.dtype)
    return op


def call_solver(op, hI, params, x0):
    """
    Code used by both solve_for_RH and solve_for_LH to call the
    sparse solver.
    """
    if x0 is not None:
        x0 = x0.flatten()
    x, info = lgmres(op,
                     hI.flatten(),
                     tol=params["env_tol"],
                     maxiter=params["env_maxiter"],
                     inner_m=params["inner_m_lgmres"],
                     outer_k=params["outer_k_lgmres"],
                     x0=x0)
    new_hI = x.reshape(hI.shape)
    return (new_hI, info)


def outermat(A, B):
    chi = A.shape[0]
    contract = [A, B]
    idxs = [[-2, -1], [-3, -4]]
    return tn.ncon(contract, idxs).reshape((chi**2, chi**2))


def dense_LH_op(A_L, lR):
    chi = A_L.shape[1]
    eye = np.eye(chi, dtype=A_L.dtype)
    term1 = outermat(eye, eye)
    term2 = ct.tmdense(A_L).reshape((chi**2, chi**2))
    term3 = outermat(eye, lR)
    mat = term1-term2+term3
    mat = mat.T
    return mat


def solve_for_LH(A_L, H, lR, params, oldLH=None,
                 dense=False):
    """
    Find the renormalized left environment Hamiltonian using a sparse
    solver.
    """
    hL_bare = ct.compute_hL(A_L, H)
    hL_div = ct.proj(hL_bare, lR)*np.eye(hL_bare.shape[0])
    hL = hL_bare - hL_div
    chi = hL.shape[0]

    if dense:
        mat = dense_LH_op(A_L, lR)
        op = LH_linear_operator(A_L, lR)
        LH = sp.linalg.solve(mat.T, hL.reshape((chi**2)))
        LH = LH.reshape((chi, chi))
    else:

        op = LH_linear_operator(A_L, lR)
        LH, info = call_solver(op, hL, params, oldLH)
        if info != 0:
            print("Warning: Hleft solution failed with code: "+str(info))
    return LH




def RH_linear_operator(A_R, rL):
    chi = A_R.shape[1]
    """
    Return, as a LinearOperator, the LHS of the equation found by
    summing the geometric series for
    the right environment Hamiltonian.
    """
    Id = np.eye(chi, dtype=A_R.dtype)

    def matvec(v):
        v = v.reshape((chi, chi))
        Th_v = ct.XR(A_R, v)
        Lv = ct.proj(rL, v)*Id
        v = v - Th_v + Lv
        v = v.flatten()
        return v
    op = LinearOperator((chi**2, chi**2), matvec=matvec, dtype=A_R.dtype)
    return op


def solve_for_RH(A_R, H, rL, params,
                 oldRH=None):
    """
    Find the renormalized right environment Hamiltonian using a sparse
    solver.
    """
    hR_bare = ct.compute_hR(A_R, H)
    hR_div = ct.proj(rL, hR_bare)*np.eye(hR_bare.shape[0])
    hR = hR_bare - hR_div
    op = RH_linear_operator(A_R, rL)
    RH, info = call_solver(op, hR, params, oldRH)
    if info != 0:
        print("Warning: RH solution failed with code: "+str(info))
    # RHL = np.abs(ct.proj(rL, RH))
    # if RHL > 1E-6:
        # print("Warning: large <L|RH> = ", str(RHL))
    return RH


