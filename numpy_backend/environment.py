import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator, lgmres

import jax
import jax.numpy as jnp

import tensornetwork as tn
import jax_vumps.numpy_impl.contractions as ct
import jax_vumps.numpy_impl.mps_linalg as mps_linalg


def compute_hL(A_L, htilde):
    """
    --A_L--A_L--
    |  |____|
    |  | h  |
    |  |    |
    |-A_L*-A_L*-
    """
    A_L_d = np.conj(A_L)
    to_contract = [A_L, A_L, A_L_d, A_L_d, htilde]
    idxs = [(2, 4, 1),
            (3, 1, -2),
            (5, 4, 7),
            (6, 7, -1),
            (5, 6, 2, 3)]
    h_L = tn.ncon(to_contract, idxs)
    return h_L


def compute_hLgen(A_L1, A_L2, A_L3, A_L4, htilde):
    """
    --A_L1--A_L2--
    |  |____|
    |  | h  |
    |  |    |
    |-A_L3-A_L4-
    """
    to_contract = [A_L1, A_L2, A_L3, A_L4, htilde]
    idxs = [(2, 4, 1),
            (3, 1, -2),
            (5, 4, 7),
            (6, 7, -1),
            (5, 6, 2, 3)]
    h_L = tn.ncon(to_contract, idxs)
    return h_L


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
        Th_v = ct.XopL(A_L, X=v)
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
    hL_bare = compute_hL(A_L, H)
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


def compute_hR(A_R, htilde):
    """
     --A_R--A_R--
        |____|  |
        | h  |  |
        |    |  |
     --A_R*-A_R*-
    """
    A_R_d = np.conj(A_R)
    to_contract = [A_R, A_R, A_R_d, A_R_d, htilde]
    idxs = [(2, -2, 1),
            (3, 1, 4),
            (5, -1, 7),
            (6, 7, 4),
            (5, 6, 2, 3)]
    h_R = tn.ncon(to_contract, idxs)
    return h_R


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
        Th_v = ct.XopR(A_R, X=v)
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
    hR_bare = compute_hR(A_R, H)
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


def solve(mpslist, delta, fpoints, H, env_solver_params, H_env=None):
    if H_env is None:
        H_env = [None, None]
    lh, rh = H_env  # lowercase means 'from previous iteration'

    A_L, C, A_R = mpslist
    rL, lR = fpoints
    LH = solve_for_LH(A_L, H, lR, lh, env_solver_params)
    RH = solve_for_RH(A_R, H, rL, rh, env_solver_params)
    H_env = [LH, RH]
    return H_env
