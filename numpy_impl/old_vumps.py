import copy

import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator, eigs, eigsh, lgmres
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence

import bhtools.tebd.contractions as ct
import bhtools.tebd.utils as utils
import bhtools.tebd.tm_functions as tm

import tensornetwork as tn

import jax_vumps.writer.Writer as Writer




##########################################################################
# Manipulations of the MPS: canonicalization and expectation values.
##########################################################################

##########################################################################
# Energy variance and subspace expansion.
###########################################################################
def null_spaces(mpslist):
    """
    Return matrices spanning the null spaces of A_L and A_R, and
    the hermitian conjugates of these, reshaped into rank
    3 tensors.
    """
    AL, C, AR = mpslist
    d, chi, _ = AL.shape
    NLshp = (d, chi, (d-1)*chi)
    ALdag = ct.fuse_left(AL).T.conj()
    NLm = sp.linalg.null_space(ALdag)
    NL = NLm.reshape(NLshp)

    ARmat = ct.fuse_right(AR)
    NRm_dag = sp.linalg.null_space(ARmat)
    NRm = np.conj(NRm_dag)
    NR = NRm.reshape((d, chi, (d-1)*chi))
    NR = NR.transpose((0, 2, 1))
    return (NL, NR)


def B2_tensor(oldlist, newlist):
    NL, NR = null_spaces(oldlist)
    AL, C, AR = newlist
    AC = ct.rightmult(AL, C)
    L = ct.XopL(AC, B=np.conj(NL))
    R = ct.XopR(AR, B=np.conj(NR))
    return np.dot(L, R.T)


def B2norm(oldlist, newlist):
    B2 = B2_tensor(oldlist, newlist)
    norm = np.linalg.norm(B2)
    return norm


##########################################################################
# Canonicalization routines translated from Jutho's TNSchool code
###########################################################################
def qrmat(A, mode="full"):
    """
    QR decomp. of A, with phase convention such that R has only positive
    elements on the main diagonal. A is a matrix.
    """
    Q, R = sp.linalg.qr(A, mode=mode)
    phases = np.diag(np.sign(np.diag(R)))
    Q = np.dot(Q, phases)
    R = np.dot(np.conj(phases), R)
    return (Q, R)


def qrpos(A):
    """
    QR decomp. of A, with phase convention such that R has only positive
    elements on the main diagonal.

    If A is an MPS tensor (d, chiL, chiR), it is reshaped appropriately
    before the throughput begins. In that case, Q will be a tensor
    of the same size, while R will be a chiR x chiR matrix.
    """
    Ashp = A.shape
    if len(Ashp) == 2:
        return qrmat(A)
    elif len(Ashp) != 3:
        print("A had invalid dimensions, ", A.shape)

    A = ct.fuse_left(A)  # d*chiL, chiR
    Q, R = qrmat(A, mode="economic")
    Q = ct.unfuse_left(Q, Ashp)
    return (Q, R)


def rqmat(A, mode="full"):
    """
    RQ decomp. of A, with phase convention such that R has only positive
    elements on the main diagonal. A is a matrix.
    """
    R, Q = sp.linalg.rq(A, mode=mode)
    phases = np.diag(np.sign(np.diag(R)))
    Q = np.dot(phases, Q)
    R = np.dot(R, np.conj(phases))
    return (Q, R)


def rqpos(A):
    """
    RQ decomp. of A, with phase convention such that R has only positive
    elements on the main diagonal.

    If A is an MPS tensor (d, chiL, chiR), it is reshaped and
    transposed appropriately
    before the throughput begins. In that case, Q will be a tensor
    of the same size, while R will be a chiL x chiL matrix.
    """
    Ashp = A.shape
    if len(Ashp) == 2:
        return rqmat(A)
    elif len(Ashp) != 3:
        print("A had invalid dimensions, ", A.shape)

    A = ct.fuse_right(A)  # chiL, d*chiR
    R, Q = qrmat(A, mode="economic")
    Q = ct.unfuse_right(Q, Ashp)
    return (Q, R)


def leftorth(A, C=None, tol=1E-13, maxiter=100):
    if C is None:
        C = np.eye(A.shape[1], dtype=A.dtype)
    #print(np.trace(ct.XopL(A))/A.shape[1])
    lam2, rho = tmeigs(A, which="LM", tol=tol, direction="left",
            v0 = np.dot(np.conj(C.T), C), nev=1)
    rho = rho + np.conj(rho.T)
    rho /= np.trace(rho)
    U, S, VH = np.linalg.svd(rho)
    C = ct.leftmult(np.sqrt(S), np.conj(VH))
    _, C = qrmat(C)
    AL, R = qrpos(ct.leftmult(C, A))
    lam = np.linalg.norm(R)
    R /= lam
    numiter = 1
    ldelt = np.abs(np.abs(np.trace(ct.XopL(AL))/A.shape[1]) - 1)
    while ldelt > tol and numiter < maxiter:
        _, C = tmeigs(A, B=np.conj(AL), v0=R, tol=ldelt/10, nev=1)
        _, C = qrpos(C)
        C /= np.linalg.norm(C)
        AL, R = qrpos(ct.leftmult(C, A))
        lam = np.linalg.norm(R)
        R = R/lam
        numiter += 1
        ldelt = np.abs(np.abs(np.trace(ct.XopL(AL))/A.shape[1]) - 1)
    C = R
    return (AL, C, lam)


def rightorth(A, C=None, tol=1E-13, maxiter=100):
    if C is None:
        C = np.eye(A.shape[1], dtype=A.dtype)
    AL, C, lam = leftorth(A.transpose((0,2,1)), C=C.T, tol=tol,
            maxiter=maxiter)
    return (AL.transpose((0,2,1)), C.T, lam)


def mixed_canonical(A, tol=1E-13):
    AL, _, _ = leftorth(A, tol=tol)
    AR, C, _ = rightorth(AL, tol=tol)
    return (AL, C, AR)


def regauge(mpslist, tol=1E-13, leftonly=False):
    AL, C, AR = mpslist
    AL, C, _ = leftorth(AR, C=C, tol=tol)
    #if not leftonly:
    #    AR, C, _ = rightorth(AL, C=C, tol=tol)
    mpslist = [AL, C, AR]
    fpoints = normalized_tm_eigs(mpslist, tol=tol)
    return mpslist, fpoints






def mixed_canonical_old(A, lam=None):
    # """
    # Bring a uniform MPS tensor into mixed canonical form.
    # """
    lam, gam = tm.canonicalized_gamma(A, lam=lam)
    A_L = ct.leftmult(lam, gam)
    A_R = ct.rightmult(gam, lam)
    chi = lam.size
    C = np.zeros(chi, dtype=A.dtype)
    C[:] = lam.real[:]
    C = np.diag(C)
    mpslist = [A_L, C, A_R]
    return mpslist


def onesiteexpect(A_R, C, O, real=True, L=None, R=None):
    A_C = ct.rightmult(A_R, C)
    E = ct.chainwithops([A_C], ops=[(O, 0)], lvec=L, rvec=R)
    if real:
        if np.abs(E.imag) > 1E-14:
            print("Warning: EV had large imaginary part ", str(E.imag))
        E = E.real
    return E

# def vumps_energy(A_L, A_C, H):
    # E = ct.chainwithops([A_L, A_C], ops=[(H, 0)])
    # return E.real

def twositeexpect(mpslist, H, real=True, divide_by_norm=False, lvec=None,
        rvec=None):
    """
    The expectation value of the two-site operator H.
    """
    A_L, C, A_R = mpslist
    A_CR = ct.leftmult(C, A_R)
    E = ct.chainwithops([A_L, A_CR], ops=[(H, 0)], lvec=lvec, rvec=rvec)
    #ER = ct.chainwithops([A_CL, A_R], ops=[(H, 0)], lvec=lvec, rvec=rvec)
    if real:
        if np.abs(E.imag) > 1E-14:
            print("Warning: EV had large imaginary part ", str(E.imag))
        E = E.real
    if divide_by_norm:
        norm = compute_norm(mpslist, real=real)
        E /= norm
    return E.real


def compute_norm(mpslist, real=True):
    return twositeexpect(mpslist, None, real=real, divide_by_norm=False)

#########################################################################
# Transfer matrix manipulations.
##########################################################################

def tmeigs(A, B=None, nev=1, ncv=20, tol=1E-12, direction="right", v0=None,
        maxiter=100, which="LR"
        ):
    if B is None:
        B = np.conj(A)
    d, chi_LA, chi_RA = A.shape
    _, chi_LB, chi_RB = B.shape
    if v0 is None:
        v0 = np.zeros((chi_LB, chi_LA), dtype=A.dtype)
        np.fill_diagonal(v0, 1)
    if direction=="left":
        outshape = (chi_LB, chi_LA)
        def tmmatvec(xvec):
            xmat = xvec.reshape(outshape)
            xmat = ct.XopL(A, B=B, X=xmat)
            xvec = xmat.flatten()
            return xvec

    elif direction=="right":
        outshape = (chi_RB, chi_RA)
        def tmmatvec(xvec):
            xmat = xvec.reshape(outshape)
            xmat = ct.XopR(A, B=B, X=xmat)
            xvec = xmat.flatten()
            return xvec
    else:
        raise ValueError("Invalid 'direction', ", direction)

    op = LinearOperator((chi_LB*chi_LA, chi_RB*chi_RA),
            matvec = tmmatvec, dtype=A.dtype)
    try:
        vals, vecs = eigsh(op, k=nev, which=which, v0=v0, ncv=ncv,
                tol=tol, maxiter=maxiter)
    except ArpackNoConvergence:
        print("Warning: tmeigs didn't converge. Using higher maxiter.")
        vals, vecs = eigsh(op, k=nev, which=which, v0=v0, ncv=ncv,
                tol=tol, maxiter=10*maxiter)
        # vals = exception.eigenvalues
        # vecs = exception.eigenvectors

    #vals, vecs = utils.sortby(vals.real, vecs, mode="LR")
    indmax = np.argmax(np.abs(vals))
    v = vals[indmax]
    V = vecs[:, indmax]
    V /= np.linalg.norm(V)
    V = V.reshape(outshape)

    return (v, V)


def normalized_tm_eigs(mpslist, oldrL=None, oldlR=None, tol=1E-12):
    A_L, C, A_R = mpslist
    A_L, C, A_R = mpslist
    _, lR = tmeigs(A_L, direction="right", tol=tol)
    _, rL = tmeigs(A_R, direction="left", tol=tol)
    lR /= np.trace(lR)
    rL /= np.trace(rL)
    rL = 0.5*(rL + np.conj(rL.T))
    lR = 0.5*(lR + np.conj(lR.T))
    return (rL, lR)


def vumps_tm_eigs(mpslist, oldevs=[None, None], tol=1E-12,
        approx_mode=False):
    """
    Find, or approximate, the dominant eigenvectors of the transfer
    matrix needed for vumps.
    """
    #approx_tm = params["dom_ev_approx"]
    if not approx_mode:
        oldrL, oldlR = oldevs
        rL, lR = normalized_tm_eigs(mpslist,
                oldrL=oldrL, oldlR=oldlR,
                tol=tol)
    else:
        C = mpslist[1]
        rL = np.dot(np.conj(C.T), C)
        lR = np.dot(C, np.conj(C.T))
        lR /= np.trace(lR)
        rL /= np.trace(rL)


    return (rL, lR)

# def shift_H(H, A_L, A_C):
    # """
    # Returns htilde = h - E*I.
    # """
    # E = vumps_energy(A_L, A_C, H)
    # d = H.shape[0]
    # #Hmat = H.reshape((d*d, d*d))
    # Emat = E*np.eye(d*d, dtype=H.dtype).reshape((d,d,d,d))
    # htilde = H - Emat
    # return htilde

##########################################################################
# Functions to compute the effective left and right Hamiltonians.
##########################################################################
def proj(A, B):
    """
    Contract A with B to find <A|B>.
    """
    idxs = [ [0, 1], [0, 1] ]
    contract = [A, B]
    ans = tn.ncon(contract, idxs)
    return ans

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
    idxs = [ (2, 4, 1),
             (3, 1, -2),
             (5, 4, 7),
             (6, 7, -1),
             (5, 6, 2, 3) ]
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
    idxs = [ (2, 4, 1),
             (3, 1, -2),
             (5, 4, 7),
             (6, 7, -1),
             (5, 6, 2, 3) ]
    h_L = tn.ncon(to_contract, idxs)
    return h_L

def LH_linear_operator(A_L, lR):
    """
    Return, as a LinearOperator, the LHS of the equation found by
    summing the geometric series for
    the left environment Hamiltonian.
    """
    chi = A_L.shape[1]
    I = np.eye(chi, dtype=A_L.dtype)

    def matvec(v):
        v = v.reshape((chi, chi))
        Th_v = ct.XopL(A_L, X=v)
        vR = proj(v, lR)*I
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
                     maxiter = params["env_maxiter"],
                     inner_m = params["inner_m_lgmres"],
                     outer_k = params["outer_k_lgmres"],
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
    term2 = tm.tmdense(A_L).reshape((chi**2, chi**2))
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
    hL_div = proj(hL_bare, lR)*np.eye(hL_bare.shape[0])
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
        Lv = proj(rL, v)*Id
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
    hR_div = proj(rL, hR_bare)*np.eye(hR_bare.shape[0])
    hR = hR_bare - hR_div
    op = RH_linear_operator(A_R, rL)
    RH, info = call_solver(op, hR, params, oldRH)
    if info != 0:
        print("Warning: RH solution failed with code: "+str(info))
    # RHL = np.abs(proj(rL, RH))
    # if RHL > 1E-6:
        # print("Warning: large <L|RH> = ", str(RHL))
    return RH


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
    idxs_2 = [ (5, -2, 4),
               (2, 4, 1),
               (3, -3, 1),
               (-1, 3, 5, 2) ]
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
    #A_C = ct.rightmult(A_L, C)
    # A_C_prime = sparse_eigensolve(apply_HAc, A_C.shape, A_C,
            # params,  A_L, A_R, Hlist, hermitian=True, which="LM")
    A_C_prime = sparse_eigensolve(apply_HAc, A_C.shape, A_C,
            params, A_L, A_R, Hlist, hermitian=True, which="SA")
    # A_C_prime = sparse_eigensolve(apply_HAc, A_C.shape, A_C,
            # params, A_L, A_R, Hlist, hermitian=False, which="SR")
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


##########################################################################
# Loss functions.
##########################################################################


def compute_eL(A_L, C, A_C):
    """
    Approximate left loss function
    eL = ||A_C - A_L . C||.
    """
    eLmid = A_C - ct.rightmult(A_L, C)
    eL = np.linalg.norm(ct.fuse_left(eLmid))
    return eL


def compute_eR(A_R, C, A_C):
    """
    Approximate right loss function
    eR = ||A_C - C . A_R||
    """
    eRmid = A_C - ct.leftmult(C, A_R)
    eR = np.linalg.norm(ct.fuse_left(eRmid))
    return eR


def vumps_loss_eLeR(a_l, a_r, C, A_C):
    """
    Approximate loss function, using A_C and C instead of H_Ac(A_C)
    and H_C(C). Near optimum, this is proportional to the full loss,
    ta
    """
    eL = compute_eL(a_l, C, A_C)
    eR = compute_eR(a_r, C, A_C)
    maxe = max(eL, eR)
    print("delta2: ", maxe)
    return maxe


##########################################################################
# Gauge matching. These compute A_L and A_R that are left/right
# isometric and that, at the same time, satisfy
# A_C = A_L . C = C . A_R as closely as possible. VUMPS has converged
# w
##########################################################################
def gauge_match_SVD(A_C, C, thresh=1E-13):
    """
    Return approximately gauge-matched A_L and A_R from A_C and C
    using an SVD. If
    """
    Ashape = A_C.shape
    Cdag = np.conj(C.T)

    AC_mat_l = ct.fuse_left(A_C)
    ACl_Cd = np.dot(AC_mat_l, Cdag)
    Ul, Sl, Vld = np.linalg.svd(ACl_Cd, full_matrices=False)
    AL_mat = np.dot(Ul, Vld)
    A_L = ct.unfuse_left(AL_mat, Ashape)

    AC_mat_r = ct.fuse_right(A_C)
    d, chi, chi = Ashape
    AC_mat_r = A_C.reshape(chi, d*chi)
    Cd_ACr = np.dot(Cdag, AC_mat_r)
    Ur, Sr, Vrd = np.linalg.svd(Cd_ACr, full_matrices=False)
    AR_mat = np.dot(Ur, Vrd)
    A_R = ct.unfuse_right(AR_mat, Ashape)

    smallest = min(min(Sl), min(Sr))
    SVD_ok = smallest > thresh
    if not SVD_ok:
        print("Singular values fell beneath threshold.")
    return (A_L, A_R, SVD_ok)


def gauge_match_polar(A_C, C):
    """
    Return approximately gauge-matched A_L and A_R from A_C and C
    using a polar decomposition.
    """
    Ashape = A_C.shape
    AC_mat_l = ct.fuse_left(A_C)
    AC_mat_r = ct.fuse_right(A_C)

    UAc_l, PAc_l = sp.linalg.polar(AC_mat_l, side="right")
    UAc_r, PAc_r = sp.linalg.polar(AC_mat_r, side="left")
    UC_l, PC_l = sp.linalg.polar(C, side="right")
    UC_r, PC_r = sp.linalg.polar(C, side="left")

    A_L = np.dot(UAc_l, np.conj(UC_l.T))
    A_L = ct.unfuse_left(A_L, Ashape)
    A_R = np.dot(np.conj(UC_r.T), UAc_r)
    A_R = ct.unfuse_right(A_R, Ashape)
    return (A_L, A_R)


def gauge_match_QR(A_C, C):
    QAC, RAC = qrpos(A_C)
    QC, RC = qrpos(C)
    A_L = ct.rightmult(QAC, np.conj(RC.T))
    errL = np.linalg.norm(RAC-RC)
    QAC, LAC = rqpos(A_C)
    QC, LC = rqpos(C)
    A_R = ct.leftmult(QC.T, QAC)
    errR = np.linalg.norm(LAC-LC)
    err = max(errL, errR)
    return (A_L, A_R, err)


def gauge_match(A_C, C):
    """
    Returns A_L and A_R matched to A_C and C.
    """
    A_L, A_R = gauge_match_polar(A_C, C)
    return (A_L, A_R)


#########################################################################
# Main VUMPS loops and closely related functions.
##########################################################################
def vumps_iter(mpslist, A_C, fpoints, H, H_env, delta, params,
               vumps_state):
    mpslist, delta, A_C = vumps_gradient(mpslist, A_C, fpoints, H,
                                         H_env, delta, params)

    TMtol = params["TM_tol"]
    if params["adaptive_tm_tol"]:
        TMtol *= delta
    fpoints = vumps_tm_eigs(mpslist, oldevs=fpoints, tol=TMtol,
                            approx_mode=params["dom_ev_approx"])
    H_env = vumps_environment(mpslist, fpoints, H, delta, params,
                              H_env=H_env)
    return (mpslist, A_C, fpoints, H_env, delta, vumps_state)


def vumps_environment(mpslist, fpoints, H, delta, params,
                      H_env=[None, None]):
    A_L, C, A_R = mpslist  # lowercase means old
    rL, lR = fpoints
    lh, rh = H_env
    enviro_params = extract_enviro_params(params, delta)
    LH = solve_for_LH(A_L, H, lR, enviro_params, oldLH=lh)
    RH = solve_for_RH(A_R, H, rL, enviro_params, oldRH=rh)
    H_env = [LH, RH]
    return H_env


def vumps_gradient(mpslist, A_C, fpoints, H, H_env, delta, params):
    """
    Work loop for vumps.
    """
    a_l, c, a_r = mpslist
    rL, lR = fpoints
    LH, RH = H_env
    Hlist = [H, LH, RH]
    Heff_params = extract_Heff_params(params, delta)
    A_C = minimize_HAc(mpslist, A_C, Hlist, Heff_params)
    C = minimize_Hc(mpslist, Hlist, Heff_params)
    A_L, A_R = gauge_match(A_C, C)
    newmpslist = [A_L, C, A_R]
    delta = vumps_loss(a_l, A_C)
    return (newmpslist, delta, A_C)


##########################################################################
# Interface.
##########################################################################
def extract_enviro_params(params, delta):
    """
    Processes the VUMPS params into those specific to the
    reduced-environment solver.
    """
    keys = ["env_tol", "env_maxiter", "outer_k_lgmres", "inner_m_lgmres"]
    enviro_params = {k: params[k] for k in keys}  # Extract subset of dict

    if params["adaptive_env_tol"]:
        enviro_params["env_tol"] = enviro_params["env_tol"]*delta
    return enviro_params


def extract_Heff_params(params, delta):
    """
    Processes the VUMPS params into those specific to the
    effective Hamiltonian eigensolver.
    """
    keys = ["Heff_tol", "Heff_ncv", "Heff_neigs"]
    Heff_params = {k: params[k] for k in keys}

    if params["adaptive_Heff_tol"]:
        Heff_params["Heff_tol"] = Heff_params["Heff_tol"]*delta
    return Heff_params


def vumps_params(path="vumpsout/",
                 chi=64,
                 checkpoint_every=500,
                 delta_0=1E-1,
                 vumps_tol=1E-14,
                 maxiter=1000,
                 outdir=None,
                 adaptive_tm_tol=True,
                 TM_ncv=40,
                 TM_tol=0.1,
                 TM_tol_initial=1E-12,
                 adaptive_env_tol=True,
                 env_tol=0.01,
                 env_maxiter=100,
                 outer_k_lgmres=10,
                 inner_m_lgmres=30,
                 adaptive_Heff_tol=True,
                 Heff_tol=0.01,
                 Heff_ncv=40,
                 Heff_neigs=1
                 ):
    """
    Default arguments for vumps. Also documents the parameters.

    Nothing here should change during a VUMPS iteration.
    """
    params = dict()
    params["chi"] = chi
    params["delta_0"] = delta_0  # Initial value for the loss function.
                                 # Must be larger than tol.
    params["vumps_tol"] = vumps_tol  # Converge to this tolerance.
    params["maxiter"] = maxiter  # Maximum iterations allowed to VUMPS.
    params["outdir"] = path      # Where to save output.

    params["adaptive_tm_tol"] = adaptive_tm_tol
    params["TM_ncv"] = TM_ncv  # Number of Krylov vectors used to diagonalize
                               # the transfer matrix.
    params["TM_tol"] = TM_tol  # Tolerance when diagonalizing the transfer
                               # matrix.
    params["TM_tol_initial"] = TM_tol_initial
    params["adaptive_env_tol"] = adaptive_env_tol #See env_tol.
    params["env_tol"] = env_tol  # Solver tolerance for finding the
                                 # reduced environments. If adaptive_env_tol
                                 # is True, the solver tolerance is the
                                 # gradient norm multiplied by env_tol.
    params["env_maxiter"] = env_maxiter #The maximum number of steps used to solve
                    #for the reduced environments will be
                    #(env_maxiter+1)*innermlgmres
    params["outer_k_lgmres"] = outer_k_lgmres #Number of vectors to carry between
                            #inner GMRES iterations (when finding the
                            #reduced environments).
    params["inner_m_lgmres"] = inner_m_lgmres #Number of inner GMRES iterations per
                            #each outer iteration (when finding the
                            #reduced environments).
    params["adaptive_Heff_tol"] = adaptive_Heff_tol #See Heff_tol.
    params["Heff_tol"] = Heff_tol #Solver tolerance for finding the
                         #effective Hamiltonians. If adaptive_Heff_tol is
                         #True, the solver tolerance is the gradient
                         #norm multiplied by Heff_tol.
    params["Heff_ncv"] = Heff_ncv #Number of Krylov vectors used to diagonalize
                           #the effective Hamiltonians.
    params["Heff_neigs"] = Heff_neigs #Number of eigenvectors found when solving
                             #the effective Hamiltonians (only 1 is
                             #returned).
    return params


def default_observables():
    """
    Default observables for vumps. Also documents how to make them.
    """
    Z = lambda mymps: observers.average(mymps, [Sig_z])
    observables = [("<Z>", Z, True)] #True prints the output to console.
    return observables




def vumps(H, params):
    """
    Find the ground state of a uniform two-site Hamiltonian.
    This is the function to call from outside.
    """

    writer = make_writer(outdir=params["outdir"])
    writer.write("VUMPS! VUMPS! VUMPS/VUMPS/VUMPS/VUMPS! VUMPS!")
    chi = params["chi"]
    delta = params["delta_0"]
    tol = params["vumps_tol"]
    max_iter = params["maxiter"]

    d = H.shape[0]
    mpslist, A_C, fpoints = vumps_initial_tensor(d, chi, params,
                                                 H.dtype)

    writer.write("And so it begins..."
    H_env = vumps_environment(mpslist, fpoints, H, delta, params)
    Niter = 0
    norm = compute_norm(mpslist)
    E = twositeexpect(mpslist, H, divide_by_norm=False)
    output(writer, Niter, delta, E, 0, norm, 99)
    deltas = []
    while delta >= tol and Niter < max_iter:
        Eold = E
        oldlist = copy.deepcopy(mpslist)

        mpslist, A_C, fpoints, H_env, delta = vumps_iter(
                mpslist, A_C, fpoints, H, H_env, delta, params)

        delta.append(delta)
        Niter += 1
        dE = E - Eold
        E = twositeexpect(mpslist, H)
        norm = compute_norm(mpslist)
        B2 = B2norm(oldlist, mpslist)

        output(writer, Niter, delta, E, dE, norm, B2)


        checkevery = params["checkpoint_every"]
        if checkevery is not None and Niter % checkevery == 0:
            vumps_print(writer, "Checkpointing...")
            allout = [mpslist, A_C, H_env]
            writer.pickle(allout, Niter)

    if Niter >= max_iter:
        writer.write("Simulation terminated at max_iter.")
    writer.write("Simulation finished. Pickling results.")
    allout = [mpslist, A_C, H_env]
    writer.pickle(allout, Niter)
    return (allout, deltas)




