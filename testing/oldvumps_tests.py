
import numpy as np
import scipy as sp
#from scipy.sparse.linalg import LinearOperator, eigs, bicgstab, eigsh
#import sys
import copy
import bhtools.tebd.contractions as ct
import bhtools.tebd.utils as utils
from bhtools.tebd.scon import scon
from bhtools.tebd.constants import Sig_x, Sig_z
import bhtools.tebd.tm_functions as tm
import bhtools.tebd.vumps as vumps
################################################################################
# Tests
################################################################################

# Utilities for testing.
def check(verbose, passed):
    if not verbose:
        return
    if passed:
        print("Passed!")
    else:
        print("Failed!")
    return



# Tests of canonization.
def is_left_isometric(A_L, rtol=1E-5, atol=1E-8, verbose=False):
    contracted = ct.XopL(A_L)
    eye = np.eye(contracted.shape[0], dtype=A_L.dtype)
    passed = np.allclose(contracted, eye, rtol=rtol, atol=atol)
    if verbose:
        print("Testing if left isometric.")
    check(verbose, passed)
    return passed

def is_right_isometric(A_R, rtol=1E-5, atol=1E-8, verbose=False):
    contracted = ct.XopR(A_R)
    eye = np.eye(contracted.shape[0], dtype=A_R.dtype)
    passed = np.allclose(contracted, eye, rtol=rtol, atol=atol)
    if verbose:
        print("Testing if right isometric.")
    check(verbose, passed)
    return passed

def is_left_canonical(A_L, R, rtol=1E-5, atol=1E-8, verbose=False):
    is_iso = is_left_isometric(A_L, rtol=rtol, atol=atol, verbose=verbose)
    contracted = ct.XopR(A_L, X=R)
    passed = is_iso and np.allclose(contracted, R, rtol=rtol, atol=atol)
    if verbose:
        print("Testing if left canonical.")
    check(verbose, passed)
    return passed
    
def is_right_canonical(A_R, L, rtol=1E-5, atol=1E-8, verbose=False):
    is_iso = is_right_isometric(A_R, rtol=rtol, atol=atol, verbose=verbose)
    contracted = ct.XopL(A_R, X=L)
    passed = is_iso and np.allclose(contracted, L, rtol=rtol, atol=atol)
    if verbose:
        print("Testing if right canonical.")
    check(verbose, passed)
    return passed

def is_mixed_canonical(mpslist, L, R, rtol=1E-5, atol=1E-8, verbose=False):
    A_L, C, A_R = mpslist
    left_can = is_left_canonical(A_L, R, rtol=rtol, atol=atol, verbose=verbose)
    right_can = is_right_canonical(A_R, L, rtol=rtol, atol=atol, verbose=verbose)
    passed = left_can and right_can
    if verbose:
        print("Testing if mixed canonical.")
    check(verbose, passed)
    return passed

# Tests of Hc and its eigenvalues.
def testHc_eigs(chi, d=2, eta=1E-14):
    """
    Tests that the sparse and dense Hc yield the same dominant eigenvector.
    (Specifically, that they yield eigenvectors of the same eigenvalue).
    """
    h = utils.random_complex((d, d, d, d))
    h = h.reshape((d**2, d**2))
    h = 0.5*(h + np.conj(h.T))
    h = h.reshape((d, d, d, d))
    L_H = utils.random_complex((chi, chi))
    L_H = 0.5*(L_H + np.conj(L_H.T))
    R_H = utils.random_complex((chi, chi))
    R_H = 0.5*(R_H + np.conj(R_H.T))
    hlist = [h, L_H, R_H]

    A = utils.random_complex((d, chi, chi)) 
    mpslist = vumps.mixed_canonical(A)
    A_L, C, A_R = mpslist


    sparseEv = vumps.minimize_Hc(mpslist, hlist, eta).flatten()
    denseEv = vumps.Hc_dense_eigs(A_L, A_R, hlist).flatten()
    Hc = vumps.Hc_dense(A_L, A_R, hlist).reshape((chi**2, chi**2))

    Hvdense = np.dot(Hc, denseEv)/denseEv
    Hvsparse = np.dot(Hc, sparseEv)/sparseEv
    passed = np.allclose(Hvdense, Hvsparse)
    check(True, passed)

def testHc(chi, d=2):
    """
    Tests that the sparse and dense apply_Hc give the same answer on random
    input.
    """
    h = utils.random_complex((d, d, d, d))
    A = utils.random_complex((d, chi, chi)) 
    mpslist = vumps.mixed_canonical(A)
    A_L, C, A_R = mpslist
    #h = 0.5*(h + np.conj(h.T))
    # A_L = utils.random_complex((d, chi, chi)) 
    # A_R = utils.random_complex((d, chi, chi)) 
    # C = utils.random_complex((chi, chi))
    hL = utils.random_complex((chi, chi))
    hR = utils.random_complex((chi, chi))
    hlist = [h, hL, hR]

    Cp_sparse = vumps.apply_Hc(C, A_L, A_R, hlist)
    print("Sparse: ", Cp_sparse)
    Cp_dense = vumps.apply_Hc_dense(C, A_L, A_R, hlist)
    print("Dense: ", Cp_dense)
    print("*")
    norm = np.linalg.norm(Cp_sparse - Cp_dense)/chi**2
    print("Norm resid: ", norm)
    if norm < 1E-13:
        print("Passed!")
    else:
        print("Failed!")


# Tests of HAc and its eigenvalues.
def testHAc(chi, d=2):
    """
    Tests that the sparse and dense apply_HAc give the same answer on random
    input.
    """
    h = utils.random_complex((d, d, d, d))
    A = utils.random_complex((d, chi, chi)) 
    mpslist = vumps.mixed_canonical(A)
    A_L, C, A_R = mpslist
    A_C = ct.rightmult(A_L, C)

    hL = utils.random_complex((chi, chi))
    hR = utils.random_complex((chi, chi))
    hlist = [h, hL, hR]
    Acp_sparse = vumps.apply_HAc(A_C, A_L, A_R, hlist)
    #print("Sparse: ", Acp_sparse)
    Acp_dense = vumps.apply_HAc_dense(A_C, A_L, A_R, hlist)
    # print("Dense: ", Acp_dense)
    # print("*")
    norm = np.linalg.norm(Acp_sparse - Acp_dense)/chi**2
    print("Test HAc.")
    print("Norm resid: ", norm)
    if norm < 1E-13:
        print("Passed!")
    else:
        print("Failed!")

def testHAc_eigs(chi, d=2, eta=1E-14):
    """
    Tests that the sparse and dense HAc yield equivalent dominant eigenvectors.
    """
    h = utils.random_complex((d, d, d, d))
    h = h.reshape((d**2, d**2))
    h = 0.5*(h + np.conj(h.T))
    h = h.reshape((d, d, d, d))
    L_H = utils.random_complex((chi, chi))
    L_H = 0.5*(L_H + np.conj(L_H.T))
    R_H = utils.random_complex((chi, chi))
    R_H = 0.5*(R_H + np.conj(R_H.T))
    hlist = [h, L_H, R_H]

    A = utils.random_complex((d, chi, chi)) 
    mpslist = vumps.mixed_canonical(A)
    A_L, C, A_R = mpslist
    
    sparseEv = vumps.minimize_HAc(mpslist, hlist, eta).flatten()
    denseEv = vumps.HAc_dense_eigs(mpslist, hlist, eta).flatten()
    HAc = vumps.HAc_dense(A_L, A_R, hlist).reshape((d*chi**2, d*chi**2))

    Hvdense = np.dot(HAc, denseEv)/denseEv
    Hvsparse = np.dot(HAc, sparseEv)/sparseEv
    passed = np.allclose(Hvdense, Hvsparse)
    check(True, passed)


def dag(A):
    return np.conj(A.T)

def LH_test(chi, d=2, tol=1E-13):
    """
    Tests that <LH|R> = 0 where LH is the renormalized effective 
    Hamiltonian of the left infinite block of a random uMPS with
    bond dimension chi. The Hamiltonian is randomized and Hermitian.
    """
    params = vumps.vumps_params()
    params["dom_ev_approx"]=False
    params["env_tol"] = 1E-12
    enviro_params = vumps.extract_enviro_params(params, params["delta_0"])
    H = utils.random_hermitian(d*d).reshape((d,d,d,d))

    print("MIXED CANONICAL")
    mpslist, rL, lR = vumps.vumps_initial_tensor(d, chi, params)

    A_L, C, A_R = mpslist
    #rL, lR = vumps.normalized_tm_eigs(mpslist, params)
    




    # print("rL - lR:", np.linalg.norm(rL-lR))
    # print("rL - rL.T:", np.linalg.norm(rL-rL.T))
    # print("rL - dag(rL):", np.linalg.norm(rL-np.conj(rL.T)))
    # print("E: ", vumps.twositeexpect(mpslist, H))
    # hL = vumps.compute_hL(A_L, H)
    # print("<hL|R>: ", vumps.proj(hL, lR)) 

    LH = vumps.solve_for_LH(A_L, H, lR, enviro_params)
    proj = np.abs(vumps.proj(LH, lR))
    print("<LH|R>:", proj)
    if proj> tol:
        print("Failed!")
    else:
        print("Passed!")

    print("GAUGE MATCHING RANDOM AC AND C")
    mpslist, rL, lR = vumps.vumps_initial_tensor(d, chi, params)
    A_C = utils.random_unitary(d*chi)[:, :chi].reshape(
                (d, chi, chi))
    C = np.diag(utils.random_rng(chi, 0.1, 1))
    A_L, A_R = vumps.gauge_match_polar(A_C, C)
    mpslist = [A_L, C, A_R]
    print("E:", vumps.twositeexpect(mpslist, H))
    rL, lR = vumps.normalized_tm_eigs(mpslist, params)



    #hL = vumps.compute_hL(A_L, H)
    # print("E: ", vumps.twositeexpect(mpslist, H))
    # print("<hL|R>: ", vumps.proj(hL, lR)) 
    LH = vumps.solve_for_LH(A_L, H, lR, enviro_params)
    proj = np.abs(vumps.proj(LH, lR))
    print("<LH|R>:", proj)
    if proj> tol:
        print("Failed!")
    else:
        print("Passed!")
    
    print("TENSORS AFTER ONE VUMPS ITERATION")

    mpslist, rL, lR = vumps.vumps_initial_tensor(d, chi, params)
    A_L, C, A_R = mpslist
    A_C = ct.rightmult(A_L, C)
    
    environment_init = [rL, lR, None, None]
    environment = vumps.vumps_environment(mpslist, H, tol, params,
            environment_init)
    vumps_state = [False, A_C]

    mpslist, delta, vumps_state = vumps.vumps_gradient(
            mpslist, H, environment, tol, params, vumps_state) 

    environment = vumps.vumps_environment(mpslist, H, tol, params,
            environment)

    A_L, C, A_R = mpslist
    rL, lR = vumps.normalized_tm_eigs(mpslist, params)

    LH = vumps.solve_for_LH(A_L, H, lR, params)
    proj = np.abs(vumps.proj(LH, lR))
    print("<LH|R>:", proj)
    if proj> tol:
        print("Failed!")
    else:
        print("Passed!")


    
    



def RH_test(chi, d=2, tol=1E-11):
    """
    Tests that <L|R_H> = 0 where R_H is the renormalized effective 
    Hamiltonian of the right infinite block of a random uMPS with
    bond dimension chi. The Hamiltonian is randomized and Hermitian.
    """
    params = vumps.vumps_params()
    params["dom_ev_approx"]=False
    mpslist, rL, lR = vumps.vumps_initial_tensor(d, chi, params)
    A_L, C, A_R = mpslist
    # evl, evr, eVl, eVr = tm.tmeigs(A_R, nev=3, ncv=30, tol=1E-13, 
            # which="both")

    #H = utils.random_hermitian(d*d).reshape((d,d,d,d))
    H = utils.H_ising(-1.0, -0.48).reshape((d,d,d,d))

    #RH = vumps.solve_for_RH(A_R, H, rL, params)
    RH = vumps.solve_for_RH(A_R, H, lR, params)
    proj = np.abs(vumps.proj(rL, RH))
    print("<L|RH>:", proj)
    if proj> tol:
        print("Failed!")
    else:
        print("Passed!")

    print("GAUGE MATCHING RANDOM AC AND C")
    mpslist, _, _ = vumps.vumps_initial_tensor(d, chi, params)
    A_C = utils.random_unitary(d*chi)[:, :chi].reshape(
                (d, chi, chi))
    # A_C = utils.random_unitary(d*chi)[:, :chi].reshape(
                # (d, chi, chi))
    A_C = utils.random_complex((d,chi,chi))
    #C = np.diag(utils.random_rng(chi, 0.1, 1))
    C = utils.random_complex((chi,chi))
    A_L, A_R, _ = vumps.gauge_match_SVD(A_C, C, 1E-15)
    mpslist = [A_L, C, A_R]
    rL, lR = vumps.normalized_tm_eigs(mpslist, params)
    RH = vumps.solve_for_RH(A_R, H, rL, params)
    proj = np.abs(vumps.proj(rL, RH))
    print("<L|RH>:", proj)
    if proj> tol:
        print("Failed!")
    else:
        print("Passed!")
    
    print("GAUGE MATCHING CANONICAL AC AND C")
    mpslist, _, _ = vumps.vumps_initial_tensor(d, chi, params)
    A_L, C, A_R = mpslist
    A_C = ct.rightmult(A_L, C)
    A_L, A_R, _ = vumps.gauge_match_SVD(A_C, C, 1E-15)
    mpslist = [A_L, C, A_R]
    rL, lR = vumps.normalized_tm_eigs(mpslist, params)
    RH = vumps.solve_for_RH(A_R, H, rL, params)
    proj = np.abs(vumps.proj(rL, RH))
    print("<L|RH>:", proj)
    if proj> tol:
        print("Failed!")
    else:
        print("Passed!")
    
    
    print("TENSORS AFTER ONE VUMPS ITERATION")
    mpslist, rL, lR = vumps.vumps_initial_tensor(d, chi, params)
    A_L, C, A_R = mpslist
    A_C = ct.rightmult(A_L, C)
    vumps_state = [True, A_C, None, None]
    mpslist, delta, vumps_state = vumps.vumps_iteration(mpslist, H, 
            params["delta_0"], params, vumps_state)
    A_L, C, A_R = mpslist
    rL, lR = vumps.normalized_tm_eigs(mpslist, params)

    RH = vumps.solve_for_RH(A_R, H, rL, params)
    proj = np.abs(vumps.proj(rL, RH))
    print("<L|RH>:", proj)
    if proj> tol:
        print("Failed!")
    else:
        print("Passed!")


