import numpy as np
import jax.numpy as jnp
import jax_vumps.vumps as vumps
import jax_vumps.numpy_backend.contractions as ct
import jax_vumps.numpy_backend.mps_linalg as np_linalg
import jax_vumps.numpy_backend.heff as np_heff
import jax_vumps.jax_backend.heff as jax_heff

import jax_vumps.jax_backend.environment as jax_env
import jax_vumps.numpy_backend.environment as np_env

import mps_flrw.bhtools.tebd.vumps as old_vumps


###############################################################################
# Tests
###############################################################################
def random_rng(shp, low=0, high=1.0):
    return (high - low) * np.random.random_sample(shp) + low


def random_complex(shp, real_low=-1.0, real_high=1.0, imag_low=-1.0,
                   imag_high=1.0):
    """
    Return a normalized randomized complex array of shape shp.
    """
    realpart = random_rng(shp, low=real_low, high=real_high)
    imagpart = 1.0j * random_rng(shp, low=imag_low, high=imag_high)
    bare = realpart + imagpart
    return bare


# Utilities for testing.
def check(verbose, lhs, rhs, thresh=1E-6):
    err = np.linalg.norm(np.abs(lhs - rhs))/lhs.size
    passed = err < thresh
    if verbose:
        if passed:
            print("Passed!")
        else:
            print("Failed by ", err)
    return (passed, err)


def is_left_isometric(A_L, rtol=1E-5, atol=1E-8, verbose=False):
    """
    Passes if A_L is left-isometric.
    """
    contracted = ct.XopL(A_L)
    eye = np.eye(contracted.shape[0], dtype=A_L.dtype)
    passed, err = check(verbose, contracted, eye, thresh=atol)
    return (passed, err)


def is_right_isometric(A_R, rtol=1E-5, atol=1E-8, verbose=False):
    """
    Passes if A_R is right-isometric.
    """
    contracted = ct.XopR(A_R)
    eye = np.eye(contracted.shape[0], dtype=A_R.dtype)
    passed, err = check(verbose, contracted, eye, thresh=atol)
    return (passed, err)


def is_left_canonical(A_L, R, rtol=1E-5, atol=1E-8, verbose=False):
    """
    Passes if A_L is left-canonical.
    """
    if verbose:
        print("Testing if left isometric, ")
    is_iso, iso_err = is_left_isometric(A_L, rtol=rtol, atol=atol,
                                        verbose=verbose)
    contracted = ct.XopR(A_L, X=R)
    if verbose:
        print("Testing if left canonical, ")
    is_can, can_err = check(verbose, contracted, R, thresh=atol)
    passed = is_iso and is_can
    err = iso_err + can_err
    return (passed, err)


def is_right_canonical(A_R, L, rtol=1E-5, atol=1E-8, verbose=False):
    """
    Passes if A_R is right-canonical.
    """
    if verbose:
        print("Testing if right isometric.")
    is_iso, iso_err = is_right_isometric(A_R, rtol=rtol, atol=atol,
                                         verbose=verbose)
    contracted = ct.XopL(A_R, X=L)
    if verbose:
        print("Testing if right canonical.")
    is_can, can_err = check(verbose, contracted, L, thresh=atol)
    passed = is_iso and is_can
    err = iso_err + can_err
    return (passed, err)


def is_mixed_canonical(mpslist, L, R, rtol=1E-5, atol=1E-8, verbose=False):
    """
    Passes if A_L, C, A_R is mixed-canonical.
    """
    A_L, C, A_R = mpslist
    if verbose:
        print("Testing if mixed canonical.")
    left_can, left_err = is_left_canonical(A_L, R, rtol=rtol, atol=atol,
                                           verbose=verbose)
    right_can, right_err = is_right_canonical(A_R, L, rtol=rtol, atol=atol,
                                              verbose=verbose)
    passed = left_can and right_can
    err = left_err + right_err
    return (passed, err)


def testHc(chi, d=2):
    """
    Tests that the sparse and dense apply_Hc give the same answer on random
    input.
    """
    h = random_complex((d, d, d, d))
    A = random_complex((d, chi, chi))
    mpslist = vumps.mixed_canonical(A)
    A_L, C, A_R = mpslist
    hL = random_complex((chi, chi))
    hR = random_complex((chi, chi))
    hlist = [h, hL, hR]

    Cp_sparse = ct.apply_Hc(C, A_L, A_R, hlist)
    print("Sparse: ", Cp_sparse)
    Cp_dense = ct.apply_Hc_dense(C, A_L, A_R, hlist)
    print("Dense: ", Cp_dense)
    print("*")
    norm = np.linalg.norm(Cp_sparse - Cp_dense)/chi**2
    print("Norm resid: ", norm)
    if norm < 1E-13:
        print("Passed!")
    else:
        print("Failed!")


# Tests of HAc and its eigenvalues.
def testHAc(chi, d=2, thresh=1E-7):
    """
    Tests that the sparse and dense apply_HAc give the same answer on random
    input.
    """
    h = random_complex((d, d, d, d))
    mpslist, A_C, _ = vumps.vumps_initialization(d, chi, dtype=np.complex64)
    A_L, C, A_R = mpslist

    hL = random_complex((chi, chi))
    hR = random_complex((chi, chi))
    hlist = [h, hL, hR]
    Acp_sparse = ct.apply_HAc(A_C, A_L, A_R, hlist)
    Acp_dense = ct.apply_HAc_dense(A_C, A_L, A_R, hlist)
    norm = np.linalg.norm(Acp_sparse - Acp_dense)/chi**2
    print("*******************************************************")
    print("Test HAc.")
    print("Norm resid: ", norm)
    if norm < thresh:
        print("Passed!")
    else:
        print("Failed!")


def do_compare_vumps_gradient(chi, d=2, dtype=np.float32, thresh=1E-7):
    print("Comparing old and new vumps gradient.")
    mpslist, A_C, fpoints = vumps.vumps_initialization(d, chi, dtype=dtype)
    delta = 1E-8
    H = random_complex((d*d, d*d))
    H = 0.5*(H + H.T.conj()).reshape((d, d, d, d))
    old_params = old_vumps.vumps_params()
    H_env0 = random_complex((chi, chi))
    H_env0 = 0.5*(H_env0 + H_env0.T.conj())
    H_env1 = random_complex((chi, chi))
    H_env1 = 0.5*(H_env1 + H_env1.T.conj())
    H_env = [H_env0, H_env1]
    oldmps, olddelta, oldAC = old_vumps.vumps_gradient(mpslist, A_C, fpoints,
                                                       H, H_env, delta,
                                                       old_params)
    oldout = oldmps + [np.array(olddelta)] + [oldAC]

    new_params = vumps.krylov_params()
    iter_data = [mpslist, A_C, fpoints, H_env]
    newmps, newAC, newdelta, _ = vumps.apply_gradient(iter_data, delta, H,
                                                      new_params)
    newout = newmps + [np.array(newdelta)] + [newAC]
    errs = [np.linalg.norm(np.abs(old - new))/old.size for old, new in
            zip(oldout, newout)]
    err = np.sum(errs)
    passed = True
    if err > thresh:
        passed = False
        print("FAILED with errors :", errs)
    else:
        print("Passed!")
    return (passed, err)


def do_compare_vumps_gauge_match(chi, d=2, dtype=np.complex128, thresh=1E-7):
    print("Comparing old and new gauge matching.")
    A_C = random_complex((d, chi, chi)).astype(dtype)
    C = random_complex((chi, chi)).astype(dtype)
    newout = np_linalg.gauge_match(A_C, C)
    oldout = old_vumps.gauge_match(A_C, C)
    errs = [np.linalg.norm(np.abs(old - new)) for old, new in
            zip(oldout, newout)]
    err = np.sum(errs)
    passed = True
    if err > thresh:
        passed = False
        print("FAILED with errors :", errs)
    else:
        print("Passed!")
    return (passed, err)


def do_compare_vumps_minimize_HAc(chi, d=2, dtype=np.complex128, thresh=1E-7):
    print("Comparing old and new minimize HAc.")
    mpslist, A_C, fpoints = vumps.vumps_initialization(d, chi, dtype=dtype)
    delta = 1E-3
    H = random_complex((d*d, d*d))
    H = 0.5*(H + H.T.conj()).reshape((d, d, d, d)).astype(dtype)
    H_env0 = random_complex((chi, chi)).astype(dtype)
    H_env0 = 0.5*(H_env0 + H_env0.T.conj())
    H_env1 = random_complex((chi, chi))
    H_env1 = 0.5*(H_env1 + H_env1.T.conj()).astype(dtype)
    Hlist = [H, H_env0, H_env1]

    old_params = old_vumps.vumps_params()
    old_heff_params = old_vumps.extract_Heff_params(old_params, delta)
    oldw, old_AC = old_vumps.minimize_HAc(mpslist, A_C, Hlist, old_heff_params)
    print(oldw)

    new_params = vumps.krylov_params()
    neww, newA_C = np_heff.minimize_HAc(mpslist, A_C, Hlist, delta,
                                        new_params)
    print(neww)

    err = np.linalg.norm(np.abs(old_AC - newA_C))
    passed = True
    if err > thresh:
        passed = False
        print("FAILED with error :", err)
    else:
        print("Passed!")
    return (passed, err)


def do_compare_vumps_minimize_Hc(chi, d=2, dtype=np.complex128, thresh=1E-7):
    print("Comparing old and new minimize Hc.")
    mpslist, A_C, fpoints = vumps.vumps_initialization(d, chi, dtype=dtype)
    delta = 1E-3
    H = random_complex((d*d, d*d)).astype(dtype)
    H = 0.5*(H + H.T.conj()).reshape((d, d, d, d))
    H_env0 = random_complex((chi, chi)).astype(dtype)
    H_env0 = 0.5*(H_env0 + H_env0.T.conj())
    H_env1 = random_complex((chi, chi)).astype(dtype)
    H_env1 = 0.5*(H_env1 + H_env1.T.conj())
    Hlist = [H, H_env0, H_env1]

    old_params = old_vumps.vumps_params()
    old_heff_params = old_vumps.extract_Heff_params(old_params, delta)
    oldw, old_C = old_vumps.minimize_Hc(mpslist, Hlist, old_heff_params)

    new_params = vumps.krylov_params()
    neww, newC = np_heff.minimize_Hc(mpslist, Hlist, delta, new_params)

    err = np.linalg.norm(np.abs(old_C - newC))
    passed = True
    if err > thresh:
        passed = False
        print("FAILED with error :", err)
    else:
        print("Passed!")
    return (passed, err)


def do_compare_vumps_apply_HAc(chi, d=2, dtype=np.complex128, thresh=1E-7):
    print("Comparing old and new apply Hc.")
    A_L = random_complex((d, chi, chi)).astype(dtype)
    A_R = random_complex((d, chi, chi)).astype(dtype)
    A_C = random_complex((d, chi, chi)).astype(dtype)

    H = random_complex((d, d, d, d)).astype(dtype)
    H_env0 = random_complex((chi, chi)).astype(dtype)
    H_env1 = random_complex((chi, chi)).astype(dtype)
    Hlist = [H, H_env0, H_env1]

    newC = ct.apply_HAc(A_C, A_L, A_R, Hlist)
    oldC = old_vumps.apply_HAc(A_C, A_L, A_R, Hlist)

    err = np.linalg.norm(np.abs(oldC - newC))
    passed = True
    if err > thresh:
        passed = False
        print("FAILED with error :", err)
    else:
        print("Passed!")
    return (passed, err)


def do_compare_vumps_compute_hR(chi, d=2, dtype=np.complex128, thresh=1E-7):
    print("Comparing old and new compute hR.")
    A_R = random_complex((d, chi, chi)).astype(dtype)
    H = random_complex((d, d, d, d)).astype(dtype)

    newhR = ct.compute_hR(A_R, H)
    oldhR = old_vumps.compute_hR(A_R, H)

    err = np.linalg.norm(np.abs(newhR - oldhR))
    passed = True
    if err > thresh:
        passed = False
        print("FAILED with error :", err)
    else:
        print("Passed!")
    return (passed, err)


def do_compare_vumps_compute_hL(chi, d=2, dtype=np.complex128, thresh=1E-7):
    print("Comparing old and new compute hL.")
    A_L = random_complex((d, chi, chi)).astype(dtype)
    H = random_complex((d, d, d, d)).astype(dtype)

    newhL = ct.compute_hL(A_L, H)
    oldhL = old_vumps.compute_hL(A_L, H)

    err = np.linalg.norm(np.abs(newhL - oldhL))
    passed = True
    if err > thresh:
        passed = False
        print("FAILED with error :", err)
    else:
        print("Passed!")
    return (passed, err)


def do_compare_vumps_apply_Hc(chi, d=2, dtype=np.complex128, thresh=1E-7):
    print("Comparing old and new apply Hc.")
    C = random_complex((chi, chi)).astype(dtype)
    A_L = random_complex((d, chi, chi)).astype(dtype)
    A_R = random_complex((d, chi, chi)).astype(dtype)

    H = random_complex((d, d, d, d)).astype(dtype)
    H_env0 = random_complex((chi, chi)).astype(dtype)
    H_env1 = random_complex((chi, chi)).astype(dtype)
    Hlist = [H, H_env0, H_env1]

    newC = ct.apply_Hc(C, A_L, A_R, Hlist)
    oldC = old_vumps.apply_Hc(C, A_L, A_R, Hlist)

    err = np.linalg.norm(np.abs(oldC - newC))
    passed = True
    if err > thresh:
        passed = False
        print("FAILED with error :", err)
    else:
        print("Passed!")
    return (passed, err)

def do_compare_vumps_environment(chi, d=2, dtype=np.complex128, thresh=1E-7):
    print("Comparing old and new vumps environment.")
    mpslist, A_C, fpoints = vumps.vumps_initialization(d, chi, dtype=dtype)
    delta = 1E-3
    H = random_complex((d*d, d*d)).astype(dtype)
    H = 0.5*(H + H.T.conj()).reshape((d, d, d, d)).astype(dtype)
    old_params = old_vumps.vumps_params()
    old_H_env = old_vumps.vumps_environment(mpslist, fpoints, H, delta,
                                            old_params)
    new_params = vumps.solver_params()
    new_H_env, _ = vumps.solve_environment(mpslist, delta, fpoints, H,
                                           new_params)
    errs = [np.linalg.norm(np.abs(old - new)) for old, new in
            zip(old_H_env, new_H_env)]
    passed = True
    err = np.sum(errs)
    if err > thresh:
        passed = False
        print("FAILED with errors :", errs)
    else:
        print("Passed!")
    return (passed, err)


def do_test_vumps_initialize():
    """
    Tests that vumps_intialization produces tensors in mixed canonical form.
    """
    d = 2
    chi = 8
    dtype = np.complex64
    mpslist, A_C, fpoints = vumps.vumps_initialization(d, chi, dtype=dtype)
    L, R = fpoints
    passed = is_mixed_canonical(mpslist, L, R, verbose=True)
    if not passed:
        print("Failed!")


def test_LH_matvec(d, chi, dtype=np.float32, thresh=1E-6):
    print("Testing the LH matvec operation: ")
    shapes = [ (d, chi, chi), (d, d, d, d), (chi, chi), (chi, chi) ]
    A_L, H, lR, x0 = np_linalg.random_tensors(shapes, dtype)
    x0 = np.ones((chi, chi))
    op = np_env.LH_linear_operator(A_L, lR)
    vn = op.matvec(x0.flatten())

    A_Lj, Hj, lRj, x0j = [jnp.array(x) for x in [A_L, H, lR, x0]]
    vj = jax_env.LH_matvec(lRj, A_Lj, x0j.flatten())

    err = np.linalg.norm(np.abs(vn - vj))/vn.size
    if err > thresh or jnp.any(jnp.isnan(vj)):
        print("FAILED with err: ", err)
    else:
        print("Passed!")


def test_RH_matvec(d, chi, dtype=np.float32, thresh=1E-6):
    print("Testing the RH matvec operation: ")
    shapes = [ (d, chi, chi), (d, d, d, d), (chi, chi), (chi, chi) ]
    A_R, H, rL, x0 = np_linalg.random_tensors(shapes, dtype)
    x0 = np.ones((chi, chi))
    op = np_env.RH_linear_operator(A_R, rL)
    vn = op.matvec(x0.flatten())

    A_Rj, Hj, rLj, x0j = [jnp.array(x) for x in [A_R, H, rL, x0]]
    vj = jax_env.RH_matvec(rLj, A_Rj, x0j.flatten())

    err = np.linalg.norm(np.abs(vn - vj))/vn.size
    if err > thresh or jnp.any(jnp.isnan(vj)):
        print("FAILED with err: ", err)
    else:
        print("Passed!")


def test_solve_for_LH(d, chi, dtype=np.float32, thresh=1E-4):
    print("Testing the LH linear solve: ")
    shapes = [ (d, chi, chi), (d, d, d, d), (chi, chi) ]
    A_L, H, lR = np_linalg.random_tensors(shapes, dtype)
    params = vumps.krylov_params()
    delta = thresh
    npLH = np_env.solve_for_LH(A_L, H, lR, params, delta)

    A_Lj, Hj, lRj = [jnp.array(x) for x in [A_L, H, lR]]
    jaxLH = jax_env.solve_for_LH(A_Lj, Hj, lRj, params, delta)
    err = np.linalg.norm(np.abs(jaxLH - npLH))/jaxLH.size
    if err > thresh or jnp.any(jnp.isnan(jaxLH)):
        print("FAILED with err: ", err)
    else:
        print("Passed!")


def test_solve_for_RH(d, chi, dtype=np.float32, thresh=1E-4):
    print("Testing the RH linear solve: ")
    shapes = [ (d, chi, chi), (d, d, d, d), (chi, chi) ]
    A_R, H, rL = np_linalg.random_tensors(shapes, dtype)
    params = vumps.krylov_params()
    delta = thresh
    npRH = np_env.solve_for_RH(A_R, H, rL, params, delta)

    A_Rj, Hj, rLj = [jnp.array(x) for x in [A_R, H, rL]]
    jaxRH = jax_env.solve_for_RH(A_Rj, Hj, rLj, params, delta)
    err = np.linalg.norm(np.abs(jaxRH - npRH))/jaxRH.size
    if err > thresh or jnp.any(jnp.isnan(jaxRH)):
        print("FAILED with err: ", err)
    else:
        print("Passed!")



def test_minimize_Hc(d, chi, thresh=1E-4):
    print("Test minimize Hc")
    delta = thresh
    params = vumps.krylov_params(n_krylov=10, max_restarts=1)
    shapes = [(d, chi, chi), (chi, chi), (d, chi, chi), (d, d, d, d),
              (chi, chi), (chi, chi), (chi, chi)]
    A_L, C, A_R, H, LH, RH, A_C = np_linalg.random_tensors(shapes)
    H = H.reshape((d**2, d**2))
    H = 0.5*(H + H.conj().T).reshape((d, d, d, d))
    LH = 0.5*(LH + LH.conj().T)
    RH = 0.5*(RH + RH.conj().T)
    Hlist = [H, LH, RH]
    mpslist = [A_L, C, A_R]
    jevC, jC = jax_heff.minimize_Hc(mpslist, Hlist, delta, params)
    jevC = float(jevC)
    jC = np.array(jC)
    print("Jax ev: ", jevC)
    print("Err jaxC: ", np.linalg.norm(ct.apply_Hc(jC, A_L, A_R, Hlist) - jevC*jC))

    #  A_Lj, Cj, A_Rj, Hj, LHj, RHj = [jnp.array(x) for x in [A_L, C, A_R, H, LH, RH]]
    #  npev, npC = np_heff.minimize_Hc(mpslist, Hlist, delta, params)
    #  print("npev: ", npev)
    #  print("Err npC: ", np.linalg.norm(ct.apply_Hc(npC, A_L, A_R, Hlist) - npev*npC))
    #  err = np.linalg.norm(np.abs(jaxout - npout))/jaxout.size
    #  if err > thresh or jnp.any(jnp.isnan(jaxout)):
    #      print("FAILED with err: ", err)
    #  else:
    #      print("Passed!")


