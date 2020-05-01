import jax.numpy as jnp
import numpy as np
import scipy as sp

import jax_vumps.arnoldi as arnoldi
import jax_vumps.operations as ops
import jax_vumps.utils as utils
import jax_vumps.gmres as gmres

def errstring(arr1, name1, arr2, name2):
    """
    Utility function to generate failure messages in a few tests. Returns
    the Frobenius norm of the difference between arr1 and arr2, and a
    string errstring to be output if desired.
    """
    error = matutils.frob(arr1, arr2)
    errormsg = "Error = " + str(error) + "\n"
    errormsg += name1 + ":\n " + str(arr1) + "\n"
    errormsg += name2 + ":\n " + str(arr2)
    return (error, errormsg)


def errorstack(errtups, passed=True, msg="\n", thresh=1E-6):
    """
    Combines the output from multiples calls to errstring into a single
    pass-or-fail condition, based on comparison of each errtups[i][0] to
    thresh. Returns this flag along with a single error message, concatenated
    from
    those of each individual call.
    """
    for error, errormsg in errtups:
        if error < thresh:
            passed = False
            msg += "**********************************************************"
            msg += errormsg
            msg += "**********************************************************"
    return (passed, msg)


def manyclose(pairs):
    """
    Loops over pairs of numpy arrays and runs allclose on each. Returns True
    if allclose passes for each pair and False otherwise.
    """
    passed = True
    for pair in pairs:
        passed = passed and jnp.allclose(pair[0], pair[1])
    return passed

###############################################################################
# OPS
###############################################################################


def test_matrix_matvec(m, n, thresh=1E-5, verbose=False):
    A, v = utils.random_tensors([(m, n), (n,)])
    mv = ops.matrix_matvec
    Av_dense = A@v
    Av_sparse = mv(A, v)
    error = jnp.linalg.norm(jnp.abs(Av_dense - Av_sparse))
    passed = True
    if error > thresh:
        passed = False
        if verbose:
            print("Test failed by : ", error)
    return (passed, error)


def do_test_matrix_matvec(thresh=1E-5):
    print("Testing ops.matrix_matvec...")
    allpassed = True
    for m in [5, 10, 15, 20]:
        for n in [5, 10, 15, 20]:
            passed, error = test_matrix_matvec(m, n, thresh=thresh,
                                               verbose=False)
            if error > thresh:
                allpassed = False
                print("Test failed at m=", m, "n = ", n, "by : ", error)
    print("Done!")
    return allpassed



###############################################################################
# ARNOLDI
###############################################################################

def test_arnoldi_orthonormality(A_mv, A_args, n_kry, v0, thresh=1E-5,
                                verbose=False):
    """
    Confirms that V from V, H = arnoldi_krylov has orthonormal columns.
    """

    V, H = arnoldi.arnoldi_krylov(A_mv, A_args, n_kry, v0)
    errtot = 0.
    passed = True
    for j in range(0, min(V.shape[1], v0.size)):
        if not np.allclose(V[:, j], np.zeros(V[:, j].size)):
            for k in range(j+1):
                err = jnp.abs(jnp.vdot(V[:, k], V[:, j]))
                errtot += err
                if not (err < thresh or (1 - err) < thresh):
                    passed = False
                    verbose = True
                    if verbose:
                        print("Index pair ", k, j, " failed by", err)
    return (passed, errtot)


def test_arnoldi_fixed_point(A_mv, A_args, n_kry, v0, thresh=1E-5,
                             verbose=False):
    """
    Confirms that V, H = arnoldi_krylov satisy A V_k = V_(k+1) H, with
    A specified as a matvec.
    """
    V, H = arnoldi.arnoldi_krylov(A_mv, A_args, n_kry, v0)
    passed = True
    err = 0.
    VH = V@H
    AV = ops.matmat(A_mv, A_args, V[:, :-1])
    err = jnp.linalg.norm(jnp.abs(AV - VH))/AV.size
    if err > thresh:
        passed = False
        if verbose:
            print("Arnoldi fixed point failed by ", err)
    return (passed, err)


def test_arnoldi_vs_numpy(A_mv, A_args, A_np_op, n_kry, thresh=1E-5,
                          verbose=True):
    N = A_np_op.shape[1]
    v0 = np.random.rand(N)
    v0j = jnp.array(v0)
    V_np, H_np = arnoldi.arnoldi_krylov_numpy(A_np_op, v0, n_kry)
    #V_jax, H_jax = arnoldi.arnoldi_krylov(A_mv, A_args, n_kry, v0j)
    V_jax, H_jax = arnoldi.arnoldi_krylov_jax(A_mv, A_args, n_kry, v0j)
    err_v = jnp.linalg.norm(jnp.abs((V_np - V_jax)))/(V_np.size)
    err_H = jnp.linalg.norm(jnp.abs((H_np - H_jax)))/(H_np.size)
    pass_v = True
    pass_H = True
    if err_v > thresh:
        pass_v = False
        if verbose:
            print("V differed by ", err_v)
            print("numpy :", V_np)
            print("Jax :", V_jax)
    if err_H > thresh:
        pass_H = False
        if verbose:
            print("H differed by ", err_H)
            print("numpy :", H_np)
            print("Jax :", H_jax)
    return (pass_v and pass_H, err_v + err_H)


def do_arnoldi_vs_numpy_dense(thresh=1E-5):
    """
    Runs the Jax and NumPy implementations of Arnoldi on random dense
    matrices and confirms the results are equal.
    """
    print("Testing Jax Arnoldi against numpy Arnoldi on random dense matrices.")
    Ns = np.arange(5, 25, 5)
    allpass = True
    for N in Ns:
        n_krys = np.arange(1, N-1, 3)
        for n_kry in n_krys:
            A = np.random.rand(N, N)
            Aj = jnp.array(A)
            jax_mv = ops.matrix_matvec
            jax_args = [Aj, ]
            np_op = ops.numpy_matrix_linop(A)
            mepass, err = test_arnoldi_vs_numpy(jax_mv, jax_args, np_op, n_kry,
                                                thresh=1E-5, verbose=True)
            if not mepass:
                print("Arnoldi vs numpy failed at N=", N, "n_kry=", n_kry,
                      "by err=", err)
            allpass = allpass and mepass
    print("Done!")
    return allpass


def do_test_gs_orthogonalize(thresh=1E-5):
    print("Testing orthogonalization.")
    V = np.array([[1., 0.],
                 [0., 1.],
                 [0., 0.]])
    Vj = jnp.array(V)
    r = np.array([1., 1., 1.])
    rj = jnp.array(r)

    hnp = np.zeros(2)
    for j in range(2):  # Subtract the projections on previous vectors
        vj = V[:, j]
        hjk = np.vdot(vj, r)
        r = r - hjk * vj
        hnp[j] = hjk

    rjax, hjax = arnoldi.gs_orthogonalize(Vj, rj)
    rpass = np.allclose(rjax, r)
    hpass = np.allclose(hjax, hnp)
    allpass = rpass and hpass
    if not allpass:
        print("Failed!")
    else:
        print("Passed!")
    #  print("rJax, r:")
    #  print(rjax, r)
    #  print("hJax, hnp:")
    #  print(hjax, hnp)
    return allpass



def do_arnoldi_tests_random_dense_matrices(thresh=1E-5):
    """
    Runs tests for orthonormality of V and the form of H with A set to various
    random dense matrices.
    """
    print("Testing Arnoldi on dense matrices.")
    mv = ops.matrix_matvec
    allpass = True
    Ns = np.arange(5, 25, 5)
    for N in Ns:
        n_krys = np.arange(1, N-1, 3)
        for n_kry in n_krys:
            shapes = [(N, N), (N,)]
            A, v0 = utils.random_tensors(shapes)
            orth_pass, orth_err = test_arnoldi_orthonormality(mv, [A], n_kry,
                                                              v0,
                                                              thresh=thresh,
                                                              verbose=True)
            if not orth_pass:
                print("Orthonormality failed at N, n_kry = ", N, n_kry,
                      "by ", orth_err)

            fp_pass, fp_err = test_arnoldi_fixed_point(mv, [A], n_kry, v0,
                                                       thresh=thresh)
            if not fp_pass:
                print("Fixed point failed at N=", N, "n_kry=", n_kry,
                      "by ", fp_err)
            allpass = allpass and orth_pass and fp_pass
    return allpass


def do_arnoldi_tests_identity(N=3, thresh=1E-5):
    """
    Runs tests for orthonormality of V and the form of H with A the 
    identity matrix.
    """
    print("Testing Arnoldi on the identity with N=", N)
    n_kry = N
    A = jnp.eye(N)
    shapes = [(N,)]
    v0, = utils.random_tensors(shapes)
    mv = ops.matrix_matvec
    orth_pass, orth_err = test_arnoldi_orthonormality(mv, [A], n_kry,
                                                      v0,
                                                      thresh=thresh)
    allpass = True
    if not orth_pass:
        allpass = False
        print("Orthonormality failed at N, n_kry", N, n_kry,
              "by ", orth_err)

    fp_pass, fp_err = test_arnoldi_fixed_point(mv, [A], n_kry, v0,
                                               thresh=thresh)
    if not fp_pass:
        allpass = False
        print("Fixed point failed at N, n_kry", N, n_kry,
              "by ", fp_err)
    print("Done!")
    return allpass


def do_arnoldi_test_fixed(N=2, n_kry=2, thresh=1E-5):
    """
    Runs tests for orthonormality of V and the form of H with A a specific
    matrix (the identity plus an upper off diagonal of 1s), and v0 all
    ones.
    """
    print("Testing Arnoldi with a fixed matrix...")
    A = np.eye(N) + np.diag(np.ones(N-1, dtype=np.float32), 1)

    A = jnp.array(A)
    v0 = np.ones(N)
    v0 = jnp.array(v0)
    mv = ops.matrix_matvec
    allpassed = True
    orth_pass, orth_err = test_arnoldi_orthonormality(mv, [A], n_kry,
                                                      v0,
                                                      thresh=thresh)
    if not orth_pass:
        allpassed = False
        print("Orthonormality failed at N, n_kry", N, n_kry,
              "by ", orth_err)

    fp_pass, fp_err = test_arnoldi_fixed_point(mv, [A], n_kry, v0,
                                               thresh=thresh)
    if not fp_pass:
        allpassed = False
        print("Fixed point failed at N, n_kry", N, n_kry,
              "by ", fp_err)
    print("Done!")
    return allpassed
#def arnoldi_tests_vumps_solve_L():

#def arnoldi_tests_vumps_solve_R():


###############################################################################
# GMRES
###############################################################################
def test_gmres_vs_np(A, b, x0, tol=1E-5, n_kry=20, maxiter=4, thresh=1E-5,
                     verbose=False):
    """
    Tests Jax GMRES against SciPy for particular input.
    """
    np_op = ops.numpy_matrix_linop(A)
    np_x, _ = sp.sparse.linalg.gmres(np_op, b, x0=x0, tol=tol,
                                     restart=n_kry,
                                     maxiter=maxiter)
    jax_mv = ops.matrix_matvec
    jax_x, err, n_iter, converged = gmres.gmres_m(jax_mv, [jnp.array(A), ],
                                                  jnp.array(b), jnp.array(x0), 
                                                  n_kry=n_kry, 
                                                  max_restarts=maxiter,
                                                  tol=tol)
    err = jnp.linalg.norm(jnp.abs(np_x - jax_x))/np_x.size
    passed = True
    if err > thresh:
        passed = False
        if verbose:
            print("Jax and SciPy gmres differed by ", err)
            print("Jax :", jax_x)
            print("SciPy :", np_x)
    return (passed, err)


def do_test_gmres_simple(tol=1E-5, verbose=False):
    """
    Tests the Jax implementation of GMRES against the analytically
    determined solution
    of a simple 2x2 system.
    """
    verbose = True
    print("Testing gmres on a fixed simple system.")
    A = jnp.array(np.array([[1, 1],
                           [3, -4]]))
    b = jnp.array(np.array([3, 2]))
    v0 = jnp.ones(2)
    n_kry = 2

    x = gmres.gmres(ops.matrix_matvec, [A, ], b, n_kry, v0)
    solution = np.array([2., 1.])
    passed = np.allclose(x, solution)
    if passed:
        print("Passed!")
    else:
        print("Failed!")
        if verbose:
            x2 = gmres.full_orthog(ops.matrix_matvec, [A, ], b, n_kry, v0)
            print("Correct x: ", solution)
            print("Jax GMRES x: ", x)
            print("Full Orthog x :", x2)
    return passed


def do_test_gmres_vs_np(thresh=1E-5):
    """
    Tests the Jax implementation of GMRES against the SciPy one, on
    various random matrices and values of n_kry.
    """
    print("Testing gmres against SciPy on random input.")
    allpassed = True
    for N in np.arange(20, 100, 20):
        n_krys = [10, 20]
        A = np.random.rand(N, N)
        x0 = np.random.rand(N)
        b = np.random.rand(N)
        for n_kry in n_krys:
            passed, err = test_gmres_vs_np(A, b, x0, n_kry=n_kry,
                                           thresh=thresh, verbose=True,
                                           maxiter=4)
            if not passed:
                print("Failed at N=", N, "n_kry=", n_kry, "by err=", err)
                allpassed = False
    return allpassed


def test_gmres(A, b, x0, tol=1E-5, n_kry=20, maxiter=None, thresh=1E-7,
               verbose=False):
    jax_mv = ops.matrix_matvec(A)
    jax_x, _ = gmres.gmres_m(jax_mv, [A], b, x0, tol, n_kry, maxiter)
    Ax = jax_mv(A, jax_x)
    err = jnp.linalg.norm(jnp.abs(Ax - b)) 
    passed = True
    if err > thresh:
        passed = False
        print("Jax err differed from solution by ", err)
        if verbose:
            print("x : ", jax_x)
            print("A : ", A)
            print("b : ", b)
            print("Ax : ", Ax)
            print("Ax should equal b.")
    return (passed, err)


def run_tests():
    num_pass = 0
    num_run = 0
    #  num_pass += do_test_matrix_matvec(thresh=1E-5)
    #  num_run += 1
    #  num_pass += do_arnoldi_tests_identity(N=3, thresh=1E-5)
    #  num_run += 1
    #  num_pass += do_arnoldi_test_fixed(N=3, n_kry=3)
    #  num_run += 1
    #  num_pass += do_arnoldi_vs_numpy_dense(thresh=1E-5)
    #  num_run += 1
    #  num_pass += do_test_gs_orthogonalize()
    #  num_run += 1
    #  num_pass += do_arnoldi_tests_random_dense_matrices(thresh=1E-5)
    #  num_run += 1
    num_pass += do_test_gmres_simple()
    num_run += 1
    num_pass += do_test_gmres_vs_np()
    num_run += 1
    print("****************************************************************")
    print(num_pass, "tests passed out of", num_run,".")
    print("****************************************************************")
