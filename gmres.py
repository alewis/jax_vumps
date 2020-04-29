
from functools import partial
from typing import Sequence, Callable
import numpy as np

import jax
import jax.numpy as jnp
from jax.ops import index, index_update, index_add

import jax_vumps.arnoldi as arnoldi


def gmres_m(A_mv, A_args, b, x0, m_krylov, maxiter=None, tol=1E-6,
            rtol=1E-4):
    """
    Given a linear mapping with (n x n) matrix representation
        A = A_mv(*A_args)
    gmres_m solves
        Ax = b          (1)
    where x and b are length-b vectors, using the method of
    Generalized Minimum RESiduals with M iterations per restart (GMRES_M).

    PARAMETERS
    ----------
    A_mv (Callable)       : Function (*A_args v) that computes A*v.
    b  (array-like, (n,)) : The vector b in Ax = b.
    x0 (array-like, (n,)) : Initial guess vector.
    m_krylov (int)        : The size of the Krylov subspace computed at each
                            restart. This controls the expense as follows: each
                            restart will involve m_krylov calls
                            to matvec, followed by Gram-Schmidt
                            orthogonalization of a matrix with one column added
                            per call.
    maxiter(int,default n): The algorithm will terminate after this many
                            restarts even if unconverged.
    atol, rtol (float)    : Absolute and relative error thresholds that
                            dictate termination.


    RETURNS
    -------
    (x, err)
    x (array-like, (n,))  : The approximate solution.
    err (float)           : Latest estimate of the approximation quality.



    DETAILED EXPLANATION
    --------------------
    GMRES_M is a "Krylov" method. The order-m Krylov subspace of a vector-matrix
    tuple (A, x) is span{x, Ax, AAx, AAAx...}, truncated to m entries; plugging
    choosing x = (A-1 b) and multiplying through by A,
    we see that the solution to (1) lies within the order-n Krylov
    space of (A, b). In practice m < n usually produces good estimates,
    particularly for matrices with degenerate spectra.

    Suppose the columns of V_k form an orthonormal basis of the
    order-m_krylov Krylov space of (A, x0). This matrix can be formed by
    the "Arnoldi" method of repeatedly computing v_k = A v_{k-1} and then
    performing Gram-Schmidt orthonormalization upon v_k and its ancestors.
    This method also produces an upper-Hessenberg matrix H satisfying

         A V_k = V_{k+1}  H_k.

    Define the residual vector
         r = b - A z
    Now the solution to (1) may be phrased as that of minimizing the residual,
         x = min_z || r - Az || (2)
    We know that x lies within the Krylov space of (A, b), which we approximate
    by that of (A, x_0) truncated to m_krylov. Under the restriction that z
    lies within this space, and writing the solution x_k = x_0 + V_k y,
    minimizing (2) reduces to minimizing
          J(y) = || beta e_1 - H_k y || (3)
    with beta = ||r_0||.
    """
    if maxiter is None or maxiter > b.size:
        maxiter = b.size
    r = b - A_mv(*A_args, x0)
    print("r = ", r)
    beta = jnp.linalg.norm(r)
    for _ in range(maxiter - 1):
        print("beta: ", beta)
        x = _gmres_iter(A_mv, A_args, m_krylov, r, beta)
        print("x = ", x)
        r = b - A_mv(*A_args, x)
        beta = jnp.linalg.norm(r)
        print("****************")
    print("beta: ", beta)

    x = _gmres_iter(A_mv, A_args, m_krylov, r, beta)
    return x


def _gmres_iter(A_mv, A_args, m_krylov, r, beta):
    r = r / beta
    K, H = arnoldi.arnoldi_krylov(A_mv, A_args, m_krylov, r)
    Q, R = jnp.linalg.qr(H, mode="complete")
    print("Q: ", Q)
    print("R: ", R)
    g = beta*jnp.ravel(Q[:-1, 0])
    print("g = ", g)
    # print(R[:-1, :-1])
    y = jax.scipy.linalg.solve_triangular(R[:-1, :], g)
    # print("y = ", y)
    # print("K = ", K)
    x = K[:, :y.size] @ y + r
    return x


def matrix_matvec(A, x):
    return A@x

