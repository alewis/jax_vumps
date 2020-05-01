
from functools import partial
from typing import Sequence, Callable
import numpy as np
import sys

import jax
import jax.numpy as jnp
from jax.ops import index, index_update, index_add

import jax_vumps.arnoldi as arnoldi


def gmres_m(A_mv, A_args, b, x0, n_kry=20, max_restarts=None, tol=1E-6,
            M=None):
    """
    Solve A x = b for x using the restarted GMRES method.

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
    n_kry (int)        : The size of the Krylov subspace computed at each
                         restart. A_mv will be called n_kry times at each
                         restart. The complexity of intermediate operations
                         besides n_kry is cubic with n_kry in this
                         implementation cubic in n_kry.

                         n_kry is set to x0.size in case it is larger than
                         this.
    max_restarts(int,default 10000): The algorithm will terminate after this many
                                 restarts even if unconverged.
    tol                 : Error threshold.
    M : Inverse of the preconditioner of A. Presently unsupported.


    RETURNS
    -------
    x (array, (n,)): The approximate solution.
    beta (float)   : Error estimate.
    n_iter (int)   : The number of iterations that were run.
    converged (bool) : True if convergence was achieved.


    DETAILED EXPLANATION
    --------------------
    GMRES_M is a "Krylov" method. The order-m Krylov subspace of a vector-matrix
    tuple (A, x) is span{x, Ax, AAx, AAAx...}, truncated to m entries; plugging
    choosing x = (A-1 b) and multiplying through by A,
    we see that the solution to (1) lies within the order-n Krylov
    space of (A, b). In practice m < n usually produces good estimates,
    particularly for matrices with degenerate spectra.

    Suppose the columns of V_k form an orthonormal basis of the
    order-n_kry Krylov space of (A, x0). This matrix can be formed by
    the "Arnoldi" method of repeatedly computing v_k = A v_{k-1} and then
    performing Gram-Schmidt orthonormalization upon v_k and its ancestors.
    This method also produces an upper-Hessenberg matrix H satisfying

         A V_k = V_{k+1}  H_k.

    Define the residual vector
         r = b - A z
    Now the solution to (1) may be phrased as that of minimizing the residual,
         x = min_z || r - Az || (2)
    We know that x lies within the Krylov space of (A, b), which we approximate
    by that of (A, x_0) truncated to n_kry. Under the restriction that z
    lies within this space, and writing the solution x_k = x_0 + V_k y,
    minimizing (2) reduces to minimizing
          J(y) = || beta e_1 - H_k y || (3)
    with beta = ||r_0||.
    """
    if M is not None:
        raise NotImplementedError("Preconditioning is unsupported.")

    if max_restarts is None:
        max_restarts = 100
    elif max_restarts < 0:
        raise ValueError("Invalid max_restarts = ", max_restarts)

    if n_kry < 1:
        raise ValueError("Invalid n_kry = ", n_kry)
    elif n_kry > x0.size:
        n_kry = x0.size

    x = x0
    r, beta = gmres_residual(A_mv, A_args, b, x)
    converged = False
    n_iter = 0
    for n in range(max_restarts):
        x = gmres_work(A_mv, A_args, n_kry, x, r, beta)
        r, beta = gmres_residual(A_mv, A_args, b, x)
        beta_rel = beta / jnp.linalg.norm(b)
        n_iter = n + 1
        if beta < tol or beta_rel < tol:
            converged = True
            break
    return x, beta_rel, n_iter, converged


@partial(jax.jit, static_argnums=(3,))
def gmres(A_mv, A_args, b, n_kry, x0):
    """
    Solve A x = b for x by the unrestarted GMRES method.
    """
    r, beta = gmres_residual(A_mv, A_args, b, x0)
    x = gmres_work(A_mv, A_args, n_kry, x0, r, beta)
    return x


def gmres_residual(A_mv, A_args, b, x):
    """
    Computes the residual vector r and norm beta which is minimized by
    GMRES, given A, b,  and a trial solution x.
    """
    r = b - A_mv(*A_args, x)
    beta = jnp.linalg.norm(r)
    return r, beta


@partial(jax.jit, static_argnums=(2,))
def gmres_work(A_mv, A_args, n_kry, x, r, beta):
    """
    The main loop body of GMRES. Given A, a trial solution x, the residual r,
    and the size n_kry of the Krylov space, iterates x towards the solution,
    by finding y in x = x_0 + V y minimizing ||beta - H y||.
    """
    v = r / beta
    Vk_1, Htilde = arnoldi.arnoldi_krylov(A_mv, A_args, n_kry, v)
    Q, Rtilde = jnp.linalg.qr(Htilde, mode="complete")
    Q = Q.T.conj()
    R = Rtilde[:-1, :]
    g = beta*jnp.ravel(Q[:-1, 0])
    y = jax.scipy.linalg.solve_triangular(R, g)
    update = Vk_1[:, :-1] @ y
    x = x + update
    return x

def full_orthog(A_mv, A_args, b, n_kry, x0):
    """
    Solve A x = b for x using the full orthogonalization method.
    """
    r = b - A_mv(*A_args, x0)
    beta = jnp.linalg.norm(r)
    v = r / beta
    e1 = np.zeros(n_kry)
    e1[0] = 1
    e1 = jnp.array(e1)

    Vtilde, Htilde = arnoldi.arnoldi_krylov(A_mv, A_args, n_kry, v)
    H = Htilde[:-1, :]
    V = Vtilde[:, :-1]
    Hinv = jnp.linalg.inv(H)
    y_k = beta * (Hinv @  e1)
    x = x0 + V @ y_k
    return x

