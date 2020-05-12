"""
Low level tensor network manipulations.

Conventions
      2                 3 4
      |                 | |
      O                  U
      |                 | |
      1                 1 2

  2---A---3            1
      |                |
      1             2--A--3
"""
import tensornetwork as tn
import jax
import jax.numpy as jnp


@jax.jit
def leftmult(lam, gam):
    """
    2--lam--gam--3
            |
            1
            |
    where lam is stored 1--lam--2
    """
    out = tn.ncon([lam, gam],
                  [[-2, 1],
                  [-1, 1, -3]], backend="jax")
    return out


@jax.jit
def rightmult(gam, lam):
    """
    2--gam--lam--3
       |
       1
       |
    """
    out = tn.ncon([gam, lam],
                  [[-1, -2, 1],
                   [1, -3]], backend="jax")
    return out


@jax.jit
def gauge_transform(gl, A, gr):
    """
            |
            1
     2--gl--A--gr--3
    """
    glA = leftmult(gl, A)
    out = rightmult(glA, gr)
    return out

###############################################################################
# Chain contractors - MPS.
###############################################################################
@jax.jit
def proj(A, B):
    """
    2   2
    |---|
    |   |
    A   B
    |   |
    |---|
    1   1
    Contract A with B to find <A|B>.
    """
    idxs = [[1, 2], [1, 2]]
    contract = [A, B]
    ans = tn.ncon(contract, idxs, backend="jax")
    return ans


# *****************************************************************************
# Single site to open legs.
# *****************************************************************************
def XopL(A, B=None, X=None):
    """
      |---A---2
      |   |
      X   |
      |   |
      |---B---1
    """
    if B is None:
        B = A.conj()
    #  B = jax.lax.cond(B is None,
    #                   A, lambda x: x.conj(),
    #                   B, lambda x: x)
    if X is not None:
        A = leftmult(X, A)
    #  A = jax.lax.cond(X is None,
    #                   A, lambda x: x,
    #                   (X, A), lambda x: leftmult(x[0], x[1]))
    idx = [(2, 1, -2),
           (2, 1, -1)]
    return tn.ncon([A, B], idx, backend="jax")


@jax.jit
def XopL_X(A, X):
    """
    Explicit specialization of XopL to the case B is None, X is not None.
    This allows one to avoid the cond calls in performance-critical code.
    """
    B = A.conj()
    A = leftmult(X, A)
    idx = [(2, 1, -2),
           (2, 1, -1)]
    return tn.ncon([A, B], idx, backend="jax")


@jax.jit
def XopR(A, B=None, X=None):
    """
      2---A---|
          |   |
          |   X
          |   |
      1---B---|
    """
    #  if B is None:
    #      B = A.conj()
    B = jax.lax.cond(B is None,
                     A, lambda x: x.conj(),
                     B, lambda x: x)

    #  if X is not None:
    #      B = rightmult(B, X)
    B = jax.lax.cond(X is None,
                     B, lambda x: x,
                     (B, X), lambda x: rightmult(x[0], x[1]))

    idx = [(2, -2, 1),
           (2, -1, 1)]
    return tn.ncon([A, B], idx, backend="jax")


@jax.jit
def XopR_X(A, X):
    """
    Explicit specialization of XopR to the case B is None, X is not None.
    This allows one to avoid the cond calls in performance-critical code.
    """
    B = rightmult(A.conj(), X)
    idx = [(2, -2, 1),
           (2, -1, 1)]
    return tn.ncon([A, B], idx, backend="jax")



# ***************************************************************************
# TWO SITE OPERATORS
# ***************************************************************************
@jax.jit
def rholoc(A1, A2):
    """
    -----A1-----A2-----
    |    |(3)   |(4)   |
    |                  |
    |                  |
    |    |(1)   |(2)   |
    -----A1-----A2------
    returned as a (1:2)x(3:4) matrix.
    Assuming the appropriate Schmidt vectors have been contracted into the As,
    np.trace(np.dot(op, rholoc.T)) is the expectation value of the two-site
    operator op coupling A1 to A2.
    """
    B1 = A1.conj()
    B2 = A2.conj()
    d = A1.shape[0]
    to_contract = [A1, A2, B1, B2]
    idxs = [(-3, 1, 2),
            (-4, 2, 3),
            (-1, 1, 4),
            (-2, 4, 3)]
    rholoc = tn.ncon(to_contract, idxs, backend="jax").reshape((d**2, d**2))
    return rholoc


@jax.jit
def twositecontract(left, right, U):
    """
       2--left-right--4
            |__|
            |U |
            ----
            |  |
            1  3
    """
    to_contract = (left, right, U)
    idxs = [(2, -2, 1),
            (3, 1, -4),
            (-1, -3, 2, 3)]
    return tn.ncon(to_contract, idxs, backend="jax")


@jax.jit
def twositeexpect(left, right, U):
    d = U.shape[0]
    rho = rholoc(left, right).reshape((d, d, d, d))
    idxs = [(1, 2, 3, 4), (1, 2, 3, 4)]
    expect = tn.ncon([rho, U], idxs, backend="jax").real
    return expect


@jax.jit
def tmdense(A):
    """
    2-A-4
      |
      |
    1-A-3
    """
    idxs = [[1, -2, -4], [1, -1, -3]]
    out = tn.ncon([A, A.conj()], idxs, backend="jax")
    return out


##############################################################################
# VUMPS environment
##############################################################################
@jax.jit
def compute_hL(A_L, htilde):
    """
    --A_L--A_L--
    |  |____|
    |  | h  |
    |  |    |
    |-A_L*-A_L*-
    """
    A_L_d = A_L.conj()
    to_contract = [A_L, A_L, A_L_d, A_L_d, htilde]
    idxs = [(2, 4, 1),
            (3, 1, -2),
            (5, 4, 7),
            (6, 7, -1),
            (5, 6, 2, 3)]
    h_L = tn.ncon(to_contract, idxs, backend="jax")
    return h_L


@jax.jit
def compute_hR(A_R, htilde):
    """
     --A_R--A_R--
        |____|  |
        | h  |  |
        |    |  |
     --A_R*-A_R*-
    """
    A_R_d = A_R.conj()
    to_contract = [A_R, A_R, A_R_d, A_R_d, htilde]
    idxs = [(2, -2, 1),
            (3, 1, 4),
            (5, -1, 7),
            (6, 7, 4),
            (5, 6, 2, 3)]
    h_R = tn.ncon(to_contract, idxs, backend="jax")
    return h_R


##############################################################################
# VUMPS heff
##############################################################################
@jax.jit
def apply_HAc(A_C, A_L, A_R, Hlist):
    """
    Compute A'C via eq 11 of vumps paper (131 of tangent space methods).
    """
    H, LH, RH = Hlist
    to_contract_1 = [A_L, A_L.conj(), A_C, H]
    idxs_1 = [(2, 1, 4),
              (3, 1, -2),
              (5, 4, -3),
              (3, -1, 2, 5)]
    term1 = tn.ncon(to_contract_1, idxs_1, backend="jax")

    to_contract_2 = [A_C, A_R, A_R.conj(), H]
    idxs_2 = [(5, -2, 4),
              (2, 4, 1),
              (3, -3, 1),
              (-1, 3, 5, 2)]
    term2 = tn.ncon(to_contract_2, idxs_2, backend="jax")

    term3 = leftmult(LH, A_C)
    term4 = rightmult(A_C, RH.T)
    A_C_prime = term1 + term2 + term3 + term4
    return A_C_prime


#  @jax.jit
#  def Hc_norm_est(A_L, A_R, Hlist):
#      """
#      Approximate norm of the effective Hamiltonian for Hc. This will be an
#      overestimate, and usually a small one.
#      """
#      return [jnp.amax(A) for A in [A_L, A_R, *Hlist]])
    #  H, LH, RH = Hlist
    #  A_Lstar = A_L.conj()
    #  to_contract = [A_L, A_Lstar, A_R, A_R.conj(), H]
    #  idxs = [(2, 4, 1),
    #          (5, 4, 8),
    #          (3, 1, 6),
    #          (7, 8, 6),
    #          (5, 7, 2, 3)]
    #  term1 = tn.ncon(to_contract, idxs, backend="jax")
    #  mat2 = LH + RH.T
    #  chi = A_L.shape[2]
    #  term2 = jnp.sqrt(chi*jnp.sum(jnp.abs(mat2)**2))
    #  Hc_norm = term1 + term2
    #return Hc_norm


@jax.jit
def apply_Hc(C, A_L, A_R, Hlist):
    """
    Compute C' via eq 16 of vumps paper (132 of tangent space methods).
    """
    H, LH, RH = Hlist
    A_Lstar = A_L.conj()
    A_C = rightmult(A_L, C)
    to_contract = [A_C, A_Lstar, A_R, A_R.conj(), H]
    idxs = [(2, 4, 1),
            (5, 4, -1),
            (3, 1, 6),
            (7, -2, 6),
            (5, 7, 2, 3)]
    term1 = tn.ncon(to_contract, idxs, backend="jax")
    term2 = LH @ C
    term3 = C @ RH.T
    C_prime = term1 + term2 + term3
    return C_prime
