"""
Contains miscellanious utility functions.
"""

import time
import jax
import jax.numpy as jnp


def random_tensors(shapes, seed=None, dtype=jnp.float32):
    """
    Returns a list of Gaussian random tensors, one for (and with) each shape
    in shapes. A random seed may optionally be specified; otherwise the
    system time is used.

    PARAMETERS
    ----------
    shapes: A list of input shapes.
    seed (default time.time()) : The random seed.
    dtype : dtype of tensors.

    RETURNS
    -------
    tensors : A list of random tensors of the given dtype, one respectively
              for each shape.
    """
    if seed is None:
        seed = int(time.time())
    key = jax.random.PRNGKey(seed)
    tensors = []
    for shape in shapes:
        key, subkey = jax.random.split(key)
        tensor = jax.random.normal(key, shape=shape, dtype=dtype)

        if tensor.iscomplexobj():
            tensor_i = jax.random.normal(key, shape=shape, dtype=dtype)
            tensor = tensor + 1.0j*tensor_i
        tensors.append(tensor)
    return tensors


def frobnormscaled(A, B=None):
    """
    The Frobenius norm of the difference between A and B, divided by the
    number of entries in A.
    """
    if B is None:
        B = jnp.zeros(A.shape)
    ans = (1./A.size)*jnp.linalg.norm(jnp.abs(jnp.ravel(A)-jnp.ravel(B)))
    return ans


def sortby(es, vecs, mode="LM"):
    """
    The vector 'es' is sorted,
    and the i's in 'vecs[:, i]' are sorted in the same way. This is done
    by returning new, sorted arrays (not in place). 'Mode' may be 'LM' (sorts
    from largest to smallest magnitude) or 'SR' (sorts from most negative
    to most positive real part).
    """
    if mode == "LM":
        sortidx = jnp.abs(es).argsort()[::-1]
    elif mode == "SR":
        sortidx = (es.real).argsort()
    essorted = es[sortidx]
    vecsorted = vecs[:, sortidx]
    return essorted, vecsorted

"""
ERROR MESSAGES
Functions here take a value 'val' and a name (the name of the variable).
They perform a test on val, and return a tuple (errflag, errstring). 
Errflag is False iff the test passed. If it failed, errflag is True,
and errstring contains an appropriate error message.
"""
def check_natural(val, name: str):
    """
    Passes when val is a natural number (an integer greater than 0).
    """
    flag = False 
    errstr = ""
    if val != round(val) or val <= 0:
        flag = True 
        errstr = name + " = " + val + " must be a natural number."
    return (flag, errstr)



