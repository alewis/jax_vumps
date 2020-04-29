"""
Miscellanious utilities for finite DMRG.
"""
import time
import jax
import jax.numpy as jnp


def paulis():
    sX = jnp.array([[0, 1], [1, 0]])
    sY = jnp.array([[0, -1j], [1j, 0]])
    sZ = jnp.array([[1, 0], [0, -1]])
    return (sX, sY, sZ)


def random_tensors(shapes, seed=None, dtype=jnp.float32):
    """
    Returns a list of Gaussian random tensors, one for (and with) each shape
    in shapes. A random seed may optionally be specified; otherwise the
    system time is used.
    """
    if seed is None:
        seed = int(time.time())
    key = jax.random.PRNGKey(seed)
    tensors = []
    for shape in shapes:
        key, subkey = jax.random.split(key)
        tensor = jax.random.normal(key, shape=shape, dtype=dtype)
        tensors.append(tensor)
    return tensors
