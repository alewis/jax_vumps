import time
import jax.numpy as jnp


def tick():
    return time.perf_counter()


def tock(t0, dat=None):
    if dat is not None:
        try:
            _ = dat.block_until_ready()
        except AttributeError:
            _ = jnp.array(dat).block_until_ready()
    return time.perf_counter() - t0
