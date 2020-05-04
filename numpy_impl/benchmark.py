import time

    

def tick():
    return time.perf_counter()


def tock(t0=0., dat=None):
    if dat is not None:
        _ = dat.block_until_ready()
    return time.perf_counter() - t0
