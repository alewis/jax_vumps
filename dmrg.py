"""
Jax implementation of single-site DMRG.

Adam GM Lewis
"""

import numpy as np
import jax.numpy as jnp

import jax_dmrg.errors as errors
import jax_dmrg.operations as op
import jax_dmrg.lanczos as lz
import jax_dmrg.benchmark as benchmark


def left_to_right(mps_chain, H_block, mpo_chain, lz_params,
                  initialization=False):
    N = len(mpo_chain)
    Es = np.zeros(N)
    t_lz = 0.
    t_qr = 0.
    t_up = 0.
    keys = ["ncv", "lz_tol", "lz_maxiter"]
    lz_param_list = [lz_params[key] for key in keys]

    for n in range(N-1):
        A = mps_chain[n]
        if not initialization:
            t0_lz = benchmark.tick()
            E, A, err = lz.dmrg_solve(A, H_block[n], H_block[n+1],
                                      mpo_chain[n], *lz_param_list)
            t_lz += benchmark.tock(t0_lz, A)

            Es[n] = E

        t0_qr = benchmark.tick()
        A, C = op.qrpos(A)
        t_qr += benchmark.tock(t0_qr, A)

        t0_up = benchmark.tick()
        mps_chain[n] = A
        H_block[n+1] = op.XopL(H_block[n], mpo_chain[n], A)
        mps_chain[n+1] = op.leftcontract(C, mps_chain[n+1])
        t_up += benchmark.tock(t0_up, mps_chain[n+1])

    n = N-1
    A = mps_chain[n]
    if not initialization:
        t0_lz = benchmark.tick()
        E, A, err = lz.dmrg_solve(A, H_block[n], H_block[n+1],
                                  mpo_chain[n],
                                  *lz_param_list)
        t_lz += benchmark.tock(t0_lz, A)
        Es[n] = E

    t0_qr = benchmark.tick()
    A, C = op.qrpos(A)
    t_qr += benchmark.tock(t0_qr, A)

    mps_chain[n] = A
    return (Es, mps_chain, H_block, t_lz, t_qr, t_up)


def right_to_left(mps_chain, H_block, mpo_chain, lz_params, initialization=False):
    keys = ["ncv", "lz_tol", "lz_maxiter"]
    lz_param_list = [lz_params[key] for key in keys]
    N = len(mpo_chain)
    Es = np.zeros(N)
    t_lz = 0.
    t_qr = 0.
    t_up = 0.
    for n in range(N-1, 0, -1):
        B = mps_chain[n]
        if not initialization:
            t0_lz = benchmark.tick()
            E, B, err = lz.dmrg_solve(B, H_block[n], H_block[n+1],
                                 mpo_chain[n], *lz_param_list)
            t_lz += benchmark.tock(t0_lz, B)
            Es[n] = E

        t0_qr = benchmark.tick()
        C, B = op.lqpos(B)
        t_qr += benchmark.tock(t0_qr, B)
        mps_chain[n] = B

        t0_up = benchmark.tick()
        H_block[n] = op.XopR(H_block[n+1], mpo_chain[n], B)
        mps_chain[n-1] = op.rightcontract(mps_chain[n-1], C)
        t_up += benchmark.tock(t0_up, mps_chain[n-1])

    n = 0
    B = mps_chain[n]

    if not initialization:
        t0_lz = benchmark.tick()
        E, B, err = lz.dmrg_solve(B, H_block[n], H_block[n+1], mpo_chain[n],
                                  *lz_param_list)
        t_lz += benchmark.tock(t0_lz, B)
        Es[n] = E

    t0_qr = benchmark.tick()
    C, B = op.lqpos(B)
    t_qr += benchmark.tock(t0_qr, B)
    mps_chain[n] = B
    return (Es, mps_chain, H_block, t_lz, t_qr, t_up)


def dmrg_single_iteration(mps_chain, H_block, mpo_chain, lz_params):
    #  EsR, mps_chain, H_block, t1_lz, t1_qr, t1_up = right_to_left(mps_chain,
    #                                                               H_block,
    #                                                               mpo_chain,
    #                                                               lz_params)
    EsR, mps_chain, H_block, t1_lz, t1_qr, t1_up = right_to_left(mps_chain,
                                                                 H_block,
                                                                 mpo_chain,
                                                                 lz_params)
    EsL, mps_chain, H_block, t2_lz, t2_qr, t2_up = left_to_right(mps_chain,
                                                                 H_block,
                                                                 mpo_chain,
                                                                 lz_params)
    t_lz = t1_lz + t2_lz
    t_qr = t1_qr + t2_qr
    t_up = t1_up + t2_up
    return (EsR, EsL, mps_chain, H_block, t_lz, t_qr, t_up)


def dmrg_single_initialization(mpo_chain, maxchi: int, N_sweeps: int,
                               lz_params: dict = None,
                               L=None, R=None, mps_chain=None):
    errflag, errstr = errors.check_natural(N_sweeps, "N_sweeps")
    if errflag:
        raise ValueError(errstr)

    errflag, errstr = errors.check_natural(maxchi, "maxchi")
    if errflag:
        raise ValueError(errstr)

    N = len(mpo_chain)
    chiM = mpo_chain[0].shape[0]
    dtype = mpo_chain[0].dtype

    if mps_chain is None:
        mps_chain = op.random_finite_mps(2, N, maxchi,
                                         dtype=dtype)
    chis = [mps.shape[0] for mps in mps_chain] + [mps_chain[-1].shape[-1]]
    print("chis: ", chis)

    if L is None:
        L = op.left_boundary_eye(chiM, dtype=dtype)
    if R is None:
        R = op.right_boundary_eye(chiM, dtype=dtype)
    H_block = [0 for _ in range(N+1)]
    H_block[0] = L
    H_block[-1] = R

    Es = np.zeros((2*N_sweeps, N))
    _, mps_chain, H_block, t_lz, t_qr, t_up = left_to_right(mps_chain, H_block,
                                                            mpo_chain,
                                                            lz_params,
                                                            initialization=True)
    #  _, mps_chain, H_block, t_lz, t_qr, t_up = right_to_left(mps_chain, H_block,
    #                                                          mpo_chain,
    #                                                          lz_params,
    #                                                          initialization=True)
    t_init = t_lz + t_qr + t_up
    return (mps_chain, mpo_chain, H_block, Es, t_init)


def dmrg_single(mpo_chain, maxchi: int, N_sweeps: int,
                lz_params: dict = None,
                L=None, R=None, mps_chain=None):
    """
    Main loop for single-site finite-chain DMRG.
    """
    t0 = benchmark.tick()
    init = dmrg_single_initialization(mpo_chain, maxchi, N_sweeps,
                                      lz_params=lz_params, L=L, R=R,
                                      mps_chain=mps_chain)
    mps_chain, mpo_chain, H_block, Es, t_init = init

    print("Initialization complete. And so it begins!")
    t_lz = 0.
    t_qr = 0.
    t_up = 0.
    N = len(mps_chain)
    for sweep in range(N_sweeps):
        out = dmrg_single_iteration(mps_chain, H_block, mpo_chain, lz_params)
        EsR, EsL, mps_chain, H_block, ti_lz, ti_qr, ti_up = out
        #  print(EsR)
        #  print(EsL)
        
        t_lz += ti_lz
        t_qr += ti_qr
        t_up += ti_up
        E = op.energy(H_block[0], H_block[-1], mpo_chain, mps_chain)
        norm = op.norm(mps_chain)
        #  Es[2*sweep, :] = EsL
        #  Es[2*sweep + 1, :] = EsR
        #  E = EsR[-1]

        # E = 0.5*(jnp.mean(EsL) + jnp.mean(EsR))
        print("Sweep:", sweep, "<E>:", E, "<psi>:", norm)
    tf = benchmark.tock(t0, mps_chain[0])

    timings = {"total": tf,
               "initialization": t_init,
               "lz": t_lz,
               "qr": t_qr,
               "update": t_up}

    return (Es, mps_chain, H_block, timings)
