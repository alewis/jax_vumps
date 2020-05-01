import numpy as np

import jax.numpy as jnp

import jax_vumps.operations as ops
import jax_vumps.utils as utils
import jax_vumps.arnoldi as arn
import jax_vumps.benchmark as benchmark


def time_arnoldi_matrix(Ns, n_kry, fname="./arnoldi_mat_timings.txt"):
    ntries = 10
    ts = np.zeros((2, len(Ns)))
    funcnames = ["Jax", "NumPy"]

    for Nidx, N in enumerate(Ns):
        print("**************************************************************")
        print("Timing N= ", N)

        npA = np.random.rand(N, N).astype(np.float32)
        np_op = ops.numpy_matrix_linop(npA)
        jaxA = jnp.array(npA)
        jax_args = [jaxA, ]
        v0 = np.random.rand(N).astype(np.float32)
        v0j = jnp.array(v0)

        def jax_arnoldi():
            return arn.arnoldi_krylov(ops.matrix_matvec, jax_args, n_kry, v0j)

        def np_arnoldi():
            return arn.arnoldi_krylov_numpy(np_op, v0, n_kry)

        for idx, f in enumerate([jax_arnoldi, np_arnoldi]):
            print("Function:", funcnames[idx])
            dts = np.zeros(ntries)
            for i in range(ntries):
                t0 = benchmark.tick()
                out = f()
                if funcnames[idx] == "Jax":
                    dts[i] = benchmark.tock(t0, out[0])
                else:
                    dts[i] = benchmark.tock(t0)
            ts[idx, Nidx] = np.min(dts)
            print("t=", ts[idx, Nidx])

    return ts

def time_arnoldi(chis, n_krylov, fname="./arnoldi_timings.txt"):
    d = 2
    chiM = 4
    ntries = 20
    ts = np.zeros((2, len(chis)))
    #np_op = ops.numpy_matrix_linop(
    funcnames = ["Jax", "NumPy"]

    for chidx, chi in enumerate(chis):
        print("**************************************************************")
        print("Timing chi= ", chi)

        npmps = np.random.rand(chi, d, chi).astype(np.float32)
        npmps /= np.linalg.norm(npmps)
        npL = np.random.rand(chiM, chi, chi).astype(np.float32)
        npL /= np.linalg.norm(npL)
        npR = np.random.rand(chiM, chi, chi).astype(np.float32)
        npR /= np.linalg.norm(npR)
        npmpo = np.random.rand(chiM, chiM, d, d).astype(np.float32)
        npmpo /= np.linalg.norm(npmpo)
        np_data = (npmpo, npL, npR)

        mps = jnp.array(npmps)
        L = jnp.array(npL)
        R = jnp.array(npR)
        mpo = jnp.array(npmpo)

        jax_map = jax_dmrg.map.SingleMPOHeffMap(mpo, L, R)
        jax_mv = jax_map.matvec

        jax_data = jax_map.data

        def jax_trid():
            return lz.tridiagonalize(jax_mv, jax_data, jnp.ravel(mps),
                                     n_krylov)


        def np_trid():
            return lz.numpy_tridiagonalize(np_mv, np_data, np.ravel(npmps),
                                           n_krylov)

        outKs = []
        outTs = []
        for idx, f in enumerate([jax_trid, np_trid]):
            print("Function:", funcnames[idx])
            dts = np.zeros(ntries)
            for i in range(ntries):
                t0 = benchmark.tick()
                out = f()
                if funcnames[idx] == "Jax":
                    dts[i] = benchmark.tock(t0, out[0])
                else:
                    dts[i] = benchmark.tock(t0)
            outKs.append(out[0])
            outTs.append(out[1])
            #ts[idx, chidx] = np.amin(dts)
            ts[idx, chidx] = np.median(dts)
            print("t=", ts[idx, chidx])

        errK = jnp.linalg.norm(jnp.abs(outKs[0] - outKs[1]))/outKs[0].size
        jnpdiag = jnp.diag(outTs[0])
        npdiag = jnp.diag(outTs[1])
        errT = jnp.linalg.norm(jnp.abs(outTs[0] - outTs[1]))/outTs[0].size
        #print(jnp.abs(outTs[0]-outTs[1]))
        #  print(outTs[0])
        #  print(outTs[1])
        print("ErrK = ", errK)
        print("ErrT = ", errT)
    return ts


def time_contract(contract_fs, funcnames, chis,
                  fname="./contract_timings.txt"):
    d = 2
    chiM = 4
    ts = np.zeros((len(contract_fs), len(chis)))

    for chidx, chi in enumerate(chis):
        print("**************************************************************")
        print("Timing chi= ", chi)
        mps, L, R, mpo = utils.random_tensors([(chi, d, chi),
                                              (chiM, chi, chi),
                                              (chiM, chi, chi),
                                              (chiM, chiM, d, d)])
        outs = []
        for idx, f in enumerate(contract_fs):
            print("Function:", funcnames[idx])
            dts = np.zeros(20)
            for i in range(20):
                t0 = benchmark.tick()
                out = f(mpo, L, R, mps)
                dts[i] = benchmark.tock(t0, out)
            outs.append(out)
            ts[idx, chidx] = np.amin(dts)
            print("t=", ts[idx, chidx])
        A = outs[0]
        for outidx, out in enumerate(outs[1:]):
            err = jnp.linalg.norm(jnp.abs(A) - jnp.abs(out))/A.size
            print("Err", outidx+1, "= ", err)
    return ts


def time_xx(chis=None, N=100, N_sweeps=1, fname="./xxtimings.txt",
        ncv=10, lz_tol=1E-12, lz_maxiter=4):
    timings = []
    if chis is None:
        chis = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        chis = np.array(chis, dtype=np.int)
    chis = np.array(chis)

    for chi in chis:
        print("**************************************************************")
        print("Timing chi= ", chi)
        _ = xx_ground_state(N, chi, 1, ncv=2, lz_maxiter=1)
        #  for mps in mps_chain:
        #      mps = mps.block_until_ready()
        _, _, _, timing = xx_ground_state(N, chi, N_sweeps, ncv=ncv,
                                          lz_tol=lz_tol, lz_maxiter=lz_maxiter)
        for key in timing:
            print(key, ": ", timing[key])
        print("One sweep: ", (timing["total"]-timing["initialization"])/N_sweeps)
        print("**************************************************************")
        timings.append(timing)
    ts = np.zeros((chis.size, 6))

    t_tot = [timing["total"] for timing in timings]
    ts[:, 0] = t_tot
    t_lz = [timing["lz"] for timing in timings]
    ts[:, 1] = t_lz
    t_qr = [timing["qr"] for timing in timings]
    ts[:, 2] = t_qr
    t_up = [timing["update"] for timing in timings]
    ts[:, 3] = t_up
    t_init = [timing["initialization"] for timing in timings]
    ts[:, 4] = t_init
    t_swp = (np.array(t_tot) - np.array(t_init)) / N_sweeps
    ts[:, 5] = t_swp
    np.savetxt(fname, ts, header="total, lz, qr, update")
    return timings


def xx_ground_state(N, maxchi, N_sweeps, ncv=20, lz_tol=1E-5, lz_maxiter=50):
    """
    Find the ground state of the quantum XX model with single-site DMRG.
    """

    mpo_chain = [ops.xx_mpo() for _ in range(N)]
    lz_params = lz.lz_params(ncv=ncv, lz_tol=lz_tol, lz_maxiter=lz_maxiter)
    Es, mps_chain, H_block, timings = dmrg.dmrg_single(mpo_chain, maxchi,
                                                       N_sweeps,
                                                       lz_params)
    return (Es, mps_chain, H_block, timings)
