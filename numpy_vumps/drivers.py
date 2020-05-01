import numpy as np
from bhtools.tebd.constants import *
import bhtools.tebd.utils as utils
import bhtools.tebd.finitemps as finitemps
#import bhtools.tebd.finitempo as finitempo
import bhtools.tebd.tebdevolver as tebdevolver
import bhtools.tebd.idmrgevolver as idmrgevolver
import bhtools.tebd.observers as observers
import bhtools.tebd.vumps as vumps
import copy
#import bhtools.tests.exactwf as exactwf
#import bhtools.tebd.dmrg as dmrg
import os
import sys

def loaddirac(path):
    #Es = np.loadtxt(path+"/E.txt")
    phibarphis = np.loadtxt(path+"/phibarphi.txt", dtype=np.complex128)
    phi1barphis = np.loadtxt(path+"/phi1barphi.txt", dtype=np.complex128)
    norm = np.loadtxt(path+"/norm.txt", dtype=np.complex128) 
    ts = phibarphis[:, 0]
    phibarphi_ev = phibarphis[:, 1] #/ norm[:, 1]
    phi1barphi_ev = phi1barphis[:, 1] #/ norm[:, 1]
    return ts, norm, phibarphi_ev, phi1barphi_ev

def preparedirac(Hchain, N, l, targetsite, m, sep):
    opstringcondl = utils.correlator_string(sep, c=1/l)
    opstringcurrentl = utils.correlator_string(sep+1, c=1/l)

    phibarphi = lambda mps: observers.opstringev(mps, opstringcondl, targetsite)
    phi1barphi = lambda mps: observers.opstringev(mps, opstringcurrentl, 
            targetsite)

    #E = lambda mps: observers.average(mps, Hchain)

    observables = [#("E", E, True),
                   ("phibarphi", phibarphi, True),
                   ("phi1barphi", phi1barphi, True),
                   ("norm", observers.norm, True)
                  ]

    opstringcond = utils.correlator_string(sep)
    opstringcurrent = utils.correlator_string(sep+1)
    phibarphicv = lambda mps: observers.opstringev(mps, opstringcond, targetsite)
    phi1barphicv = lambda mps: observers.opstringev(mps, opstringcurrent, 
            targetsite)
    to_converge = [phibarphicv, phi1barphicv]
    return observables, to_converge

def diractebd(N, l, targetsite=0, sep=2, m=1, maxchi=128, order="fourth", 
        nst_obs = 10, x0=0, tf=2.0, dt=0.1):
    Hchain = [utils.H_dirac(n, l, m) for n in range(N-1)]
    observables = preparedirac(Hchain, N, l, targetsite, m, sep)
    path = os.path.dirname(os.path.realpath(__file__))+"/testout/finitediracmpsconverge"
    
    mps = finitemps.randommps(2, N, maxchi)
    evolver = tebdevolver.TEBD_Evolver(mps, Hchain, silent=False, outdir=path, 
                           timestamp=False, to_observe=observables, 
                           scriptcopy=False)
    t0 = 0.
    err = evolver.evolve(t0, tf, dt, tf/nst_obs)
    print("err: ", err)
    ts, Es, norm, phibarphi_ev, phi1barphi_ev = loaddirac(path)
    return mps, ts, phibarphi_ev, phi1barphi_ev

def thirringpath(base, g, m, l, chi, trimto=7):
    niceg = str(g).replace(".", "_")
    nicem = str(m).replace(".", "_")
    nicel = str(l).replace(".", "_")
    if trimto is not None: 
        niceg = niceg[:trimto]
        nicem = nicem[:trimto]
        nicel = nicel[:trimto]
        
    if nicel == "0_0037878787878787876":
        nicel = "0_003787878787878788"
    if nicel == "0_003333333333333333":
        nicel = "0_0033333333333333335"
    if nicel == "0_0027777777777777775":
        nicel = "0_002777777777777778"
    if nicel == "0_002525252525252525":
        nicel = "0_0025252525252525255"

    out = base + "/_g_" + niceg + "_m_" + nicem + "_l_" + nicel + "_chi_" + str(chi)
    return out


def mps_thirring_sweep(rmax, Nr, ms, gs, lfacts, maxchis,
        basepath="/testout/thirringsweep",
        minlam=1E-13,
        convergence_params = tebdevolver.convergence_params(),
        hotstart=None):

    for g in gs:
        for m in ms:
            #print(convergence_params.dt_0)
            mps_converging_currents(rmax, Nr, m, g=g, lfacts=lfacts,
                    maxchis=maxchis, path=basepath, minlam=minlam, 
                    convergence_params = convergence_params,
                    hotstart = None)
            #hotstart = mps
    return None    



def mps_converging_currents(rmax, Nr, m, g=None, lfacts=[2,4,8], 
        maxchis=[64],
        path="/testout/convergingcurrents",
        minlam=1E-13, convergence_params=tebdevolver.convergence_params(),
        hotstart = None):
    if lfacts and lfacts[0]!=1:
        print("Warning: lfacts[0]!=1, but this function does not prepend a 1")
    lfacts = lfacts
    ls = []
    
    for lidx, lf in enumerate(lfacts):

        rs, l = np.linspace(rmax, 0, num=lf*Nr, retstep=True,
                endpoint=False)#[1:]
        l = l * (-1)
        rs = rs[::-1]
        ls.append(l)
        maxsep = rs.size
        for maxchi in maxchis:
            thepath = thirringpath(path, g, m, l, maxchi)
            mps_infcorrelatorsim(m, l, maxsep, maxchi, g=g, path=thepath,
                    minlam=minlam, convergence_params=convergence_params,
                    hotstart = None)
            #hotstart = mps
            #Return the first mps, so a loop around this can hotstart from
            #the beginning of the internal loops (i.e. if an external routine
            #changes the mass, it can this way start from the original lf
            #and chi).
            #To do: always hotstart from the closest previous sim.
            # if mps1 is None:
                # mps1 = copy.deepcopy(hotstart)
        # if returnmps:
            # allmps.append(mps)


def vumps_thirring_sweep(rmax, Nr, ms, gs, lfacts, maxchis,
        basepath="/testout/vumpsthirringsweep",
        tol=1E-8, maxiter=100):

    for g in gs:
        for m in ms:
            #print(convergence_params.dt_0)
            vumps_converging_currents(rmax, Nr, m, g=g, lfacts=lfacts,
                    maxchis=maxchis, path=basepath, tol=tol,
                    maxiter=maxiter) 
            #hotstart = mps
    return None    

def vumps_converging_currents(rmax, Nr, m, g=None, lfacts=[2,4,8], 
        maxchis=[32, 64],
        path="/testout/convergingcurrents", tol=1E-8,
        maxiter=300):
    if lfacts and lfacts[0]!=1:
        print("Warning: lfacts[0]!=1, but this function does not prepend a 1")
    lfacts = lfacts
    ls = []
    
    for lidx, lf in enumerate(lfacts):

        rs, l = np.linspace(rmax, 0, num=lf*Nr, retstep=True,
                endpoint=False)#[1:]
        l = l * (-1)
        rs = rs[::-1]
        ls.append(l)
        for maxchi in maxchis:
            thepath = thirringpath(path, g, m, l, maxchi)
            vumps_thirring(maxchi, m, l, g=g, path=thepath, tol=tol,
                    maxiter=maxiter)
        
"""
Functions to directly run VUMPS.
"""

def runvumps(H, path, chi, tol, maxiter):
    here = sys.path[0]
    fullpath = here + path
    vumps_params = vumps.vumps_params(path=fullpath,
            chi=chi,
            vumps_tol=tol,
            maxiter=maxiter
            )
    out = vumps.vumps(H, params=vumps_params)
    return out


def vumps_XXZ(chi, delta, nu=2, l=1, tol=1E-4, path="/testout/vumpsXXZ",
            maxiter=100):
    H = utils.H_XXZ(delta=delta, nu=nu, l=l)
    out = runvumps(H, path, chi, tol, maxiter)
    return out


def vumps_thirring(chi, m, l, g, tol=1E-12, 
        path="/testout/vumpsthirring",
        maxiter=300):
    H0 = utils.H_Thirring(0, l, m, g=g)
    H1 = utils.H_Thirring(1, l, m, g=g)
    H = utils.H_join(H0, H1).real
    out = runvumps(H, path, chi, tol, maxiter)
    return out
    

def vumps_thirring_mod(chi, l, m=None, tol=1E-12, 
        g=None, path="/testout/vumpsthirring",
        maxiter=300):
    if m is None or m==0 or m==0.:
        return vumps_thirring_massless(chi, l, tol=tol, g=g,
                path=path, maxiter=maxiter)
    H0 = utils.H_Thirring_Mod(l, n=0, m=m, g=g)
    H1 = utils.H_Thirring_Mod(l, n=1, m=m, g=g)
    H = utils.H_join(H0, H1)
    out = runvumps(H, path, chi, tol, maxiter)
    return out


def vumps_thirring_massless(chi, l, tol=1E-12, g=None, 
        path="/testout/vumpsthirring", maxiter=300):
    H = utils.H_Thirring_Mod(l, n=0, g=g)
    out = runvumps(H, path, chi, tol, maxiter)
    return out


def vumps_ising(J, h, chi, tol=1E-12, path="/testout/vumpsIsingtest",
        maxiter=200):
    H = utils.H_ising(J, h)
    out = runvumps(H, path, chi, tol, maxiter)
    return out



"""
TEBD
"""
def thirringmodtebd(m, l, maxchi, g=None, 
        path="/testout/diracinfconverge", minlam=1E-13,
        tf=4, dt=0.01):

    H0 = utils.H_Thirring_Mod(l, n=0, m=m, g=g) 
    H1 = utils.H_Thirring_Mod(l, n=1, m=m, g=g) 
    Hchain = [H0, H1]
    here = sys.path[0]
    #here = os.path.dirname(os.path.realpath(__file__))
    fullpath = here + path

    print("**********************************************************")
    print("Beginning simulation with l= ", l , "m = ", m, " chi = ", maxchi, "g = ", g)
    print("**********************************************************")
    E = lambda mymps: observers.allsites(mymps, Hchain)
    observables = [
            ("E", E, True)
            ] 
    mps = finitemps.randominfinitemps(2, 2, 8, maxchi=maxchi, minlam=minlam)
    evolver = tebdevolver.TEBD_Evolver(mps, Hchain,  outdir=fullpath, 
                                       timestamp=False, to_observe=observables, 
                                       scriptcopy=False)
    evolver.evolve(0, tf, dt, tf/10)

    print("Complete!")
    return mps


def mps_thirringbig(m, l, maxsep, maxchi, g=None, 
        path="/testout/diracinfconverge", minlam=1E-13,
        convergence_params=tebdevolver.convergence_params(),
        normtarget=1.1, 
        hotstart=None):

    H0 = utils.H_Thirring(0, l, m, g=g) 
    H1 = utils.H_Thirring(1, l, m, g=g) 
    Hchain = [utils.H_join(H0, H1)]
    here = sys.path[0]
    #here = os.path.dirname(os.path.realpath(__file__))
    fullpath = here + path
    seps = [i+1 for i in range(maxsep)]
    #print(seps)
    correlator_func = lambda mps: observers.JWcorrs_at_seps(mps,
            0, seps, c=1/l)
    to_converge_func = lambda mps: observers.JWcorrs_at_seps(mps,
            0, seps, c=1)
    E = lambda mymps: observers.allsites(mymps, Hchain)
    observables = [("phi^d_phi", correlator_func, False),
            ("norm", observers.norm, False),
            ("E", E, True)
            ] 

    my_convergence_params = copy.deepcopy(convergence_params)
    thisdt_0 = my_convergence_params.dt_0
    thisdt_min = my_convergence_params.dt_min
    Uexpmax = np.amax(np.abs(utils.Uexp(thisdt_0*H0, verbose=False)))
    while Uexpmax >= normtarget: 
        thisdt_0 /= 1.1
        thisdt_min /= 1.1
        Uexpmax = np.amax(np.abs(utils.Uexp(thisdt_0*H0, verbose=False)))
    my_convergence_params.dt_0 = thisdt_0 
    my_convergence_params.dt_min = thisdt_min 

    # if normtarget:
        # dt0 = convergence_params.dt0
        # U = utils.Uexp(
        # U = np.exp(

    to_converge = [to_converge_func]

    print("**********************************************************")
    print("Beginning simulation with l= ", l , "m = ", m, " chi = ", maxchi, "g = ", g)
    print("**********************************************************")
    print("Using refined timestep: ", thisdt_0)
    if hotstart is None:
        mps = finitemps.randominfinitemps(2, 2, 8, maxchi=maxchi, minlam=minlam)
    else:
        mps = hotstart
        mps.maxchi = maxchi
    evolver = tebdevolver.TEBD_Evolver(mps, Hchain,  outdir=fullpath, 
                                       timestamp=False, to_observe=observables, 
                                       scriptcopy=False)
    evolver.converge(to_converge, params=my_convergence_params) 

    print("Complete!")
    return mps



def mps_infcorrelatorsim(m, l, maxsep, maxchi, g=None, 
        path="/testout/diracinfconverge", 
        thresh=1E-4,
        minlam=1E-13,
        convergence_params=tebdevolver.convergence_params(),
        normtarget=1.1, 
        hotstart=None):

    H0 = utils.H_Thirring(0, l, m, g=g) 
    H1 = utils.H_Thirring(1, l, m, g=g) 
    Hchain = [H0, H1]
    here = sys.path[0]
    #here = os.path.dirname(os.path.realpath(__file__))
    fullpath = here + path
    seps = [i+1 for i in range(maxsep)]
    #print(seps)
    correlator_func = lambda mps: observers.JWcorrs_at_seps(mps,
            0, seps, c=1/l)
    to_converge_func = lambda mps: observers.JWcorrs_at_seps(mps,
            0, seps, c=1)
    E = lambda mymps: observers.allsites(mymps, Hchain)
    observables = [("phi^d_phi", correlator_func, False),
            ("norm", observers.norm, False),
            ("E", E, True)
            ] 

    my_convergence_params = copy.deepcopy(convergence_params)
    thisdt_0 = my_convergence_params.dt_0
    thisdt_min = my_convergence_params.dt_min
    Uexpmax = np.amax(np.abs(utils.Uexp(thisdt_0*H0, verbose=False)))
    while Uexpmax >= normtarget: 
        thisdt_0 /= 1.1
        thisdt_min /= 1.1
        Uexpmax = np.amax(np.abs(utils.Uexp(thisdt_0*H0, verbose=False)))
    my_convergence_params.dt_0 = thisdt_0 
    my_convergence_params.dt_min = thisdt_min 


    # if normtarget:
        # dt0 = convergence_params.dt0
        # U = utils.Uexp(
        # U = np.exp(

    to_converge = [to_converge_func]

    print("**********************************************************")
    print("Beginning simulation with l= ", l , "m = ", m, " chi = ", maxchi, "g = ", g)
    print("**********************************************************")
    print("Using refined timestep: ", thisdt_0)
    if hotstart is None:
        mps = finitemps.randominfinitemps(2, 2, 8, maxchi=maxchi, minlam=minlam)
    else:
        mps = hotstart
        mps.maxchi = maxchi
    evolver = tebdevolver.TEBD_Evolver(mps, Hchain,  outdir=fullpath, 
                                       timestamp=False, to_observe=observables, 
                                       scriptcopy=False)
    evolver.converge(to_converge, params=my_convergence_params) 

    print("Complete!")
    return mps








def rundmrg(m, l, maxsep, maxchi, nsteps, g=None, 
        path="/testout/thirring_dmrg", 
        minlam=1E-13):
        
    H0 = utils.H_Thirring(0, l, m, g=g) 
    H1 = utils.H_Thirring(1, l, m, g=g) 
    Hchain = [H0, H1]
    here = sys.path[0]
    fullpath = here + path
    #E = lambda mymps: observers.average(mymps, Hchain)
    E = lambda mymps: observers.allsites(mymps, Hchain)
    observables = [("E", E, True)]

    mps = finitemps.randominfinitemps(2, 2, min(maxchi, 8), maxchi=maxchi, minlam=minlam)
    evolver = idmrgevolver.iDMRG_Evolver(mps, Hchain,  outdir=fullpath, 
                                       timestamp=False, to_observe=observables, 
                                       scriptcopy=False)

    evolver.evolveNsteps(nsteps)

    print("Complete!")
    return mps
