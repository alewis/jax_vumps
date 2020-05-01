import numpy as np
import bhtools.tebd.utils as utils
from bhtools.tebd.constants import Sig_d, Sig_z, Sig_u, Sig_beta
import bhtools.tebd.contractions as ct
import copy

def good_corrs(corrs, lf, Nr):
    good_idxs = np.arange(lf-1, lf*Nr, lf)
    good_array = corrs[:, good_idxs]
    return good_array

def JW_corrs_2cell(A0, A1, maxsep, l=1):

    throughput = ct.XopL(A0, O=Sig_d)
    throughput = ct.XopL(A1, O=Sig_z, X=throughput)
    capE = ct.XopR(A0, O=Sig_u)
    capO = ct.XopR(A1, O=Sig_u)
    oshp = (8, maxsep)

    output = np.zeros(oshp, dtype=np.complex128)
    #output[:4, 0] = [2, 3, 1, 2] #the coordinate separations
    output[0, :] = 2*l*np.arange(1, maxsep+1)
    output[1, :] = output[0, :] + l 
    sign0 = 1.0j
    for m in range(0, maxsep):
        s = 2*(m+1)
        sign = sign0**s
        throughput = ct.XopL(A0, X=throughput, O=Sig_z)
        output[4, m] = (1/l)*sign*ct.proj(throughput, capE)
        
        s += 1
        sign = sign0**s
        throughput = ct.XopL(A1, X=throughput, O=Sig_z)
        output[5, m] = (1/l)*sign*ct.proj(throughput, capO)

    return output

def JW_corrs(mpslist, maxsep, l=1):#, supersite=True):
    """
    Compute the JW correlators mpslist at each separation (number
    of inserted transfer matrices)
    from 0 to maxsep. This function reuses intermediate results
    and is thus much more efficient than computing each correlator
    separately.

    Does not compute the local (quadratic) correlator.
    """
    A_L, C, A_R = mpslist
    if A_L.shape[0]==4:
        return JW_corrs_supersite(mpslist, maxsep, l=l)

    A_C = ct.rightmult(A_L, C)
    throughput = ct.XopL(A_C, O=Sig_d)
    cap = ct.XopR(A_R, O=Sig_u)
    oshp = (8, maxsep)

    output = np.zeros(oshp, dtype=np.complex128)
    #output[:4, 0] = [2, 3, 1, 2] #the coordinate separations
    output[0, :] = 2*l*np.arange(1, maxsep+1)
    output[1, :] = output[0, :] + l
    sign0 = 1.0j
    for m in range(0, maxsep):
        s = 2*(m+1)
        sign = sign0**s
        throughput = ct.XopL(A_R, X=throughput, O=Sig_z)
        output[4, m] = (1/l)*sign*ct.proj(throughput, cap)
        
        s += 1
        sign = sign0**s
        throughput = ct.XopL(A_R, X=throughput, O=Sig_z)
        output[5, m] = (1/l)*sign*ct.proj(throughput, cap)

    return output

def JW_corrs_supersite(mpslist, maxsep, l=1):
    """
    Compute the JW correlators mpslist at each separation (number
    of inserted transfer matrices)
    from 0 to maxsep. This function reuses intermediate results
    and is thus much more efficient than computing each correlator
    separately.

    Does not compute the local (quadratic) correlator.
    """
    A_L, C, A_R = mpslist
    A_C = ct.rightmult(A_L, C)
    I = np.eye(2, dtype=np.complex128)
    op0_E = np.kron(Sig_d, Sig_z)
    op0_O = np.kron(I, Sig_d)
    op1_E = np.kron(Sig_u, I)
    op1_O = np.kron(Sig_z, Sig_u)

    
    #Close the left of each chain.
    throughput = [ct.XopL(A_C, O=op0) 
                    for op0 in [op0_E, op0_E, op0_O, op0_O]]

    #Close the right of each chain.
    caps = [ct.XopR(A_R, O=op1) for op1 in [op1_E, op1_O, op1_E, op1_O]]
    #offsets = [0, 1, -1, 0]
    coefs = [1, 1.0j, 1, 1.0j]
    oshp = (8, maxsep)

    output = np.zeros(oshp, dtype=np.complex128)
    output[:4, 0] = [2, 3, 1, 2] #the coordinate separations
    
    ZZ = np.kron(Sig_z, Sig_z)
    for m in range(0, maxsep):
        coefs = [-coef for coef in coefs] #the factors of i from JW
        for i in range(len(caps)): 
            output[i+4, m] = (1/l)*coefs[i]*ct.proj(throughput[i], caps[i])

            throughput[i] = ct.XopL(A_R, X=throughput[i], O=ZZ)
            output[i, m] = output[i, 0] + 2*m
    output[:4, :] = output[:4, :] * l
    return output

def single_site(mpslist, O):
    A_L, C, A_R = mpslist
    A_C = ct.rightmult(A_L, C)
    left = ct.XopL(A_C, O=O)
    right = ct.XopR(A_R)
    ev = ct.proj(left, right)
    return ev

def single_site_thirring(mpslist, O, even=True):
    I = np.eye(2, dtype=np.complex128)
    d = mpslist[0].shape[0]
    if d==4:
        if even:
            op = np.kron(O, I)
        else:
            op = np.kron(I, O)
    elif d==2:
        op = O
    else:
        raise ValueError("Bad physical dimension "+str(d))
    return single_site(mpslist, op)

def electron_spin(mpslist):
    return single_site_thirring(mpslist, Sig_z, even=True).real

def positron_spin(mpslist):
    return single_site_thirring(mpslist, Sig_z, even=False).real

def total_spin(mpslist):
    e = electron_spin(mpslist)
    p = positron_spin(mpslist)
    return e+p

def electron_density(mpslist):
    return single_site_thirring(mpslist, Sig_beta, even=True).real

def positron_density(mpslist):
    return single_site_thirring(mpslist, Sig_beta, even=False).real

def charge_density(mpslist):
    e = electron_density(mpslist)
    p = positron_density(mpslist)
    return e+p


def JWcorrs_at_seps(mps, site0, seps, c=1):
    """
    Generates Jordan-Wigner from site0 to each site in indices.
    """
    ops = [utils.correlator_string(s, c=c) for s in seps]
    answers = [opstringev(mps, op, site0) for op in ops]
    return answers



def sitestats(mps, op):
    exp = siteexpect(mps, op)
    return np.mean(exp), np.std(exp) 

def siteexpect(mps, op):
    opsize = int(len(op.shape)/2)
    d = mps.d
    nops = mps.N-opsize+1
    expchain = np.zeros(nops)
    for n in range(nops):
        optup = (op, n)
        expchain[n] = mps.expectationvalue(ops=[optup])
    return expchain

def correlator(mps, op1, site1, op2, site2, real=True):
    answer = mps.correlator(op1, site1, op2, site2, real=real)
    return answer

def opstringev(mps, ops, site0):
    answer = mps.opstringev(ops, site0)
    return answer


def total(mps, opchain, leftoffset=0, rightoffset=0):
    exps = allsites(mps, opchain, leftoffset=leftoffset, rightoffset=rightoffset)
    thetotal = np.sum(exps)
    return thetotal

def average(mps, opchain, leftoffset=0, rightoffset=0):
    thetotal = total(mps, opchain, leftoffset=0, rightoffset=0)
    Nsites = len(opchain)-rightoffset
    themean = thetotal/Nsites
    return np.array([themean])

def allsites(mps, opchain, leftoffset=0, rightoffset=0):
    """
    Compute a list of expectation values at adjacent sites of the mps.
    Those sites are in the range [leftoffset:(-1-rightoffset)].
    opchain is the list of operators whose expectation value is to be computed.
    """
    if mps.N-rightoffset <= leftoffset:
        raise IndexError("Invalid left/right offsets", 
                leftoffset, rightoffset)
    sites = np.arange(leftoffset, len(opchain)-rightoffset)
    # sites = np.arange(0, mps.N)[leftoffset:(-1-rightoffset)]
    if len(sites) > len(opchain):
        raise IndexError("opchain had only", len(opchain), 
                "entries for", len(sites), " sites.") 
    optups = list(zip(opchain[leftoffset:len(opchain)-rightoffset], sites))
    exps = np.array([mps.expectationvalue(ops=[optup,]) for optup in optups] )
    return exps

def schmidtvectors(mps):
    chi = mps.maxchi
    if chi is None:
        raise NotImplementedError("schmidtvector requires that maxchi be set")
    bigarr = np.zeros(len(mps.lams), chi)
    for i, lam in enumerate(mps.lams):
        bigarr[:lam.size, i] = lam[:]
    return bigarr

def schmidtsite(mps, site):
    chi = mps.maxchi
    if chi is None:
        raise NotImplementedError("schmidtvector requires that maxchi be set")
    schmidtarr = np.zeros(chi)
    lenmps = len(mps.lams[site])
    schmidtarr[:lenmps] = mps.lams[site][:]
    return schmidtarr

def schmidttuples(N):
    schmidtnames = ["Lambda_"+str(i) for i in range(N)]
    schmidtsites = [lambda mps: schmidtsite(mps, i) for i in range(N)] 
    tuples = [(name, func, False) for name, func in 
            zip(schmidtnames, schmidtsites)]
    return tuples

def norm(mps):
    return np.array([mps.norm()])

def outputvalue(value):
    return np.array([value])








# def all2site(mps, opchain, x_n=None):
    # """
    # mps: an MPS object of length N
    # Uchain: a chain of N-1 nearest-neighbour operators (dxdxdxd numpy arrays)
    # x_n: an optional function mapping n to a coordinate value. If it is not
         # specified it is set to return n.

    # Compute the expectation value of the two-site operator Uchain for each of 
    # the sites in the chain. Returns an [N-1, N-1] array 
    # arr[:, 0] = x_n(n), arr[:, 1] = <Uchain[n]>. 
    # """
    # if x_n is None:
        # x_n = lambda n: n
    # sites = np.arange(0, mps.N-1)
    # xlist = map(x_n, sites)
    # opzip = zip(opchain, sites)
    # explist = map(lambda z: mps.expectationvalue(ops=[z,]), opzip)
    # return np.array([sites, xlist, explist]).T

# def all3site(mps, Uchain, x_n=None):
    # """
    # mps: an MPS object of length N
    # Uchain: a chain of N-2 nearest-neighbour operators ([d,]*6 numpy arrays)
    # x_n: an optional function mapping n to a coordinate value. If it is not
         # specified it is set to return n.

    # Compute the expectation value of the three-site operator Uchain for each of 
    # the sites in the chain. Returns an [N-2, N-2] array 
    # arr[:, 0] = x_n(n), arr[:, 1] = <Uchain[n]>. 
    # """
    # if x_n is None:
        # x_n = lambda n: n
    # sites = np.arange(1, mps.N-1)
    # xlist = list(map(x_n, sites))
    # explist = list(map(lambda n: mps.canonical3site(mps, Uchain[n-1], n), sites))
    # return np.array([sites, xlist, explist]).T
