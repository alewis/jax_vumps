""" This file stores utility functions useful to many different MPS 
implementations.  
"""
import numpy as np
from scipy.optimize import minimize_scalar
from bhtools.tebd.constants import Sig_x, Sig_y, Sig_z, Sig_u, Sig_d, Sig_alpha, Sig_beta
from collections import MutableSequence
import scipy.special as spe
import scipy.linalg as spla
from functools import reduce
from bhtools.tebd.scon import scon

def abs2(x):
    return x.real**2 + x.imag**2

def largest_difference(arraysA, arraysB):
    """
    Return the largest elementwise frobenius difference between two lists of 
    arrays.
    """
    diffs = [np.sqrt(A**2 - B**2) for A, B, in zip(arraysA, arraysB)]
    maxdiff = 0
    for diff in diffs:
        thismax = np.amax(np.abs(diff))
        if thismax > maxdiff:
            maxdiff = thismax
    return maxdiff

def frobnorm(A, B=None):
    print("size: ", A.size)
    if B is None:
        B = np.zeros(A.shape)
    ans = (1./A.size)*np.sqrt(np.sum(np.abs(np.ravel(A)-np.ravel(B))**2))
    return ans

class ModularList(MutableSequence):
    """
    A list that wraps back on itself. Construct from a standard list.

    modlist = ModularList["A", "B", "C"]
    modlist[2] 
    >> "B"
    modlist[4]
    >> "A"
    modlist[1200]
    >> "A"
    print(m for m in modlist)
    >>"A", "B", "C"
    big = modlist[0:7:2]
    >> ["A", "C", "B", "A"]
    big[5] = "D"
    >> ["D", "C", "B", "A"]
    big[:-1]
    >> ["A", "B", "C", "D"]
    del modlist[1]
    >> ["A", "C"]
    del modlist[5]
    >> IndexError()
    """

    def __init__(self, data):
        """ Construct a ModularList from a list. No type-checking is done.
            The list itself is stored as a member.
        """
        self.list = data

    # def __deepcopy__(self, memo):
        # return ModularList(copy.deepcopy(self.list))


    def __len__(self):
        """ Returns the length of the internal list.
        """
        return len(self.list)

    def sort(self, key=None, reverse=False):
        """ Sorts the internal list.
        """
        return self.list.sort(key=key, reverse=reverse)

    def insert(self, key, x):
        """ Inserts an element into the internal list at position x.
            This function does not behave modularly.
        """
        self.list.insert(key, x)

    def __iter__(self):
        """ This handles list comprehensions, which do not behave 
            modularly.
        """
        return (x for x in self.list.__iter__() if x is not None)

    def __slicetorange(self, key):
        """ Converts a slice to a range over the sliced indices.
        """
        rawstop = len(self) if key.stop is None else key.stop
        start, stop, step = key.indices(rawstop)
        return range(start, stop, step)
    
    def __getitem__(self, key):
        """ Retrieve the indices spanned by 'key', which is either a slice
            or an int. The index wraps around len(self) modularly.
        """
        if isinstance(key, slice):
            return [self.__getitem__(ii) for ii in self.__slicetorange(key)]
        return self.list[key%len(self)]
    
    def __setitem__(self, key, val):
        """ Set the indices spanned by 'key', which is either a slice
            or an int, to the values val. val is assumed to have the same length
            as key. The index wraps around len(self) modularly.
        """
        if isinstance(key, slice):
            therange = self.__slicetorange(key)
            for ii, vv in zip(therange, val):
                self.__setitem__(ii, vv)
        self.list[key%len(self)] = val
    
    def __delitem__(self, key):
        """ Delete the entry at 'key' from the internal list. This function
            does not wrap modularly.
        """
        del self.list[key]

def extendify(ops, As):
    opidxs = [idx for op, idx in ops if idx is not None]
    if opidxs:
        opmats = [op for op, idx in ops]
        maxlength = max([len(op.shape)//2 for op in opmats])
        minidx, maxidx = (min(opidxs), max(opidxs))
        maxidx += (maxlength-1)
        As, shift = embiggen(As, minidx, maxidx)
        newidxs = [idx - shift for idx in opidxs]
        ops = [(opmat, idx) for opmat, idx in zip(opmats, newidxs)]
    return (ops, As)

def embiggen(mylist, start, stop):
    """
    Return a list containing enough copies of the 
    data within a list that the
    span between 'start' and 'stop' is accomodated. E.g.
    mylist = ["A", "B", "C"])
    embiggen(mylist, 5, 10)
    >>(["A", "B", "C", "A", "B", "C", "A", "B", "C"], 4)
             ^5                       ^10
    """
    N = len(mylist)
    #span = stop-start
    shift = N*(start//N)
    #shiftstart = start - shift
    shiftstop = stop - shift
    periods = shiftstop//N + 1
    biglist = mylist*periods
    return (biglist, shift)


def sortby(es, vecs, mode="LM"):
    """
    The vector 'es' is sorted, 
    and the i's in 'vecs[:, i]' are sorted in the same way. This is done
    by returning new, sorted arrays (not in place). 'Mode' may be 'LM' (sorts
    from largest to smallest magnitude) or 'SR' (sorts from most negative
    to most positive real part).
    """
    if mode=="LM":
        sortidx = np.abs(es).argsort()[::-1]
    elif mode=="SR":
        sortidx = (es.real).argsort()
    essorted = es[sortidx]
    vecsorted = vecs[:, sortidx]
    return essorted, vecsorted


def isqrt(n):
    i = int(np.round(np.sqrt(n)))
    if i**2 != n:
        raise ValueError("Input", n, " was not a perfect square.") 
    return i


def Uexp(H, thresh=1.0E10, verbose=True):
    """Compute exp(U).
       Where U has shape (d, d, d, d).
    """
    Hdag = np.transpose(np.conj(H))
    hermiticity = np.sqrt(np.sum(H**2-Hdag**2))/H.size
    if np.abs(hermiticity) and verbose > 1E-12: 
        print("Warning: H was only Hermitian up to ", hermiticity)

    d = H.shape[0]
    Umat = H.reshape((d*d, d*d))
    Umatexp = spla.expm(Umat)

    maxval = np.amax(np.abs(Umatexp))

    if verbose and maxval > thresh:
        print("Warning: U contains the very large value ", maxval)
        maxH = np.amax(np.abs(Umat))
        print("Max of H ", maxH)
    Uexp = Umatexp.reshape((d, d, d, d))
    return Uexp

def correlator_string(sep, c=1):
    """
    Builds the JW string
    c*(-i)^sep x sig^d x Prod_(s-1) sig^z x sig^u
    """
    # thestring = [Sig_u]
    # for m in range(1, sep):
        # thestring.append(Sig_z)
    # thestring.append(Sig_d)
    # coef = (-1.0j)**sep
    # thestring = [op*coef for op in thestring]
    coef = c*(-1.0j)**sep
    thestring = [coef*Sig_d]
    for m in range(1, sep):
        #thestring.append(np.eye(2, dtype=np.complex128))
        thestring.append(Sig_z)
    thestring.append(Sig_u)
    thestring = [op for op in thestring]
    return thestring

"""
Given r*, return 1-2M/r where r is defined implicitly by
r* = r + 2M ln( (r-2M)/2M) .
"""
def omegarstar(rstar, M):
    def err(r):
        # if r<=2.*M:
            # r = 2.*M + 1E-14
        return (rstar - r - 2.*M*np.log((r-2.*M)/2.*M))**2
    
    res = minimize_scalar(err, bounds=(2*M, 1E20), method='bounded')
    if not res.success:
        raise ValueError("Inversion to get r(r*) failed.")
    return 1.-(2.*M/res.x)

def join_H(H1, H2):
    """
    Fuse two nearest-neighbour Hamiltonians into one acting on a 
    contracted unit-cell.
    """
    d = H1.shape[0]
    II = np.eye(d**2, dtype=np.complex128).reshape((d, d, d, d))
    contract = [H1, H2, II]
    idxs = [ (-1, 1, -5, -6),
            (-2, -3, 1, 2),
            (2, -4, -7, -8) ]
    bigH = scon(contract, idxs)
    bigH = bigH.reshape((d**2, d**2, d**2, d**2))
    return bigH

def H_XX():
    H = np.kron(Sig_x, Sig_x) + np.kron(Sig_y, Sig_y)
    H = H.reshape(2, 2, 2, 2)
    return H

def H_diracflat(n, a, m):
    return H_dirac(n, a, m)

def H_dirac(n, l, m=0):
    """
    H = (1/4l)*(X_n X_n+1 + Y_n Y_n+1) + (m/2)*(-1)^n *(Z_n - eye_n)
    """
    H = (1.0/(4*l)) * (np.kron(Sig_x, Sig_x) + np.kron(Sig_y, Sig_y))
    if n%2 == 1: #if n is odd 
        m *= -1
    if m!=0:
        I = np.eye(2)
        oneminusz = 0.5*(I - Sig_z) #i'm pretty sure this is sig_down
        H += m*np.kron(oneminusz, I)
    H = H.reshape((2,2,2,2))
    return H

def foursite(A, B, C, D):
    return reduce(np.kron, [A, B, C, D])

def H_join(H1, H2):
    I = np.eye(2, dtype=H1.dtype)
    H1 = H1.reshape((4,4))
    H2 = H2.reshape((4,4))
    t1 = reduce(np.kron, [H1, I, I])
    t2 = reduce(np.kron, [I, H2, I])
    H = t1 + t2
    H = H.reshape((4,4,4,4))
    return H


def H_ThirringBig(l, m, g=None):
    I = np.eye(2, dtype=np.complex128)
    XX = foursite(Sig_x, Sig_x, I, I)
    XX += foursite(I, Sig_x, Sig_x, I)
    XX += foursite(I, I, Sig_x, Sig_x)
    
    YY = foursite(Sig_y, Sig_y, I, I)
    YY += foursite(I, Sig_y, Sig_y, I)
    YY += foursite(I, I, Sig_y, Sig_y)
    H_kin = (1.0/(4*l))*(XX + YY)

    loc = Sig_z - I
    locplus = foursite(loc, I, I, I) + foursite(I, I, loc, I)
    locminus = foursite(I, loc, I, I) + foursite(I, I, I, loc)
    locfull = locplus - locminus
    H_mass = (m/2)*locfull

    H_int = 0.
    if g is not None:
        ZZ = foursite(Sig_z, Sig_z, I, I)
        H_int = -(1/8)*(1/l)*g*ZZ

    H = H_kin + H_mass + H_int
    H = H.reshape((4, 4, 4, 4))
    return H



def H_ThirringDMRG(n, l, m, g=None, LR='L'):
    H_free = H_dirac(n, l, m=m).reshape((4,4))
    if g is None or g==0. or g==0:
        return H_free.reshape((2,2,2,2))
    #int_coef = (1/8)*(1/l)*g
    #int_mat = np.eye(2) - Sig_z
    #H_int = -int_coef*np.kron(int_mat, int_mat)
    H_int = -(1/8)*(1/l)*g*np.kron(Sig_z, Sig_z)
    H_full = H_free + H_int
    H_full = H_full.reshape((2,2,2,2))
    return H_full

def H_Thirring(n, l, m, g=None):
    H_free = H_dirac(n, l, m=m).reshape((4,4))
    if g is None or g==0. or g==0:
        H_full = H_free.reshape((2,2,2,2))
    else:
        H_int = -(1/8)*(1/l)*g*np.kron(Sig_z, Sig_z)
        H_full = H_free + H_int
        H_full = H_full.reshape((2,2,2,2))
    return H_full

def H_Thirring_ModD(l, n=0, m=None, g=None):
    """
    The Thirring Hamiltonian, modified to incorporate current 
    conservation.
    """
    eps = (np.pi/(1+g/np.pi))*(g/np.pi + 0.5)
    G =  (-4*eps/np.pi)*(1/np.tan(eps))
    nu = 2*eps / (np.pi*np.sin(eps))
    plus_minus = np.kron(Sig_u, Sig_d)
    minus_plus = np.kron(Sig_d, Sig_u)
    #kinetic_term = (1/2)*(nu/l)*(plus_minus + minus_plus)
    kinetic_term = (1/2)*(nu/l)*(plus_minus + minus_plus)

    mass_term = 0
    I = np.eye(2)#, dtype=np.complex128)
    if m is not None:
        #mtilde = m / nu
        if n%2 == 1:
            m *= -1
        mass_term = m * np.kron(Sig_beta, I)
        #mass_term -= 0.5 * m * np.kron(I, Sig_beta)

    int_term = 0
    if g is not None:
        #int_term = delta * (nu/l) * np.kron(Sig_beta, Sig_beta)
        betaminus = Sig_beta - 0.5*I
        int_term = -(G/(4*l)) * (np.kron(betaminus, betaminus))
    # lam = 0
    # if n%2 == 1:
        # lam *= -1
    # Hpen = lam*np.kron(Sig_z,I)
    

    H = kinetic_term + mass_term + int_term #+ Hpen
    H = H.reshape((2, 2, 2, 2)).real
    return H


def H_Thirring_Mod(l, n=0, m=None, g=None):
    """
    The Thirring Hamiltonian, modified to incorporate current 
    conservation.
    """
    if g is None:
        gam = 0.5*np.pi
    else:
        gam = 0.5*(np.pi-g)
    nu = 2*gam/(np.pi*np.sin(gam))
    plus_minus = np.kron(Sig_u, Sig_d)
    minus_plus = np.kron(Sig_d, Sig_u)
    #kinetic_term = (1/2)*(nu/l)*(plus_minus + minus_plus)
    kinetic_term = (1/2)*(nu/l)*(plus_minus + minus_plus)

    mass_term = 0
    I = np.eye(2)#, dtype=np.complex128)
    if m is not None:
        #mtilde = m / nu
        if n%2 == 1:
            m *= -1
        mass_term = m * np.kron(Sig_beta, I)
        #mass_term -= 0.5 * m * np.kron(I, Sig_beta)

    int_term = 0
    if g is not None:
        delta = np.cos(gam)
        #int_term = delta * (nu/l) * np.kron(Sig_beta, Sig_beta)
        betaminus = Sig_beta - 0.5*I
        int_term = delta * (nu/l) * (np.kron(betaminus, betaminus))
    # lam = 0
    # if n%2 == 1:
        # lam *= -1
    # Hpen = lam*np.kron(Sig_z,I)
    

    H = kinetic_term + mass_term + int_term #+ Hpen
    H = H.reshape((2, 2, 2, 2)).real
    return H




def H_ScaledThirring(n, ml, gtilde=None):
    H_free = H_dirac(n, 1, m=ml).reshape((4, 4))

    if gtilde is None or gtilde==0. or gtilde==0:
        return H_free.reshape((2,2,2,2))
    H_int = -(1/4)**gtilde*np.kron(Sig_z, Sig_z)
    H_full = H_free + H_int
    H_full = H_full.reshape((2,2,2,2))
    return H_full

def H_rindler(n, l, m, alpha, x0=0):
    omfunc = lambda x: np.exp(alpha*x)
    omratio = lambda x: alpha
    return H_diraccurved(n, l, m, omegafunc=omfunc, omegaprimeratio=omratio,
            x0=x0)

def H_diraccurved(n, l, m, omegafunc=None, 
        omegaprimeratio=None, x0=0., thresh=1E8):
    if omegafunc is None:
        omegafunc = lambda x: 1.
    if omegaprimeratio is None:
        omegaprimeratio = lambda x: 0.


    thisomprimerat = omegaprimeratio(x0+n*l)
    coef = (1/2)*(1.0/l + 0.5*thisomprimerat)
    H = coef*(1/2)*(np.kron(Sig_x, Sig_x) + np.kron(Sig_y, Sig_y))
    if n%2 == 1 :
        m *= -1
    if m!= 0:
        thisomfunc = omegafunc(x0+n*l)
        if np.abs(thisomfunc) > thresh:
            print("Warning: Omega had the very large value ", thisomfunc)


        I = np.eye(2, dtype=np.complex128)
        oneminusz = 0.5*(I - Sig_z) #i'm pretty sure this is sig_down
        H += m*thisomfunc*np.kron(oneminusz, I)
        #H += 0.5*m*thisomfunc*(np.kron(oneminusz, I) + np.kron(I, oneminusz))
    H = H.reshape((2,2,2,2))
    
    return H






def diracT10(a):
    term1 = reduce(np.kron, (Sig_u, Sig_z, Sig_d))
    term2 = np.conj(term1).T
    T10 = (-1.0j/(4*a)) * (term1 - term2)
    shape = [2,]*6
    return T10.reshape(shape)

def diracT11(a):
    term1 = np.kron(Sig_u, Sig_d)
    term2 = np.conj(term1).T
    T11 = (1.0/(2*a))*(term1 + term2)
    return T11.reshape([2]*4)


"""
 the :w
"""
def buildHam(thissiteops, nextsiteops):
    d = thissiteops[0].shape[0]
    ham = tc.twositezeros(d)
    for a, b, in zip(thissiteops, nextsiteops):
        ham.data += tc.twositeop(a, b).data
    return ham


     


#Check signs
# def HLW(Xl, Yl, a):
    # chil = Xl.shape[0]
    # if Yl.shape[0] != chil:
        # raise ValueError("Inconsistent shapes: ", Xl.shape, Yl.shape)
    # H = (1./(4.*a)) * (np.kron(Xl, Sig_x) + np.kron(Yl, Sig_y))
    # H = np.reshape(H, (chil, 2, chil, 2))
    # return H

# def HWR(Xr, Yr, n, a, m, M):
    # chir = Xr.shape[0]
    # if Yr.shape[0] != chir:
        # raise ValueError("Inconsistent shapes: ", Xr.shape, Yr.shape)
    # H = (1./(4.*a)) * (np.kron(Sig_x, Xr) + np.kron(Sig_y, Yr))
    # omega = omegarstar(n*a, M)
    # if n%2 == 1:
        # m *= -1
    # if m!=0:
        # H += m * np.kron(Sig_u, Sig_d)
    # H = H.reshape(H, (2,2,2,2))
    # return H

"""
    kronpermute(M, perm):

M: an array of shape (2*d,)*N
perm: a container of N integers 
(N and d are derived from these)

A is assumed to be the result of N Kronecker products, i.e.
for N=5 and d*d matrices A,B,C,D,E, M=reduce(kron, (A,B,C,D,E)).
kronpermute returns the output which would have arisen from
G=reduce(kron, array((A,B,C,D,E)).transpose(perm)). Thus 
for perm=(2,1,3,0,4) we would get reduce(kron, (C,B,D,A,E)).

Note that M^T = A^T B^T C^T D^T E^T. Thus if transposed product
matrices are desired it is sufficient to feed in M^T as input.

G is returned with the same shape as M.
"""
def kronpermute(M, perm):
    Mshp = M.shape
    N = len(Mshp)
    d = int(Mshp[0]/2)

    if not (max(perm) == N-1 and min(perm)) == 0:
        raise ValueError("Invalid permutation ", perm)
    if not (np.array(Mshp) == np.array([2*d,]*N)).all():
        raise ValueError("Input matrix M had invalid shape ", M.shape)
    if not len(perm) == N:
        raise ValueError("perm", perm, "must be same length as M.shape ", Mshp)
   
    perm2 = [p + N for p in perm]
    fullperm = list(perm) + perm2
    G = M.reshape(*(2*N)*(d,)).transpose(fullperm).reshape(d**N, d**N)
    return G

"""
H: a [d,]*6 numpy array.

Given H = A  B  C, return 
G = 0.5*(AIBICI + IAIBIC.T)
"""
def ancillanextnearest(H):
    d = H.shape[0]
    if not H.shape == (d,)*6:
        raise ValueError("H had invalid shape ", H.shape, "expected ",
                (d,)*6)
    I = np.eye(d) 
    H = H.reshape((d**3, d**3))
    ABCIII = reduce(np.kron, (H, I, I, I)).reshape((d**2,)*6)
    AIBICI = kronpermute(ABCIII, (0,3,1,4,2,5)).reshape((d**6, d**6))
    IAIBIC_T = kronpermute(ABCIII, (3,0,4,1,5,2)).reshape((d**6, d**6)).T
    ancilla = 0.5*(AIBICI + IAIBIC_T)
    return ancilla.reshape((d**2,)*6)
"""
H: a (d,d,d,d) numpy array.

Given H = A  B, return 
G = 0.5*(AIBI + IAIB.T)
"""
def ancillanearest(H):
    d = H.shape[0]
    if not H.shape == (d,d,d,d):
        raise ValueError("H had invalid shape ", H.shape, "expected ",
                (d,d,d,d))
    I = np.eye(d) 
    H = H.reshape((d**2, d**2))
    ABI = np.kron(H, I).reshape((d**2, d**2, d**2))
    AIB = kronpermute(ABI, (0,2,1)).reshape((d**3,d**3))
    IAIB_T = np.kron(I, AIB).T
    AIBI = np.kron(AIB, I)
    h_anc = 0.5*(AIBI + IAIB_T)#.reshape((d**2, d**2, d**2, d**2))
    return h_anc.reshape((d**2,d**2,d**2,d**2))

"""
    supersiteop(op)
op: a dxd numpy array

Given a one-site operator op, form the supersite operator 
op_s = 0.5*(op I + I op^T). This is the single-site analogue to
formancillaham, which is for nearest-neighbour operators.
"""
def supersiteop(op):
    d = op.shape[0]
    if not (len(op.shape) == 2 and op.shape[1] == d):
        raise ValueError("Input had invalid shape ", op.shape)
    eye = np.eye(d)
    return 0.5*(np.kron(op, eye) + np.kron(eye, op.T))


def H_diracSC(x0, n, a, m, M):
    x = x0 + n*a + 0.5*a
    omega = omegarstar(x, M)
    return H_dirac(n, a, m*omega)


# def H_FLRWstepscalar(t, hparms, last=True):
    # """
    # 2(n)H = pi_n^2/(l^2*C(n)) + C(n)(phi_(n+1)-phi_n)^2/l^2 + m^2 phi_n^2
    # C(n) = A + B*tanh(C*n).
    # """
    # m = hparms[0]      #mass
    # delta = hparms[1]  #lattice spacing
    # phi = hparms[2]    #field operator
    # pi  = hparms[3]    #momentum operator
    # d   = hparms[4]    #SP hilbert space dimension
    # A   = hparms[5]    #conformal factor.
    # tstep = hparms[6]

    # if t < tstep:
      # c_eta = 1.
    # else:
      # c_eta = A
    # eye = np.eye(d)

    # m2phiphi = m*m*np.dot(phi, phi)
    # delta2pipi = np.dot(pi, pi)
    # massterm = np.kron(m2phiphi, eye)
    # piterm = (1/(delta**2*c_eta)) * np.kron(delta2pipi, eye) 

    # phi_n = np.kron(phi, eye) 
    # phi_np1 = np.kron(eye, phi)
    # forward_diff = phi_np1 - phi_n
    # derivterm = (c_eta/delta**2)*np.dot(forward_diff, forward_diff)#/(delta**2)
    # H = 0.5*np.sqrt(c_eta)*(piterm + derivterm + massterm)
    # Hshpd = H.reshape((d, d, d, d))
    # return Hshpd


# def H_FLRWscalar(t, hparms, last=True):
    # """
    # 2(n)H = sqrt(C(n))*[pi_n^2/(C(n)) + C(n)(phi_(n+1)-phi_n)^2/l + l*m^2 phi_n^2]
    # C(n) = A + B*tanh(C*n).
    # """
    # m = hparms[0]      #mass
    # delta = hparms[1]  #lattice spacing
    # phi = hparms[2]    #field operator
    # pi  = hparms[3]    #momentum operator
    # d   = hparms[4]    #SP hilbert space dimension
    # A   = hparms[5]    #A in conformal time.
    # B   = hparms[6]    #B in conformal time.
    # C   = hparms[7]    #C in conformal time.

    # c_eta = A + B * np.tanh(C*t)
    # eye = np.eye(d)

    # #delta2 = delta*delta
    # m2phiphi = delta*m*m*np.dot(phi, phi)
    # pipi = np.dot(pi, pi)
    # massterm = np.kron(m2phiphi, eye)
    # piterm = (1./(c_eta)) * np.kron(pipi, eye) 

    # phi_n = np.kron(phi, eye) 
    # phi_np1 = np.kron(eye, phi)
    # forward_diff = phi_n - phi_np1 
    # derivterm = (c_eta/delta)*np.dot(forward_diff, forward_diff)
    # H = 0.5*np.sqrt(c_eta)*(piterm + derivterm + massterm)
    # Hshpd = H.reshape((d, d, d, d))
    # return Hshpd
# def H_FLRWscalar(t, hparms, last=True):
    # """
    # 2(n)H = l*sqrt(C(n))*[pi_n^2/(l*C(n)) + C(n)(phi_(n+1)-phi_n)^2/l^2 + m^2 phi_n^2]
    # C(n) = A + B*tanh(C*n).
    # """
    # m = hparms[0]      #mass
    # delta = hparms[1]  #lattice spacing
    # phi = hparms[2]    #field operator
    # pi  = hparms[3]    #momentum operator
    # d   = hparms[4]    #SP hilbert space dimension
    # A   = hparms[5]    #A in conformal time.
    # B   = hparms[6]    #B in conformal time.
    # C   = hparms[7]    #C in conformal time.

    # c_eta = A + B * np.tanh(C*t)
    # eye = np.eye(d)

    # delta2 = delta*delta
    # m2phiphi = m*m*np.dot(phi, phi)
    # delta2pipi = np.dot(pi, pi)
    # massterm = np.kron(m2phiphi, eye)
    # piterm = (1/(c_eta*delta2)) * np.kron(delta2pipi, eye) 

    # phi_n = np.kron(phi, eye) 
    # phi_np1 = np.kron(eye, phi)
    # forward_diff = phi_np1 - phi_n
    # derivterm = (c_eta/delta2)*np.dot(forward_diff, forward_diff)#/(delta**2)
    # H = 0.5*delta*np.sqrt(c_eta)*(piterm + derivterm + massterm)
    # Hshpd = H.reshape((d, d, d, d))
    # return Hshpd

def H_flatscalar(t, hparms, last=True):
    """
    2H = pi_n^2/a^2 + (phi_(n+1)-phi_n)^2/a^2 + m^2 phi_n^2

    """
    m = hparms[0]      #mass
    delta = hparms[1]  #lattice spacing
    phi = hparms[2]    #field operator
    pi  = hparms[3]    #momentum operator
    d   = hparms[4]    #SP hilbert space dimension
    eye = np.eye(d)

    m2phiphi = m*m*np.dot(phi, phi)
    delta2pipi = (1/(delta**2))*np.dot(pi, pi)
    massterm = np.kron(m2phiphi, eye)
    piterm = np.kron(delta2pipi, eye)/(delta**2) 

    phi_n = np.kron(phi, eye) 
    phi_np1 = np.kron(eye, phi)
    forward_diff = phi_np1 - phi_n
    derivterm = np.dot(forward_diff, forward_diff)/(delta**2)
    H = 0.5*(piterm + derivterm + massterm)
    Hshpd = H.reshape((d, d, d, d))
    return Hshpd

def H_ising(J, h):
    ham = J*np.kron(Sig_x, Sig_x) + h*np.kron(Sig_z, np.eye(2))
    ham = ham.reshape(2, 2, 2, 2)
    return ham

def H_XXZ(delta=1, nu=2, l=1):
    """
    H = (-1/8l)*[nu*[UD + DU] + delta*ZZ]
    """
    H = nu*(np.kron(Sig_u, Sig_d) + np.kron(Sig_d, Sig_u)) 
    H += delta*np.kron(Sig_z, Sig_z)
    H *= -(1/(8*l))
    H = H.real
    return H.reshape((2,2,2,2))

# def H_ising(t, hparms, last=False):
    # J = hparms[0]
    # h = hparms[1]
    # H = -J * (np.kron(Sig_x, Sig_x)) 
    # H += h*np.kron(Sig_z, np.eye(2))
    # #H += 0.5*h * (np.kron(Sig_z, np.eye(2)) + np.kron(np.eye(2), Sig_z))
    # H *= -1
    # Hshpd = H.reshape((2,2,2,2))
    # #print Hshpd
    # return Hshpd

def MPO_ham(ham):
    print("joaijobijeo")
    d = ham.shape[0]
    MPOd = d**2
    ham = ham.reshape((MPOd, MPOd))
    eye = np.eye(MPOd)
    #MPOh = 0.5*(np.kron(ham, eye) + np.kron(eye, ham.T))
    MPOh = 0.5*(np.kron(ham.T, eye) + np.kron(eye, ham))
    return MPOh.reshape((MPOd, MPOd, MPOd, MPOd))





def U_eye():
    eye = np.kron(np.eye(2), np.eye(2)).reshape((2,2,2,2))
    return eye

def random_rank(n, k):
    """Make a random nxn matrix of rank-k.
    """
    if k>n:
        raise ValueError("k", k, "was greater than n", n)
    out = np.zeros((n,n), dtype=np.complex128)
    for thisk in range(0, k):
        vec = random_complex((n, 1))
        out += np.kron(vec.T, vec)
    return out

def random_rng(shp, low=0, high=1.0):
    return (high - low) * np.random.random_sample(shp) + low

def random_hermitian(diaglength):
    A = random_complex((diaglength, diaglength))
    return 0.5*(A+np.conj(A).T)

def random_unitary(diaglength):
    A = random_hermitian(diaglength)
    return spla.expm(-1.j*A)


def unitary_gamma(d, chi):
    U = random_unitary(d*chi)
    U = U[:chi, :].reshape((chi, d, chi))
    U = U.transpose((1,0,2))
    return U

def random_complex(shp, real_low=-1.0, real_high=1.0, imag_low=-1.0, 
        imag_high=1.0):
    """Return a normalized randomized complex matrix of shape shp.
    """
    realpart = random_rng(shp, low=real_low, high=real_high)
    imagpart = 1.0j * random_rng(shp, low=imag_low, high=imag_high)
    bare = realpart + imagpart
    #bare /= la.norm(bare)
    return bare

def randomunitary(N):
    A = random_complex((N, N))/np.sqrt(2) 
    Q, R = spla.qr(A)
    r = np.diag(R) 
    L = r / np.abs(r) 
    return Q*L
"""
Returns a (d, chi, chi) gamma tensor normalized so that it has r proportional
to I and largest tm eigenvalue = 1.
"""
def randomunitarygamma(d, D):
    U = randomunitary(d*D)[0:D, :]
    return U.reshape((d, D, D))

    

def testham(perm=(0,1,2,3)):
    from .scon import scon

    A = random_complex((2,3,3))
    B = random_complex((2,3,3))
    to_contract1 = [Sig_y, Sig_x, A, B]
    idx1 = [ [1, -1],
             [2, -2],
             [1, -3, 3],
             [2, 3, -4] ]
    AB1 = scon(to_contract1, idx1)
    
    H = np.kron(Sig_y, Sig_x)
    H = H.reshape(2,2,2,2)
    H = np.transpose(H, perm)
    to_contract2 = [H, A, B]
    idx2 = [ [1, 2, -1, -2],
             [1, -3, 3],
             [2, 3, -4] ]
    AB2 = scon(to_contract2, idx2)

    print("perm:", perm)
    print("diff: ", AB1-AB2)


def isingexactE(j, h):
    lam = j/h
    E = -h*2/np.pi * (1+lam) * spe.ellipe((4*lam/(1+lam)**2))
    return E


