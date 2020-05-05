"""
A docstring
"""

import numpy as np
import scipy as sp
import numpy.linalg as npla
import scipy.linalg as spla

from functools import reduce
from bhtools.tebd.scon import scon
import bhtools.tebd.utils as utils
#from scipy.linalg import solve


"""
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




"""
2--A--3
   |
   O
   |
   1
"""
def Aop(A, O):
    return Bop(O.T, A)


"""
   1
   |
   O
   |
2--B--3
"""
def Bop(O, B):
    return scon( [O, B],
                 [(1, -1),
                  (1, -2, -3)])
    #return np.einsum("ij, jkl", O, B)

def leftgather(left, right):
    """
    2--left--right--3
        |     |
        1     1 (vectorized)
        |     |
    Mirror image if fromleft is false.
    """
    dl, chil, chim = left.shape
    dr, chim, chir = right.shape
    #d2 = left.shape[0]*right.shape[0]
    lr = np.dot(left, right).transpose((0,2,1,3))
    lr = lr.reshape((dl*dr, chil, chir))
    return lr

def qrmat(A, mode="full"):
    """
    QR decomp. of A, with phase convention such that R has only positive
    elements on the main diagonal. A is a matrix.
    """
    Q, R = sp.linalg.qr(A, mode=mode)
    phases = np.diag(np.sign(np.diag(R)))
    Q = np.dot(Q, phases)
    R = np.dot(np.conj(phases), R)
    return (Q, R)

def qrpos(A):
    """
    QR decomp. of A, with phase convention such that R has only positive
    elements on the main diagonal. 

    If A is an MPS tensor (d, chiL, chiR), it is reshaped appropriately
    before the throughput begins. In that case, Q will be a tensor
    of the same size, while R will be a chiR x chiR matrix.
    """
    Ashp = A.shape
    if len(Ashp) == 2:
        return qrmat(A)
    elif len(Ashp) != 3:
        print("A had invalid dimensions, ", A.shape)

    A = fuse_left(A) #d*chiL, chiR 
    Q, R = qrmat(A, mode="economic")
    Q = unfuse_left(Q, Ashp)
    return (Q, R)

def rqmat(A, mode="full"):
    """
    RQ decomp. of A, with phase convention such that R has only positive
    elements on the main diagonal. A is a matrix.
    """
    R, Q = sp.linalg.rq(A, mode=mode)
    phases = np.diag(np.sign(np.diag(R)))
    Q = np.dot(phases, Q)
    R = np.dot(R, np.conj(phases))
    return (Q, R)

def rqpos(A):
    """
    RQ decomp. of A, with phase convention such that R has only positive
    elements on the main diagonal. 

    If A is an MPS tensor (d, chiL, chiR), it is reshaped and
    transposed appropriately
    before the throughput begins. In that case, Q will be a tensor
    of the same size, while R will be a chiL x chiL matrix.
    """
    Ashp = A.shape
    if len(Ashp) == 2:
        return rqmat(A)
    elif len(Ashp) != 3:
        print("A had invalid dimensions, ", A.shape)

    A = fuse_right(A) #chiL, d*chiR 
    R, Q = qrmat(A, mode="economic")
    Q = unfuse_right(Q, Ashp)
    return (Q, R)

def fuse_left(A):
    oldshp = A.shape
    d, chiL, chiR = oldshp
    A = A.reshape(d*chiL, chiR)
    return A

def unfuse_left(A, shp):
    return A.reshape(shp)

def fuse_right(A):
    oldshp = A.shape
    d, chiL, chiR = oldshp
    A = A.transpose((1, 0, 2)).reshape((chiL, d*chiR))
    return A

def unfuse_right(A, shp):
    d, chiL, chiR = shp
    A = A.reshape((chiL, d, chiR)).transpose((1, 0, 2))
    return A
    



"""
    |     |
    1     1 (vectorized)
    |     |
3--left--right--4
    |     |
    2     2
    |     |
"""
def leftgathermpo(left, right):
    dl1, dl2, chil, chim = left.shape
    dr1, dr2, chim, chir = right.shape
    if dl1 != dl2 or dr1 !=dr2:
        raise ValueError("bad shapes", left.shape, right.shape)
    d2 = dl1*dr1
    lr = np.dot(left, right) #dupL, ddownL, chil, dupR, ddownR, chir
    lr = lr.transpose((0,3,1,4,2,5)) #dupL, dupR, ddownL, ddownR, chil, chir
    lr = lr.reshape((d2, d2, chil, chir))
    return lr



"""
2--lam--gam--3
        |
        1
        |
where lam is stored 1--lam--2
If lam is None this is a no-op.
lam can either be a vector of diagonal entries or a matrix.
This function also works if gam is a matrix.
"""
def leftmult(lam, gam):
    if lam is None:
        return gam
    ngam = len(gam.shape)
    nlam = len(lam.shape)
    if nlam==1:
        return lam[:, None]*gam
    if nlam==2:
        if ngam==2:
            return np.dot(lam, gam) #lambda is a matrix, note this assumes lam[2] hits gam[2]
        if ngam==3:
            idx = ([-2, 1], [-1, 1, -3])
            return scon([lam, gam], idx)

            #return np.einsum('bi, aic', lam, gam)
    raise IndexError("Invalid shapes. Gamma: ", gam.shape, "Lambda: ", lam.shape)

"""
2--gam--lam--3
   |
   1
   |
where lam is stored 1--lam--2
If lam is None this is a no-op.
lam can either be a vector of diagonal entries or a matrix.
This function also works if gam is a matrix.
"""
def rightmult(gam, lam):
    if lam is None:
        return gam
    nlam = len(lam.shape)
    if nlam==1:
        return lam*gam
    if nlam==2:
        return np.dot(gam, lam)
    raise IndexError("Invalid shapes. Gamma: ", gam.shape, "Lambda: ", lam.shape)

def gauge_transform(gl, A, gr):
    """
            |
            1
     2--gl--A--gr--3
    """
    return rightmult(leftmult(gl, A), gr)

################################################################################
#Chain contractors - MPS.
###############################################################################
# def XA(X, A):
    # """
    # 1 3
    # | |
    # X-A-5
    # | |
    # 2 4
    # """
    # return np.dot(X, A)

# def AX(A, X):
    # """
    # 3 1
    # | |
  # 5-A-X
    # | |
    # 4 2
    # """
    # A = np.transpose((A, (0,1,3,2)))
    # return np.dot(X, A)


"""
  |---B--1
  |   |     
  X---A--2     
  |   |              
  |---B*-3
"""
def XopLmixed(A, B, X):
    out = scon([B, A, np.conj(B), X],
                  [[1,3,-1],
                  [1,2,4,-2],
                  [2,5,-3],
                  [3,4,5]]
                 )
    raise NotImplementedError()
    return out

"""
  1---B--|
      |  |   
  2---A--X     
      |  |            
  3---B*-|
"""
def XopRmixed(A, B, X):
    out = scon([B, A, np.conj(B), X],
                  [
                    [1, -1, 3],
                    [1, 2, -2, 4],
                    [2, -3, 5],
                    [3, 4, 5]
                  ]
                 )
    raise NotImplementedError()
    return out

    

"""
  |---B1----B2----...-BN*-1
  |   |     |         |
  X---A1----A2----...-A3--2     
  |   |     |         |
  |---B1*---B2*---...-BN*-3
  (note: AN and BN will be contracted into a single matrix)
"""
def leftchainmixed(As, Bs, X):
    raise NotImplementedError()
    for A, B in zip(As, Bs):
        X = XopLmixed(A, B, X)
    return X

def rightchainmixed(As, Bs, X):
    raise NotImplementedError()
    for A, B in zip(As[::-1], Bs[::-1]):
        X = XopRmixed(A, B, X)
    return X

"""
  |---A1---A2---...-AN-2
  |   |    |         |
  X   |    |   ...   |
  |   |    |         |
  |---B1---B2---...-BN-1
  (note: AN and BN will be contracted into a single matrix)
"""
def leftchain(As, Bs=None, X=None):
    if Bs is None:
        Bs = list(map(np.conj, As))
    for A, B in zip(As, Bs):
        X = XopL(A, B=B, X=X)
    return X

"""
  2---A1---A2---...-AN--
      |    |         | |
      |    |   ...   | X
      |    |         | |
  1---B1---B2---...-BN--
  (note: AN and BN will be contracted into a single matrix)
"""
def rightchain(As, Bs=None, X=None):
    if Bs is None:
        Bs = list(map(np.conj, As))
    for A, B in zip(As[::-1], Bs[::-1]):
        X = XopR(A, B=B, X=X)
    return X

def leftchainop(As, Op=None, Bs=None, X=None):
    """
      |---A1---A2---...-AN--2
      |   |    |         |
      |   |____|__ ...  _|_                
      X  | Op      ...    |
      |  |________ ...  __|
      |   |    |         |
      |---B1---B2---...-BN--1
      (note: AN and BN will be contracted into a single matrix)
    """
    if Op is None:
        return leftchain(As, Bs=Bs, X=X)
    if Bs is None:
        Bs = map(np.conj, As)
    N = len(As)
    d = As[0].shape[0]
    O = Op.reshape((d**N, d**N)) 
    bigA = reduce(leftgather, As)
    bigB = reduce(leftgather, Bs)
    return XopL(bigA, B=bigB, O=O, X=X) 

def rightchainop(As, Op=None, Bs=None, X=None):
    """
      2---A1---A2---...-AN--|
          |    |         |  |
          |____|__ ...  _|_ |               
         | Op      ...    | X
         |________ ...  __| |
          |    |         |  |
      1---B1---B2---...-BN--|
      (note: AN and BN will be contracted into a single matrix)
    """
    if Op is None:
        return rightchain(As, Bs=Bs, X=X)
    if Bs is None:
        Bs = [np.conj(A) for A in As]
    N = len(As)
    d = As[0].shape[0]
    O = Op.reshape((d**N, d**N)) 
    bigA = reduce(leftgather, As[::-1])
    bigB = reduce(leftgather, Bs[::-1])
    return XopR(bigA, B=bigB, O=O, X=X) 



    # topchis = list(range(1, 4*N, 4))
    # topchis.append(-1)
    # botchis = list(range(2, 4*N, 4))
    # botchis.append(-2)
    # topds = list(range(3, 4*N, 4))
    # botds = list(range(4, 4*(N+1), 4))
    
    # if X is not None:
        # to_contract = [X,]
        # Xidx = (topchis[0], botchis[0])
        # idxs = [Xidx,]
    # else:
        # botchis[0] = topchis[0]
        # to_contract = []
        # idxs = []
    # Aidxs = [(td, tcl, tcr) for 
                # td, tcl, tcr in zip(topds, topchis[:-1], topchis[1:])]
    # Bidxs = [(bd, bcl, bcr) for 
                # bd, bcl, bcr in zip(botds, botchis[:-1], botchis[1:])]
    # # print [A.shape for A in As]
    # # print [B.shape for B in Bs]
    # # print Op.shape
    # # if X is not None:
        # # print X.shape
   # # Opidxs = [list(topds) + list(botds)]
    # Opidxs = [list(botds) + list(topds)]
    # #Opidxs = [list(botds[::-1]) + list(topds[::-1])]

    # to_contract += [Op,]+As+Bs
    # #to_contract += [Op,]+Bs+As
    # idxs += Opidxs + Aidxs + Bidxs
    # #print "Idxs: ", idxs
    # out  = scon(to_contract, idxs)
    # return out
# def leftchainop(As, Op=None, Bs=None, X=None):
    # if Op is None:
        # return leftchain(As, Bs=Bs, X=X)
    # if Bs is None:
        # Bs = list(map(np.conj, As))
    # N = len(As)
    # topchis = list(range(1, 4*N, 4))
    # topchis.append(-1)
    # botchis = list(range(2, 4*N, 4))
    # botchis.append(-2)
    # topds = list(range(3, 4*N, 4))
    # botds = list(range(4, 4*(N+1), 4))
    
    # if X is not None:
        # to_contract = [X,]
        # Xidx = (topchis[0], botchis[0])
        # idxs = [Xidx,]
    # else:
        # botchis[0] = topchis[0]
        # to_contract = []
        # idxs = []
    # Aidxs = [(td, tcl, tcr) for 
                # td, tcl, tcr in zip(topds, topchis[:-1], topchis[1:])]
    # Bidxs = [(bd, bcl, bcr) for 
                # bd, bcl, bcr in zip(botds, botchis[:-1], botchis[1:])]
    # # print [A.shape for A in As]
    # # print [B.shape for B in Bs]
    # # print Op.shape
    # # if X is not None:
        # # print X.shape
   # # Opidxs = [list(topds) + list(botds)]
    # Opidxs = [list(botds) + list(topds)]
    # #Opidxs = [list(botds[::-1]) + list(topds[::-1])]

    # to_contract += [Op,]+As+Bs
    # #to_contract += [Op,]+Bs+As
    # idxs += Opidxs + Aidxs + Bidxs
    # #print "Idxs: ", idxs
    # out  = scon(to_contract, idxs)
    # return out
    

"""
  |---A1---A2---...-AN---
  |   |    |        |   |
  X   |    |   ...  |   Y
  |   |    |        |   |
  |---B1---B2---...-BN---
"""
def chainnorm(As, Bs=None, X=None, Y=None):
    X = leftchain(As, Bs=Bs, X=X)
    if Y is not None:
        X = np.dot(X, Y.T)
    return np.trace(X)



def proj(X, Y=None):
    """
    2   2
    |---|
    |   |
    X   Y
    |   |
    |---|
    1   1
    Be careful that Y especially is in fact stored this way!
    """
    if Y is not None:
        X = np.dot(X, Y.T)
    return np.trace(X)


def chainloop(N, X, Y, ops, chainopfunc, chainnormfunc):
    """
    Low-level function called by e.g. chainwithops.
    """
    here = 0
    newops = [(op, site) for op, site in ops if op is not None]
    for op, site in newops:
        if site < 0:
            raise IndexError("Invalid site: ", site)
        oplength = len(op.shape)//2 #sites this operator acts on
        if site+oplength > N:
            raise IndexError("Sites, oplength ", site, oplength, 
                             "inconsistent with chain of length", N)
        X = chainopfunc(here, site, X, None)
        X = chainopfunc(site, site+oplength, X, op)
        here = site+oplength
    return chainnormfunc(here, X, Y)

def chainwithops(As, Bs=None, ops=[(None, None)], lvec=None, rvec=None):
    """
    Given a list of ket and, optionally, bra tensors, along with a list 
    of operators and sites they act upon, compute <Bs|O1\otimesO2...|As>.
    As/Bs= [np.array1(chi1, d, chi2), np.array2(chi2, d, chi1) ...] 
        -> local tensors of the ket/bra with all Schmidt vectors contracted in.
           If Bs is None it will be np.conj(As) and this function will compute
           an expectation value.
    lvec=np.array(chi1, chi1), rvec=np.array(chiN, chiN) 
        -> left and right operators connecting bond indices of As to Bs.
           Identity if None. These could be the transfer-matrix-eigenvectors,
           but we usually contract those into As before calling this 
           in practice.
    ops=[(np.array(Op1), Int(site1)), (np.array(Op2), Int(site2)), ...] -> 
        list of tuples, each containing an operator and a site number. The
        operator dimensions can be arbitrary multiples of d; if larger than
        d they will be interpreted as acting upon the necessary number of
        sites starting from the specified site number. 
        If the list has more than one entry this computes a correlator.  
        If the operators overlap for whatever reason they will be 
        sequentially applied in their listed order.
    """
    if Bs is None:
        Bs = list(map(np.conj, As))
    def leftchainopf(begin, end, X, Op):
        return leftchainop(As[begin:end], Bs=Bs[begin:end], Op=Op, X=X)
    def leftchainnormf(here, X, Y):
        return chainnorm(As[here:], Bs[here:], X=X, Y=Y)
    answer = chainloop(len(As), lvec, rvec, ops, leftchainopf, leftchainnormf)
    return answer

def mpochainwithops(As, ops=[(None, None)], lvec=None, rvec=None):
    def leftchainopf(begin, end, X, Op):
        return mpoleftchainop(As[begin:end], Op=Op, X=X)
    def leftchainnormf(here, X, Y):
        return mpochainnorm(As[here:], X=X, Y=Y)
    answer = chainloop(len(As), lvec, rvec, ops, leftchainopf, leftchainnormf) 
    return answer

################################################################################
#Chain contractors - MPO.
###############################################################################
"""
       /|   /|       /|   /|
1--X--|A1--|A2--..--|A3--|A4--2 
       \|   \|       \|   \|
    _________..___________
    |        Op          |
    |____________________|
"""
def mpoleftchainop(As, Op=None, X=None):
    if Op is None:
        return mpoleftchain(As, X=X)
    As[0] = mpoleftmult(As[0], X=X)
    N = len(As)
    botds = list(range(1, 3*N, 3))
    topds = list(range(2, 3*N, 3))
    chis = [-1,] + list(range(3, 3*N, 3)) + [-2,]
    to_contract = [Op,]+As
    opidxs = [list(topds) + list(botds)]

    Aidxs = list(zip(botds, topds, chis[:-1], chis[1:]))
    idxs = opidxs + Aidxs
    newX = scon(to_contract, idxs)
    #newX = newX.reshape((newX.shape[-1]))
    return newX


"""
        /|   /|       /|   /|
1-- X--|A1--|A2--..--|A3--|A4--2 
        \|   \|       \|   \|
"""
def mpoleftchain(As, X=None):
    for A in As:
        X = mpooptrace(A, X=X)
    return X

"""
      /|   /|       /|   /|
--X--|A1--|A2--..--|A3--|A4--Y-- 
      \|   \|       \|   \|
"""
def mpochainnorm(As, X=None, Y=None):
    X = mpoleftchain(As, X=X)
    if Y is not None:
        X = np.dot(X, Y)
    return np.trace(X)


"""   1
      |   
3--X--A--4
      |   
      2
"""
def mpoleftmult(A, X=None):
    if X is None:
        return A
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=0)
    XA = np.einsum("ij, abjk", X, A)
    return XA

"""
   1
   |   
3--A--X--4
   |   
   2
"""
def mporightmult(A, X=None):
    if X is None:
        return A
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=1)
    AX = np.einsum("abij, jk", A, X)
    return AX

"""
   1
   |
   X
   |   
3--A--4
   |   
   2
"""
def mpotopmult(A, X=None):
    if X is None:
        return A
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=0)
    XA = np.einsum("ai,ijkl", X, A)
    return XA

"""
   1
   |   
3--A--4
   |   
   X
   |
   2
"""
def mpobotmult(A, X=None):
    if X is None:
        return A
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=1)
    AX = np.einsum("abde, bc", A, X)
    return AX

"""
  /|
1-|A--2
  \|
"""
def mpotrace(A):
    return np.einsum("aacd", A).reshape((A.shape[2], A.shape[3]))


"""    Op          
       /|   
1--X--| A--2
       \|   
MPO equivalent of XopL
"""
def mpooptrace(A, Op=None, X=None):
    d, d2, chiL, chiR = A.shape
    if d != d2:
        raise ValueError("Inconsistent physical dimensions in MPO ", A.shape)
    A = mpoleftmult(A, X=X)
    A = mpotopmult(A, X=Op)
    A = mpotrace(A)
    return A



def mpogate(A, U=None):
    """  
         |
         U 
         |
       --A--
         |
        Udag
         |
    """
    if U is None:
        return A
    dA1, dA2, chiL, chiR = A.shape
    d = dA1
    if dA1!=dA2:
        raise ValueError("A had inconsistent physical dimension ", A.shape)
    if np.squeeze(U).shape != (dA1, dA2):
        raise ValueError("squeeze(U) had bad shape; unsqueezed shape:",U.shape)
    A = mpotop(A, U).reshape((d,d,chiL*chiR))
    UdagT = np.conj(U)
    UAUdag = np.dot(UdagT, A).reshape((d, d, chiL, chiR)) 
    return UAUdag

def mpotop(A, U):
    """  
         |
         U 
         |
       --A--
         |
         |
    """
    dA1, dA2, chiL, chiR = A.shape
    d = dA1
    if dA1!=dA2:
        raise ValueError("A had inconsistent physical dimension ", A.shape)
    if np.squeeze(U).shape != (dA1, dA2):
        raise ValueError("squeeze(U) had bad shape; unsqueezed shape:",U.shape)
    Amat = A.reshape((d, d*chiL*chiR))
    UA = np.dot(U, Amat).reshape((d, d, chiL, chiR))
    return UA

#******************************************************************************
#Single site to open legs.

"""
  |---A---2
  |   | 
  X   O  
  |   |
  |---B---1

"""
def XopL(A, O=None, X=None, B=None):
    if B is None:
        B = np.conj(A)
    A = leftmult(X, A)
    if O is not None:
        B = Bop(O, B)
    #return np.einsum("jia, jib", A, B)
    #return np.einsum("ija, ijb", A, B)
    idx = [(2, 1, -2),
            (2, 1, -1)]
    return scon([A, B], idx) #the scon call is much faster

"""
  2---A---|
      |   |
      O   X
      |   |
  1---B---|
"""
def XopR(A, O=None, X=None, B=None):
    if B is None:
        B = np.conj(A)
    B = rightmult(B, X)
    if O is not None:
        B = Bop(O, B)
    idx = [(2, -2, 1),
           (2, -1, 1)]
    return scon([A, B], idx)

    #return np.einsum("iaj, ibj", A, B)




#****************************************************************************
#TWO SITE OPERATORS
#****************************************************************************

def rholoc(A1, A2, B1=None, B2=None):
    """
    -----A1-----A2-----
    |    |(3)   |(4)   |
    |                  |
    |                  |
    |    |(1)   |(2)   |
    -----B1-----B2------
    returned as a (1:2)x(3:4) matrix.
    Assuming the appropriate Schmidt vectors have been contracted into the As,
    np.trace(np.dot(op, rholoc.T)) is the expectation value of the two-site
    operator op coupling A1 to A2.
    """
    if B1 is None:
        B1 = np.conj(A1)
    if B2 is None:
        B2 = np.conj(A2)
    d = A1.shape[0]
    to_contract = [A1, A2, B1, B2]
    idxs = [ (-3, 1, 2),
             (-4, 2, 3),
             (-1, 1, 4),
             (-2, 4, 3) ]
    rholoc = scon(to_contract, idxs).reshape((d**2, d**2))
    return rholoc

# def DMRG_superblock(left, right, hL, h, hR):
    # theta = twositecontract(left, right, h)
    # theta = thetaleft(theta, hL)
    # theta = thetaright(theta, hR)
    # return theta


"""
   2--left-right--4
        |__|    
        |U |  
        ----
        |  |
        1  3 
"""
def twositecontract(left, right, U=None):
    if U is None:
        return np.dot(left, right)
    else:
        to_contract = (left, right, U)
        idxs = [ (2, -2, 1),
                 (3, 1, -4),
                 (-1, -3, 2, 3)]
        return scon(to_contract, idxs)


"""
      2--th--4
        |__|    
        |U |  
        ----
        |  |
        1  3 
"""
def twositecontracttheta(theta, U):
    to_contract = (theta, U)
    idxs = [ (1, -2, 3, -4),
             (-1, 1, -3, 3) ]
    return scon(to_contract, idxs)
"""
   --theta--4
   |__|  |
   |hL|  3
   ----
   |  |
   2  1 
"""
def thetaleft(theta, hL):
    to_contract = (theta, hL)
    idxs = [ (2, 1, -3, -4),
             (-2, -1, 1, 2) ]
    try:
        ans = scon(to_contract, idxs)
    except ValueError:
        errstr = "Theta shape: " + str(theta.shape) + "\n"
        errstr += "hL shape: " + str(hL.shape) + "\n"
        raise ValueError(errstr)
    return ans

"""
2--theta
   | |__|  
   1 |hR|  
     ----
     |  |
     3  4 
"""
def thetaright(theta, hR):
    to_contract = (theta, hR)
    idxs = [ (-1, -2, 1, 2),
             (-3, -4, 1, 2) ]
    return scon(to_contract, idxs)

"""    
 A--3  4  
|_|    |
|H|    |   
---    |
| |    |
 A*--1 2
"""
def DMRG_hL1(A, Adag, hL):
    d = A.shape[0]
    I = np.eye(d, dtype=A.dtype)
    to_contract = (A, Adag, hL, I)
    A_idxs = (4, 3, -3)
    Adag_idxs = (2, 1, -1)
    h_idxs = (1, 2, 3, 4)
    I_idxs = (-2, -4)
    answer = scon(to_contract, (A_idxs, Adag_idxs, h_idxs, I_idxs))
    return answer


"""    
--A--3 4  
| |____|    
| |_H__|   
| |    |
--A*-1 2
"""
def DMRG_hL2(A, Adag, h):
    to_contract = (A, Adag, h)
    A_idxs = (2, 1, -3)
    Adag_idxs = (3, 1, -1)
    h_idxs = (3, -2, 2, -4)
    answer = scon(to_contract, (A_idxs, Adag_idxs, h_idxs))
    return answer


def DMRG_hL(A, hL, h, Adag=None):
    if Adag is None:
        Adag = np.conj(A)
    term1 = DMRG_hL1(A, Adag, hL)
    term2 = DMRG_hL2(A, Adag, h)
    answer = term1 + term2
    return answer


"""    
  3 4-B-----  
  |   |____|    
  |   |_H__|  
  |   |    |
  1 2-B*---|
"""
def DMRG_hR1(B, Bdag, hR):
    d = B.shape[0]
    I = np.eye(d, dtype=B.dtype)
    to_contract = (B, Bdag, hR, I)
    B_idxs = (4, -4, 2)
    Bdag_idxs = (3, -2, 1)
    h_idxs = (3, 1, 4, 2)
    I_idxs = (-1, -3)
    answer = scon(to_contract, (B_idxs, Bdag_idxs, h_idxs, I_idxs))
    return answer

"""    
  3   4-B-  
  |____| |   
  |_H__| | 
  |    | |
  1   2-B*
"""
def DMRG_hR2(B, Bdag, h):
    to_contract = (B, Bdag, h)
    B_idxs = (3, -4, 1)
    Bdag_idxs = (2, -2, 1)
    h_idxs = (-1, 2, -3, 3)
    answer = scon(to_contract, (B_idxs, Bdag_idxs, h_idxs))
    return answer
    

def DMRG_hR(B, hR, h, Bdag=None):
    if Bdag is None:
        Bdag = np.conj(B)
    term1 = DMRG_hR1(B, Bdag, hR)
    term2 = DMRG_hR2(B, Bdag, h)
    answer = term1 + term2
    return answer


    # LR = np.dot(left, right)
    # if U is None:
        # return LR
    # d, chiL, d, chiR = LR.shape
    # U.shape = (d,d,d**2)
    # #U = U.reshape((d,d,d**2))
    # LR = LR.transpose((1,0,2,3))
    # #LR.shape = (chiL, d**2, chiR)
    # LR = LR.reshape((chiL, d**2, chiR))
    # ans = np.dot(U, LR).reshape((d, d, chiL, chiR))
    # ans = ans.transpose((0,2,1,3))
    # return ans

    # if U is None:
    # else:
        # ans = np.einsum('jbi, kid, jkac', left, right, U) 
    #return ans



def H_times_psi(H, left, right, HL=None, HR=None):
    """
    Serves as matvec for the DMRG eigensolver.
    """
    d = H.shape[0]
    Hmat = H.reshape((d*d, d*d))
    Hnorm = npla.norm(Hmat)
    Hshift = (Hmat - Hnorm*np.eye(d*d)).reshape((d,d,d,d))
    if HL is not None:
        left = leftmult(HL, left)
    if HR is not None:
        right = rightmult(right, HR)
    answer = twositecontract(left, right, U=Hshift)
    return answer
    

def DMRGeigs(left, right, H, HL=None, HR=None):
    """
    Eigensolver for DMRG.
    """
    Hshift = Hbig - Hnorm * np.eye(d1*chi1*d2*chi2)
    Es, Vs = sp.sparse.linalg.eigs(Hshift, v0=thvec, k=1)
    print(Es)
    E0 = Es[0] + Hnorm
    thnew = Vs[:, 0]
    thnew = thnew / npla.norm(thnew)
    thnew = thnew.reshape((d1, chi1, d2, chi2))
    
    return (E0, thnew)

def svd(th, minlam=1E-13, maxchi=None, normalize=True, fixed=False):
    """Computes the singular value decomposition of the input matrix
       'th', which should be the (chi*d x chi*d) result of a contraction
       between two nodes (and, possibly, a two-site operator).
    """

    d1, chi1, d2, chi2 = th.shape
    newshp = (d1*chi1, d2*chi2)
    #th.shape = newshp
    # thview = th.view()
    # thview.shape = newshp 
    th = th.reshape(newshp) #note - this does a copy and should be optimized
    try:
        U, s, V = npla.svd(th, full_matrices=False, compute_uv=True)
    except:
        print("Divide-and-conquer SVD failed. Trying gesvd...")
        U, s, V = spla.svd(th, full_matrices=False, overwrite_a=False,
                check_finite=True, lapack_driver='gesvd')
    #print(s)
    if maxchi is None or maxchi > s.size:
        maxchi = s.size
    truncidx = maxchi
    S = s[:truncidx]

    if minlam is not None and S[-1] < minlam:
        toosmallidx = maxchi - np.searchsorted(S[::-1], minlam)
        if toosmallidx==0:
            print("Warning: very small singular values: ", s)
            truncidx = 1
            #raise ValueError("Could not truncate because there were no singular "
            #      "values greater than the minimum ", minlam, ". s was: ", s)
        if fixed:
            S[toosmallidx:] = minlam
        else:
            truncidx = toosmallidx
            S = S[:truncidx]
    X = U[:, :truncidx]
    X = X.reshape((d1, chi1, truncidx))
    Y = V[:truncidx, :]
    Y = Y.reshape((truncidx, d2, chi2))
    Y = Y.transpose((1,0,2))
    truncerror = npla.norm(s[truncidx:])
    #truncerror = np.sqrt(np.sum(s2[truncidx:]))
    #print(S)
    if normalize:
        S /= npla.norm(S)
    #print(S)
    #print("**")
    #print(S[:-1])
    return X, Y, S, truncidx, truncerror

def truncateandreshape(U, s, V, truncidx, d, chi1, chi2):
    """
    U, s, V are the output of an SVD. s will be truncated to 'truncidx', 
    and U and V will be similarly truncated so that UsV remains well-defined.
    U and Y will then be reshaped into gamma tensors; this involves transposing
    Y.

    This function turns out to be surprisingly expensive for some reason.
    """
    X = U[:, :truncidx]
    X = X.reshape((d, chi1, truncidx))
    #V = np.conj(V)
    Y = V[:truncidx, :]
    Y = Y.reshape((truncidx, d, chi2))
    Y = Y.transpose((1,0,2))
    #S = s[:truncidx]
    truncerror = npla.norm(s[truncidx:])
    # if truncerror > 1E-8:
        # print("Warning - large truncation error: ", truncerror)
    
    return X, s[:truncidx], Y, truncerror


def newsvd(th, err=1E-12, minlam=1E-13, maxchi=None, normalize=True):
    """Computes the singular value decomposition of the input matrix
       'th', which should be the (chi*d x chi*d) result of a contraction
       between two nodes (and, possibly, a two-site operator).
    """

    d1, chi1, d2, chi2 = th.shape
    newshp = (d1*chi1, d2*chi2)
    th = th.reshape(newshp)
    if maxchi is None:
        maxchi = max(newshp)
    US = svd.fit_transform(th)
    S = svd.singular_values_
    V = svd.components_

    #We still need to truncate any poorly conditioned values 
    toosmall = S<minlam
    if toosmall[0] == True:
        raise ValueError("No nonzero singular values!")
    
    truncidx = np.argmax(toosmall)
    if truncidx == 0:
        truncidx = len(S)
    #reshape the dimension which is shared with S
    XS = US[:, :truncidx]#.reshape((d1, chi1, truncidx))
    s = S[:truncidx]
    X = XS*(1./s)#.reshape((d1,chi1,truncidx))
    X = X.reshape((d1,chi1,truncidx))
    Y = V[:truncidx, :].reshape((truncidx, d2, chi2))
    Y = np.transpose(Y, (1,0,2))
    error = 1.0-np.sum(svd.explained_variance_ratio_)
    if error > err:
        print("Warning - large error: ", error)
    if normalize:
        s /= npla.norm(s)
    return X, Y, s, truncidx

def makechis(d, N, maxchi):
    """Create the vector of chis for the chain.
       This is a length-N+1 list of exponents of d. The exponents are of 
       the form
       [0, 1, 2, 1, 0] (if N+1 is odd)
       [0, 1, 2, 2, 1, 0] (if N+1 is even)
       Any numbers in the above exceeding ln(maxchi) / ln(d) are replaced
       with maxchi.
    """
    last = N
    maxexp = int(np.log(maxchi) // np.log(d))
    exp = list(range(0, (last+1)//2))
    reverse = exp[::-1]
    if last % 2 != 1:
      exp = exp + [last//2]
    exp = exp + reverse
    
    for i in range(0, len(exp)):
        if exp[i] > maxexp:
          exp[i] = maxexp
    chis = np.power(d, exp, dtype=int)
    return chis

def randommps(d, n, maxchi, minlam=1E-13):
    chis = makechis(d, n, maxchi)
    lams = []
    for i in range(len(chis)):
        lamshape = (chis[i])
        thislam = utils.random_rng(lamshape, 0., 1.)
        thislam /= npla.norm(thislam)
        lams.append(thislam)
    
    gams = []
    for i in range(len(chis)-1):
        gamshape = (d, chis[i], chis[i+1])
        thisgam = utils.random_complex(gamshape)
        gams.append(thisgam)
    return lams, gams
