import numpy as np
import scipy as sp
from scipy.stats import chi2
import matplotlib.pyplot as plt

def block_lanczos(A,Z0,q,reorth=0):
        
    Z = np.copy(Z0)
    d,b = Z.shape
    
    M = [ np.zeros((b,b),dtype=A.dtype) ]*q
    R = [ np.zeros((b,b),dtype=A.dtype) ]*(q+1)
    
    Q = np.zeros((d,b*(q+1)),dtype=A.dtype)

    Q[:,0:b],R[0] = np.linalg.qr(Z)
    for k in range(0,q):
        
        # New set of vectors
        Qk = Q[:,k*b:(k+1)*b]
        Qkm1 = Q[:,(k-1)*b:k*b]
        Z = A@Qk - Qkm1@(R[k].conj().T) if k>0 else A@Qk
        
        # Tridiagonal block M
        M[k] = Qk.conj().T@Z
        Z -= Qk@M[k]

        # reorthogonalization
        if reorth>k:
            Z -= Q[:,:k*b]@(Q[:,:k*b].conj().T@Z)
        
        # Pivoted QR
        Z_,R_,p = sp.linalg.qr(Z,pivoting=True,mode='economic')
        R[k+1] = R_[:,np.argsort(p)]
        
        # Orthogonalize again if R is rank deficient
        if reorth>k:
            r = np.abs(np.diag(R[k+1]))
            r_idx = np.nonzero(r<np.max(r)*1e-10)[0]
            Z_[:,r_idx] -= Qk@(Qk.conj().T@Z_[:,r_idx])
            Z_[:,r_idx],R_ = np.linalg.qr(Z_[:,r_idx])
            Z_[:,r_idx] *= np.sign(np.diag(R_))
            
        Q[:,(k+1)*b:(k+2)*b] = Z_

    return Q,M,R


def par_lanczos(A,Z0,q,reorth=0):
        
    Z = np.copy(Z0)
    d,b = Z.shape
    
    M = np.zeros((q,b),dtype=A.dtype)
    R = np.zeros((q+1,b),dtype=A.dtype)
    
    Q = np.zeros((d,q+1,b),dtype=A.dtype)

    for j in range(b):
        R[0,j] = np.linalg.norm(Z[:,j])
        Q[:,0,j] = Z[:,j] / R[0,j]

    for k in range(0,q):
        
        AQk = A@Q[:,k]
        for j in range(b):
            Qk = Q[:,k,j]
            Qkm1 = Q[:,k-1,j]
            Z = AQk[:,j] - Qkm1*(R[k,j]) if k>0 else AQk[:,j]

            M[k,j] = Qk.conj().T@Z
            Z -= Qk*M[k,j]

            if reorth>k:
                Z -= Q[:,:k,j]@(Q[:,:k,j].conj().T@Z)
                Z -= Q[:,:k,j]@(Q[:,:k,j].conj().T@Z)
                
            R[k+1,j] = np.linalg.norm(Z)
            Q[:,k+1,j] = Z / R[k+1,j]

    return Q,M,R


def get_block_tridiag(M,R):

    q = len(M)
    b = len(M[0])
    
    T = np.zeros((q*b,q*b),dtype=M[0].dtype)

    for k in range(q):
        T[k*b:(k+1)*b,k*b:(k+1)*b] = M[k]

    for k in range(q-1):
        T[(k+1)*b:(k+2)*b,k*b:(k+1)*b] = R[k]
        T[k*b:(k+1)*b,(k+1)*b:(k+2)*b] = R[k].conj().T
        
    return T


def krylov_trace_quadrature(A,b,q,n1,m,n2):

    d = A.shape[0]
   
    Ω = np.random.randn(d,b)
    Ψ = np.random.randn(d,m)

    Θ_defl = np.array([])
    W_defl = np.array([])
    Q = np.zeros((d,0))
    if b>0:
        QQ,M,R = block_lanczos(A,Ω,q+n1,reorth=q)
        Q = QQ[:,:(q+1)*b]

        T = get_block_tridiag(M,R[1:-1])
        Θ,S = np.linalg.eigh(T)
        Sqb = S.conj().T[:,:(q+1)*b]

        Θ_defl = np.copy(Θ)
        W_defl = np.linalg.norm(Sqb,axis=1)**2
    
    Θ_rem1 = np.array([])
    W_rem1 = np.array([])
    W_rem2 = 0
    Y = Ψ - Q@(Q.conj().T@Ψ)

    Qt,Mt,Rt = par_lanczos(A,Y,n2,reorth=0)
    for i in range(m):

        try:
            Θt,St = sp.linalg.eigh_tridiagonal(Mt[:,i],Rt[1:-1,i])
        except:
            Tt = np.diag(Mt[:,i]) + np.diag(Rt[1:-1,i],-1) + np.diag(Rt[1:-1,i],1)
            Θt,St = np.linalg.eigh(Tt)

        Sm2Rt = St.conj().T[:,0]*Rt[0,i]

        Θ_rem1 = np.hstack([Θ_rem1,Θt])
        W_rem1 = np.hstack([W_rem1,Sm2Rt**2/m])
     
    return np.hstack([Θ_defl,Θ_rem1]),np.hstack([W_defl+W_rem2,W_rem1])


def krylov_trace_restart_quadrature(A,g,r,b,q,n1,m,n2):
    
    assert b>0, 'b must be positive'
    
    d = A.shape[0]
   
    Ω = np.random.randn(d,b)
    Ψ = np.random.randn(d,m)

    λmin = np.inf
    λmax = -np.inf
    for i in range(r):
        QQ,M,R = block_lanczos(A,Ω,q,reorth=q)
        
        T = get_block_tridiag(M+[np.zeros((b,b))],R[1:])
        Θ,S = np.linalg.eigh(T)

        # lazy approach: find good least squares polynomial on estimated spectrum interval
        λmin = np.min(np.append(Θ,λmin))
        λmax = np.max(np.append(Θ,λmax))

        tt = np.polynomial.chebyshev.chebpts1(1000)
        xx = tt*1.01*(λmax-λmin)/2 + (λmax+λmin)/2
        yy = g(xx)
        p = np.polynomial.Chebyshev.fit(xx,yy,q)

        Y = S@np.diag(p(Θ))@S.conj().T[:,:b]
        Ω = QQ[:,:(q+1)*b]@Y@R[0]
    
    Θ_defl = np.array([])
    W_defl = np.array([])

    QQ,M,R = block_lanczos(A,Ω,q+n1,reorth=q)
    T = get_block_tridiag(M[:q+1],R[1:q+1])
    Θ,S = np.linalg.eigh(T)
    Q = QQ[:,:(q+1)*b]

    T = get_block_tridiag(M,R[1:-1])
    Θ,S = np.linalg.eigh(T)
    Sqb = S.conj().T[:,:(q+1)*b]

    Θ_defl = np.copy(Θ)
    W_defl = np.linalg.norm(Sqb,axis=1)**2

    
    Θ_rem1 = np.array([])
    W_rem1 = np.array([])
    W_rem2 = 0
    Y = Ψ - Q@(Q.conj().T@Ψ)

    Qt,Mt,Rt = par_lanczos(A,Y,n2,reorth=0)
    for i in range(m):

        try:
            Θt,St = sp.linalg.eigh_tridiagonal(Mt[:,i],Rt[1:-1,i])
        except:
            Tt = np.diag(Mt[:,i]) + np.diag(Rt[1:-1,i],-1) + np.diag(Rt[1:-1,i],1)
            Θt,St = np.linalg.eigh(Tt)

        Sm2Rt = St.conj().T[:,0]*Rt[0,i]

        Θ_rem1 = np.hstack([Θ_rem1,Θt])
        W_rem1 = np.hstack([W_rem1,Sm2Rt**2/m])
        
    return np.hstack([Θ_defl,Θ_rem1]),np.hstack([W_defl+W_rem2,W_rem1])




def mv_lanczos(A,ω,f,n):
    Q,M,R = par_lanczos(A,ω,n)
    
    Aω = np.zeros_like(ω)
    
    for i in range(ω.shape[1]):
        try:
            Θ,S = sp.linalg.eigh_tridiagonal(M[:,i],R[1:-1,i])
        except:
            T = np.diag(M[:,i]) + np.diag(R[1:-1,i],-1) + np.diag(R[1:-1,i],1)
            Θ,S = np.linalg.eigh(T)
            
        Aω[:,i] = Q[:,:-1,i]@(S@(f(Θ)*S[0,:].conj()))*R[0,i]
        
    return Aω

def ada_krylov_basis(A,Z0,f,n,epsilon,delta):
    # Carry out the variance reduction step of Adaptive Trace Estimation
        
    Z = np.copy(Z0)
    d,b = Z.shape
    
    M = []
    R = [np.zeros((b,b))]
    C = sampleC(epsilon,delta)
    costs = []

    Q,R[0] = np.linalg.qr(Z)
    k = 0
    while True:
        # New set of vectors
        Qk = Q[:,k*b:(k+1)*b]
        Qkm1 = Q[:,(k-1)*b:k*b]
        Z = A@Qk - Qkm1@(R[k].conj().T) if k>0 else A@Qk
        
        # Tridiagonal block M 
        M.append(Qk.conj().T@Z)
        Z -= Qk@M[k]
        
        # Reorthogonalization
        Z -= Q@(Q.conj().T@Z)
        
        # Pivoted QR
        Z, RR, p = sp.linalg.qr(Z,pivoting=True,mode='economic')
        R.append(RR[:,inv_perm(p)])
        
        # Orthogonalize again if R is rank deficient
        rankR = np.linalg.matrix_rank(RR, tol = abs(RR).max()*1e-10)
        if rankR < b:
            Z[:,rankR:] -= Q@(Q.conj().T@Z[:,rankR:])
            Z[:,rankR:] -= Z[:,:rankR]@(Z[:,:rankR].conj().T@Z[:,rankR:])
            Z[:,rankR:], _ = np.linalg.qr(Z[:,rankR:])
                    
        Q = np.hstack((Q,Z))
        
        k = k + 1
      
        if k >= n: # Test for cost minimization
            qb = (k-n+1)*b
            T = get_block_tridiag(M,R[1:])
            Θ,S = np.linalg.eigh(T)
            fS = f(Θ)*S[:qb,:] 

            fnorm2 = np.linalg.norm(fS[:,:qb],'fro')**2 + 2*np.linalg.norm(fS[:,qb:],'fro')**2
            costs.append(qb - n*C*fnorm2)
            
            if len(costs) >= 3:
                if (costs[-1] > costs[-2]) and (costs[-2] > costs[-3]):
                    break       
    #endWhile
    
    
    tDefl = np.trace(fS@S[:qb,:].conj().T)
    Q = Q[:,:qb]
    
    return tDefl, Q

def ada_hpp_basis(A,f,n,epsilon,delta): 
    d = A.shape[0]
    C = sampleC(epsilon,delta)
    Q = np.zeros((d,0))
    costs = []
    m = 0
    
    tDefl = 0
    
    while True:
        ω = np.random.randn(d,1)
        y = mv_lanczos(A,ω,f,n)
        y = y - Q@(Q.conj().T@y)
        q = y/np.linalg.norm(y)
        Q = np.hstack((Q,q))
        
        Aq = mv_lanczos(A,q,f,n)
        tk = np.vdot(q,Aq)
        tDefl += tk
        
        m += 2 + C*( 2*np.linalg.norm(Q.conj().T@Aq)**2 - (tk)**2 - 2*np.linalg.norm(Aq)**2 )
        costs.append(m)
                
        if len(costs) >= 3:
            if (costs[-1] > costs[-2]) and (costs[-2] > costs[-3]):
                break 
    #endWhile 
    
    return tDefl, Q

def ada_rem(A,Q,f,n,epsilon,delta):
    d = A.shape[0]
    
    tRem = 0
    tFro = 0
    k = 0
    m = np.Inf
    C = sampleC(epsilon,delta)
    
    while k < m:
        k = k + 1
        
        ψ = np.random.randn(d,1)
        y = ψ - Q@(Q.conj().T@ψ)
        
        Ay = mv_lanczos(A,y,f,n)
        
        tRem += np.vdot(y,Ay)
        tFro += np.linalg.norm(Ay)**2
        
        α = alphaFun(delta,k)
        m = C*tFro/(k*α)
        
    #endWhile
    
    return tRem/k, k
    
def ada_krylov(A,f,b,n,epsilon,delta):
    # Adaptive trace estimation algorithm
    
    d  = A.shape[0]
    Z0 = np.random.randn(d,b)
    
    tDefl, Q = ada_krylov_basis(A,Z0,f,n,epsilon,delta)
    tRem, m = ada_rem(A,Q,f,n,epsilon,delta)   
    
    tEst = tDefl + tRem
    costDefl = Q.shape[1] + (n-1)*b
    costRem  = n*m
        
    return tEst, costDefl, costRem
        
def ada_hpp(A,f,n,epsilon,delta):
    # Implementation of A-Hutch++
    
    tDefl, Q = ada_hpp_basis(A,f,n,epsilon,delta)
    tRem, m = ada_rem(A,Q,f,n,epsilon,delta)
    
    tEst = tDefl + tRem
    costDefl = 2*n*Q.shape[1]
    costRem  = n*m
    
    return tEst, costDefl, costRem
    
    
def alphaFun(delta,k):
    return chi2.isf(1-delta,df=k)/k

def sampleC(epsilon,delta):
    return 4*np.log(2/delta)/epsilon**2

def inv_perm(e):
    inve = np.copy(e)
    inve[e] = np.arange(len(e))
    return inve