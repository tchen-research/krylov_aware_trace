import numpy as np
import scipy as sp
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
