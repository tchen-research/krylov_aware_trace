import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import clear_output

class get_hamiltonian():

    def __init__(self,Jx,Jy,Jz,s):
        self.N = len(Jz)
        self.s = s
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.dtype = np.float128

        self.M = int(2*self.s+1)
        self.Sx = np.zeros((self.M,self.M),dtype='complex')
        self.Sy = np.zeros((self.M,self.M),dtype='complex')
        self.Sz = np.zeros((self.M,self.M),dtype='complex')
        for i in range(self.M):
            for j in range(self.M):
                self.Sx[i,j] = ((i==j+1)+(i+1==j))*np.sqrt(s*(s+1)-(s-i)*(s-j))/2
                self.Sy[i,j] = ((i+1==j)-(i==j+1))*np.sqrt(s*(s+1)-(s-i)*(s-j))/2j
                self.Sz[i,j] = (i==j)*(s-i)

    def __matmul__(self,v):
                
        if v.ndim == 2:
            m,n = v.shape
        else:
            m = len(v)
            n = 1 
    
        out = np.zeros((m,n),dtype='complex')

        for j in range(self.N):
            if  np.count_nonzero(self.Jx[:,j]) != 0 or np.count_nonzero(self.Jy[:,j]) != 0:
                I1 = self.M**j
                I2 = self.M**(self.N-j-1)
                Sxj_v = ((self.Sx@v.T.reshape(n,I1,-1,I2))).reshape(n,-1).T
                Syj_v = ((self.Sy@v.T.reshape(n,I1,-1,I2))).reshape(n,-1).T
                Szj_v = ((self.Sz@v.T.reshape(n,I1,-1,I2))).reshape(n,-1).T

                # symmetry
                for i in range(j):
                    if self.Jx[i,j] != 0 or self.Jy[i,j] != 0:
                        I1 = self.M**i
                        I2 = self.M**(self.N-i-1)
                        Sxi_Sxj_v = ((self.Sx@Sxj_v.T.reshape(n,I1,-1,I2))).reshape(n,-1).T
                        Syi_Syj_v = ((self.Sy@Syj_v.T.reshape(n,I1,-1,I2))).reshape(n,-1).T

                        out += (2-(i==j))*self.Jx[i,j] * Sxi_Sxj_v
                        out += (2-(i==j))*self.Jy[i,j] * Syi_Syj_v 

                out += self.Jz[j] * Szj_v
            
        return out.flatten() if n==1 else out
    
    def tosparse(self):
                
        out = sp.sparse.coo_matrix((self.M**self.N,self.M**self.N),dtype='complex')

        for j in range(self.N):
            if  np.count_nonzero(self.Jx[:,j]) != 0 or np.count_nonzero(self.Jy[:,j]) != 0:
                I1 = sp.sparse.eye(self.M**j,dtype='complex')
                I2 = sp.sparse.eye(self.M**(self.N-j-1),dtype='complex')
                Sxj = sp.sparse.kron(sp.sparse.kron(I1,self.Sx),I2)
                Syj = sp.sparse.kron(sp.sparse.kron(I1,self.Sy),I2)
                Szj = sp.sparse.kron(sp.sparse.kron(I1,self.Sz),I2)

                for i in range(j):
                    if self.Jx[i,j] != 0 or self.Jy[i,j] != 0:
                        I1 = sp.sparse.eye(self.M**i,dtype='complex')
                        I2 = sp.sparse.eye(self.M**(self.N-i-1),dtype='complex')
                        Sxi_Sxj = sp.sparse.kron(sp.sparse.kron(I1,self.Sx),I2)@Sxj
                        Syi_Syj = sp.sparse.kron(sp.sparse.kron(I1,self.Sy),I2)@Syj
                        #Szi_Szj = sp.sparse.kron(sp.sparse.kron(I1,self.Sz),I2)@Szj

                        out += (2-(i==j))*self.Jx[i,j] * Sxi_Sxj
                        out += (2-(i==j))*self.Jy[i,j] * Syi_Syj

                out += self.Jz[j] * Szj
            
        return out