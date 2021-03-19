import numpy as np
from scipy.linalg import eig
from pylsa.utils import *
from pylsa.transforms import *
from pylsa.dmsuite import *
from pylsa.decorators import *
import matplotlib.pyplot as plt

#-------------------------------------------------------------------
@io_decorator
def solve_rbc1d(Ny=100,Ra=1708,Pr=1,alpha=3.14,plot=True ):
    #----------------------- Parameters ---------------------------
    nu = np.sqrt(Pr/Ra)
    kappa = 1/np.sqrt(Ra*Pr)

    #----------------- diiscrete diff matrices  -------------------
    _,D1y = chebdif(Ny-1,1)  # chebyshev in y-direction
    y,D2y = chebdif(Ny-1,2)
    #Transform to y=[0,1]
    y,D1y,D2y = chebder_transform(y,D1y,D2y, zerotoone_transform)
    N, I= Ny, np.eye(Ny)
    #----------------------- mean flow -----------------------------
    # RBC FLOW
    U = U_y = 0.0*y
    T   = -1.0*y+1 ; T_y = D1y@T;
    # Derivatives
    UU, UU_y = np.diag(U), np.diag(U_y)
    _ , TT_y = np.diag(T), np.diag(T_y)
    #-------------------- construct matrix  ------------------------
    L2d = UU*1.j*alpha + nu*(alpha**2*I - D2y)
    K2d = UU*1.j*alpha + kappa*(alpha**2*I - D2y)

    #lhs
    L11 = 1*L2d        ; L12 = 0*UU_y  ; L13 = 1.j*alpha*I; L14 =  0*I
    L21 = 0*I          ; L22 = 1*L2d   ; L23 = 1*D1y      ; L24 = -1*I
    L31 = 1.j*alpha*I  ; L32 = 1*D1y   ; L33 = 0*I        ; L34 =  0*I
    L41 = 0*I          ; L42 = 1*TT_y  ; L43 = 0*I        ; L44 =  1*K2d
    
    #rhs
    M11 = 1*I          ; M12 = 0*I     ; M13 = 0*I        ; M14 = 0*I
    M21 = 0*I          ; M22 = 1*I     ; M23 = 0*I        ; M24 = 0*I
    M31 = 0*I          ; M32 = 0*I     ; M33 = 0*I        ; M34 = 0*I
    M41 = 0*I          ; M42 = 0*I     ; M43 = 0*I        ; M44 = 1*I

    #-------------------- boundary conditions ----------------------
    L1 = np.block([ [L11,L12,L13,L14] ]);  M1 = np.block([ [M11,M12,M13,M14] ]) #u
    L2 = np.block([ [L21,L22,L23,L24] ]);  M2 = np.block([ [M21,M22,M23,M24] ]) #v
    L3 = np.block([ [L31,L32,L33,L34] ]);  M3 = np.block([ [M31,M32,M33,M34] ]) #p
    L4 = np.block([ [L41,L42,L43,L44] ]);  M4 = np.block([ [M41,M42,M43,M44] ]) #T

    yi = np.array( [*range(Ny)] ); yi = yi.flatten()

    # u
    bcA = np.argwhere( (np.abs(yi)==0) | (np.abs(yi)==Ny-1) ).flatten() # pos
    L1[bcA,:] = np.block([ [1.*I,  0.*I, 0.*I, 0.*I  ] ])[bcA,:] # dirichlet
    M1[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I, 0.*I  ] ])[bcA,:]

    # v
    bcA = np.argwhere( (np.abs(yi)==0) | (np.abs(yi)==Ny-1) ).flatten() # pos
    L2[bcA,:] = np.block([ [0.*I,  1.*I, 0.*I, 0.*I  ] ])[bcA,:] # dirichlet
    M2[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I, 0.*I  ] ])[bcA,:]
    #L2[bcB,:] = np.block([ [0.*I,1.*D1y, 0.*I, 0.*I  ] ])[bcA,:] # neumann
    #M2[bcB,:] = np.block([ [0.*I,  0.*I, 0.*I, 0.*I  ] ])[bcA,:]

    # p
    bcA = np.argwhere( (np.abs(yi)==0) | (np.abs(yi)==Ny-1) ).flatten()
    L3[bcA,:] = np.block([ [0.*I,  0.*I,1.*D1y, 0.*I ] ])[bcA,:] # neumann
    M3[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I , 0.*I ] ])[bcA,:]

    # T
    bcA = np.argwhere( (np.abs(yi)==0) | (np.abs(yi)==Ny-1) ).flatten() # pos
    L4[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I, 1.*I  ] ])[bcA,:] # dirichlet
    M4[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I, 0.*I  ] ])[bcA,:]

    #----------------------- solve EVP -----------------------------
    L = np.block([ [L1], [L2], [L3], [L4]])
    M = np.block([ [M1], [M2], [M3], [M4]])
    evals,evecs = eig(L,1.j*M)

    # Post Process egenvalues
    evals, evecs = sort_evals(evals,evecs,imag=True)
    evals, evecs = remove_evals(evals,evecs,cut=400)
    #--------------------- post-processing -------------------------
    if plot:
        blue = (0/255, 137/255, 204/255)
        red  = (196/255, 0, 96/255)
        yel   = (230/255,159/255,0)

        fig,(ax0,ax1,ax2) = plt.subplots(ncols=3, figsize=(8,3)) 
        ax0.set_title("Eigenvalues")
        ax0.set_xlim(-1,1);   ax0.set_ylim(-10,1); ax0.grid(True)
        ax0.scatter(np.real(evals[:]),np.imag(evals[:]), marker="o", edgecolors="k", s=60, facecolors='none'); 

        ax1.set_ylabel("y"); ax1.set_title("Largest Eigenvector")
        ax1.plot(np.abs(evecs[0*N:1*N,-1:]),y,  marker="", color=blue, label=r"$|u|$")
        ax1.plot(np.abs(evecs[1*N:2*N,-1:]),y,  marker="", color=red,  label=r"$|v|$")
        ax1.plot(np.abs(evecs[2*N:3*N,-1:]),y,  marker="", color="k" , label=r"$|p|$")
        ax1.legend(loc="lower right")
        ax2.set_ylabel("y"); ax2.set_title("Largest Eigenvector")
        ax2.plot(np.abs(evecs[3*N:4*N,-1:]),y,  marker="", color=yel , label=r"$|T|$")
        ax2.legend()
        plt.tight_layout(); plt.show()
    return evals,evecs