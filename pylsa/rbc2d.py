import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from pylsa.utils import *
from pylsa.transforms import *
from pylsa.dmsuite import *
from pylsa.decorators import *
import matplotlib.pyplot as plt

#--------------------------------------------------------------------
@io_decorator
def solve_rbc2d(Nx=21,Ny=21, Ra=2000, Pr=1,aspect=1, directsolver=False,plot=True):
    #----------------------- Parameters ---------------------------
    nu = np.sqrt(Pr/Ra)
    kappa = 1/np.sqrt(Ra*Pr)

    #------------- setup discrete diff matrices  -------------------
    # chebyshev in x-direction
    x,D1x = chebdif(Nx-1,1)
    x,D2x = chebdif(Nx-1,2)
    # Transform [0,L]
    x,D1x,D2x = chebder_transform(x,D1x,D2x, 
        zerotoL_transform,L=aspect)

    # chebyshev in y-direction
    y,D1y = chebdif(Ny-1,1)
    y,D2y = chebdif(Ny-1,2)
    # Transform [0,1]
    y,D1y,D2y = chebder_transform(y,D1y,D2y, zerotoone_transform)

    #---------------------- extend mat to 2d -----------------------
    xx,yy = np.meshgrid(x,y,indexing="ij")
    Iy,Ix = np.eye(Ny), np.eye(Nx)
    N   = Ny*Nx          ; I   = np.eye(Ny*Nx)
    D1y = np.kron(Ix,D1y); D2y = np.kron(Ix,D2y)
    D1x = np.kron(D1x,Iy); D2x = np.kron(D2x,Iy)

    #----------------------- mean flow -----------------------------
    # RBC FLOW
    U    =    0*yy;   U = U.flatten(); U_y = D1y@U; U_x = D1x@U 
    T    = -1.0*yy+1; T = T.flatten(); T_y = D1y@T; T_x = D1x@T
    W    =    0*yy;   W = W.flatten(); W_y = D1y@W; W_x = D1x@W
    # Derivatives
    UU, UU_y, UU_x = np.diag(U), np.diag(U_y), np.diag(U_x)
    WW, WW_y, WW_x = np.diag(W), np.diag(W_y), np.diag(W_x)
    _ , TT_y, TT_x = np.diag(T), np.diag(T_y), np.diag(T_x)

    #-------------------- construct matrix  ------------------------
    L2d = WW@D1y + UU@D1x - nu*(D2x + D2y)
    K2d = WW@D1y + UU@D1x - kappa*(D2x + D2y)

    #lhs
    L11 = 1*L2d+1*UU_x ; L12 = 1*UU_y         ; L13 = 1*D1x      ; L14 =  0*I
    L21 = 1*WW_x       ; L22 = 1*L2d+1*WW_y   ; L23 = 1*D1y      ; L24 = -1*I
    L31 = 1*D1x        ; L32 = 1*D1y          ; L33 = 0*I        ; L34 =  0*I
    L41 = 1*TT_x       ; L42 = 1*TT_y         ; L43 = 0*I        ; L44 =  1*K2d

    #rhs
    M11 = 1*I          ; M12 = 0*I     ; M13 = 0*I        ; M14 = 0*I
    M21 = 0*I          ; M22 = 1*I     ; M23 = 0*I        ; M24 = 0*I
    M31 = 0*I          ; M32 = 0*I     ; M33 = 0*I        ; M34 = 0*I
    M41 = 0*I          ; M42 = 0*I     ; M43 = 0*I        ; M44 = 1*I

    #-------------------- boundary conditions ----------------------
    SV = 4 # number of variables
    L1 = np.block([ [L11,L12,L13,L14] ]);  M1 = np.block([ [M11,M12,M13,M14] ]) #u
    L2 = np.block([ [L21,L22,L23,L24] ]);  M2 = np.block([ [M21,M22,M23,M24] ]) #v
    L3 = np.block([ [L31,L32,L33,L34] ]);  M3 = np.block([ [M31,M32,M33,M34] ]) #p
    L4 = np.block([ [L41,L42,L43,L44] ]);  M4 = np.block([ [M41,M42,M43,M44] ]) #T

    xi, yi = np.meshgrid([*range(Nx)],[*range(Ny)],indexing="ij"); xi = xi.flatten(); yi = yi.flatten()
    # y-direction
    # u
    bcA = np.argwhere( (np.abs(yi)==0) | (np.abs(yi)==Ny-1) ).flatten() # pos
    L1[bcA,:] = np.block([ [1.*I,  0.*I, 0.*I, 0.*I  ] ])[bcA,:] # dirichlet
    M1[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I, 0.*I  ] ])[bcA,:]

    # v
    bcA = np.argwhere( (np.abs(yi)==0) | (np.abs(yi)==Ny-1) ).flatten() 
    L2[bcA,:] = np.block([ [0.*I,  1.*I, 0.*I, 0.*I  ] ])[bcA,:] # dirichlet
    M2[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I, 0.*I  ] ])[bcA,:]

    # p
    bcA = np.argwhere( (np.abs(yi)==0) | (np.abs(yi)==Ny-1) ).flatten()
    L3[bcA,:] = np.block([ [0.*I,  0.*I,1.*D1y, 0.*I ] ])[bcA,:] # neumann
    M3[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I , 0.*I ] ])[bcA,:]

    # T
    bcA = np.argwhere( (np.abs(yi)==0) | (np.abs(yi)==Ny-1) ).flatten() # pos
    L4[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I, 1.*I  ] ])[bcA,:] # dirichlet
    M4[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I, 0.*I  ] ])[bcA,:]

    # x-direction
    # u
    bcA = np.argwhere( (np.abs(xi)==0) | (np.abs(xi)==Nx-1) ).flatten() # pos
    L1[bcA,:] = np.block([ [1.*I,  0.*I, 0.*I, 0.*I  ] ])[bcA,:] # dirichlet
    M1[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I, 0.*I  ] ])[bcA,:]

    # v
    bcA = np.argwhere( (np.abs(xi)==0) | (np.abs(xi)==Nx-1) ).flatten() 
    #bcB = np.argwhere( (np.abs(xi)==1) | (np.abs(xi)==Nx-2) ).flatten()
    L2[bcA,:] = np.block([ [0.*I,  1.*I, 0.*I, 0.*I  ] ])[bcA,:] # dirichlet
    M2[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I, 0.*I  ] ])[bcA,:]

    # p
    bcA = np.argwhere( (np.abs(xi)==0) | (np.abs(xi)==Nx-1) ).flatten()
    L3[bcA,:] = np.block([ [0.*I,  0.*I,1.*D1x, 0.*I ] ])[bcA,:] # neumann
    M3[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I , 0.*I ] ])[bcA,:]

    # T
    bcA = np.argwhere( (np.abs(xi)==0) | (np.abs(xi)==Nx-1) ).flatten() # pos
    L4[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I, 1.*D1x  ] ])[bcA,:] # neumann
    M4[bcA,:] = np.block([ [0.*I,  0.*I, 0.*I, 0.*I    ] ])[bcA,:]

    #----------------------- solve EVP -----------------------------
    L = np.block([ [L1], [L2], [L3], [L4]])
    M = np.block([ [M1], [M2], [M3], [M4]])
    M = 1.j*M

    if (directsolver):
        evals,evecs = eig(L,M)
    else:
        Li = np.linalg.inv(L)
        evals, evecs = eigs(Li@(M), k=N*SV,sigma=0,which="LI",tol=1e-4, maxiter=2000); #shift and invert
        evals = 1/evals

    # Post Process egenvalues
    evals, evecs = sort_evals(evals,evecs,imag=True)
    evals, evecs = remove_evals(evals,evecs,cut=100) 

    if plot:
        #plot most unstable mode
        mode = 1
        N     = Ny*Nx
        U0    = np.real(evecs[0*N:1*N,-mode].reshape(Nx,Ny))
        V0    = np.real(evecs[1*N:2*N,-mode].reshape(Nx,Ny))
        P0    = np.real(evecs[2*N:3*N,-mode].reshape(Nx,Ny))
        T0    = np.real(evecs[3*N:4*N,-mode].reshape(Nx,Ny))

        cmap = "viridis"
        xx,yy = np.meshgrid(x,y,indexing="ij")
        fig,ax=plt.subplots(ncols=4,figsize=(12,3),dpi=120)
        ax[0].contourf(xx,yy,T0,cmap=cmap,levels=np.linspace(np.amin(T0),np.amax(T0),50))
        ax[1].contourf(xx,yy,P0,cmap=cmap,levels=np.linspace(np.amin(P0),np.amax(P0),50))
        ax[2].contourf(xx,yy,U0,cmap=cmap,levels=np.linspace(np.amin(U0),np.amax(U0),50))
        ax[3].contourf(xx,yy,V0,cmap=cmap,levels=np.linspace(np.amin(V0),np.amax(V0),50))

        ax[0].title.set_text('t')
        ax[1].title.set_text('p')
        ax[2].title.set_text('u')
        ax[3].title.set_text('v')
        for a in ax:
            a.set_aspect(1)
        plt.show()

    return evals, evecs,x,y