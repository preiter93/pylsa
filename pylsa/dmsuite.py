import numpy as np
from scipy.linalg import toeplitz

# -- Full dmsuite for python: https://github.com/labrosse/dmsuite

def chebdif(ncheb, mder):
    """
    Calculate differentiation matrices using Chebyshev collocation.
    Returns the differentiation matrices D1, D2, .. Dmder corresponding to the
    mder-th derivative of the function f, at the ncheb Chebyshev nodes in the
    interval [-1,1].
    Parameters
    ----------
    ncheb : int, polynomial order. ncheb + 1 collocation points
    mder   : int
          maximum order of the derivative, 0 < mder <= ncheb - 1
    Returns
    -------
    x  : ndarray
         (ncheb + 1) x 1 array of Chebyshev points
    DM : ndarray
         mder x ncheb x ncheb  array of differentiation matrices
    Notes
    -----
    This function returns  mder differentiation matrices corresponding to the
    1st, 2nd, ... mder-th derivates on a Chebyshev grid of ncheb points. The
    matrices are constructed by differentiating ncheb-th order Chebyshev
    interpolants.
    The mder-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    .. math::
    f^{(m)}_i = D^{(m)}_{ij}f_j
    The code implements two strategies for enhanced accuracy suggested by
    W. Don and S. Solomonoff :
    (a) the use of trigonometric  identities to avoid the computation of
    differences x(k)-x(j)
    (b) the use of the "flipping trick"  which is necessary since sin t can
    be computed to high relative precision when t is small whereas sin (pi-t)
    cannot.
    It may, in fact, be slightly better not to implement the strategies
    (a) and (b). Please consult [3] for details.
    This function is based on code by Nikola Mirkov
    http://code.google.com/p/another-chebpy
    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
    ..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
    Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487
    Examples
    --------
    The derivatives of functions is obtained by multiplying the vector of
    function values by the differentiation matrix. The N-point Chebyshev
    approximation of the first two derivatives of y = f(x) can be obtained
    as
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import dmsuite as dm
    >>> ncheb = 32; mder = 2; pi = np.pi
    >>> x, D = dm.chebdif(ncheb, mder)        # first two derivatives
    >>> D1 = D[0,:,:]                   # first derivative
    >>> D2 = D[1,:,:]                   # second derivative
    >>> y = np.sin(2 * pi * x)                      # function at Chebyshev nodes
    >>> yd = 2 * pi * np.cos(2 * pi * x)        # theoretical first derivative
    >>> ydd = - 4 * pi ** 2 * np.sin(2 * pi * x)  # theoretical second derivative
    >>> fig, axe = plt.subplots(3, 1, sharex=True)
    >>> axe[0].plot(x, y)
    >>> axe[0].set_ylabel(r'$y$')
    >>> axe[1].plot(x, yd, '-')
    >>> axe[1].plot(x, np.dot(D1, y), 'o')
    >>> axe[1].set_ylabel(r'$y^{\prime}$')
    >>> axe[2].plot(x, ydd, '-')
    >>> axe[2].plot(x, np.dot(D2, y), 'o')
    >>> axe[2].set_xlabel(r'$x$')
    >>> axe[2].set_ylabel(r'$y^{\prime\prime}$')
    >>> plt.show()
    """

    if mder >= ncheb + 1:
        raise Exception('number of nodes must be greater than mder')

    if mder <= 0:
        raise Exception('derivative order must be at least 1')

    DM = np.zeros((mder, ncheb + 1, ncheb + 1))
    # indices used for flipping trick
    nn1 = np.int(np.floor((ncheb + 1) / 2))
    nn2 = np.int(np.ceil((ncheb + 1) / 2))
    k = np.arange(ncheb+1)
    # compute theta vector
    th = k * np.pi / ncheb

    # Compute the Chebyshev points

    # obvious way
    #x = np.cos(np.pi*np.linspace(ncheb-1,0,ncheb)/(ncheb-1))
    # W&R way
    x = np.sin(np.pi*(ncheb - 2 * np.linspace(ncheb, 0, ncheb + 1))/(2 * ncheb))
    x = x[::-1]

    # Assemble the differentiation matrices
    T = np.tile(th/2, (ncheb + 1, 1))
    # trigonometric identity
    DX = 2*np.sin(T.T+T)*np.sin(T.T-T)
    # flipping trick
    DX[nn1:, :] = -np.flipud(np.fliplr(DX[0:nn2, :]))
    # diagonals of D
    DX[range(ncheb + 1), range(ncheb + 1)] = 1.
    DX = DX.T

    # matrix with entries c(k)/c(j)
    C = toeplitz((-1.)**k)
    C[0, :] *= 2
    C[-1, :] *= 2
    C[:, 0] *= 0.5
    C[:, -1] *= 0.5

    # Z contains entries 1/(x(k)-x(j))
    Z = 1 / DX
    # with zeros on the diagonal.
    Z[range(ncheb + 1), range(ncheb + 1)] = 0.

    # initialize differentiation matrices.
    D = np.eye(ncheb + 1)

    for ell in range(mder):
        # off-diagonals
        D = (ell + 1) * Z * (C * np.tile(np.diag(D), (ncheb + 1, 1)).T - D)
        # negative sum trick
        D[range(ncheb + 1), range(ncheb + 1)] = -np.sum(D, axis=1)
        # store current D in DM
        DM[ell, :, :] = D

    # return only one differntiation matrix
    D = DM[mder-1,:,:]
    # mirror x [1,-1] to x [-1,1] (for convenience)
    x, D = x[::-1], D[::-1,::-1] 
    return x, D

def fourdif(nfou, mder):
    """
    Fourier spectral differentiation.
    Spectral differentiation matrix on a grid with nfou equispaced points in [0,2pi)
    INPUT
    -----
    nfou: Size of differentiation matrix.
    mder: Derivative required (non-negative integer)
    OUTPUT
    -------
    xxt: Equispaced points 0, 2pi/nfou, 4pi/nfou, ... , (nfou-1)2pi/nfou
    ddm: mder'th order differentiation matrix
    Explicit formulas are used to compute the matrices for m=1 and 2.
    A discrete Fouier approach is employed for m>2. The program
    computes the first column and first row and then uses the
    toeplitz command to create the matrix.
    For mder=1 and 2 the code implements a "flipping trick" to
    improve accuracy suggested by W. Don and A. Solomonoff in
    SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    The flipping trick is necesary since sin t can be computed to high
    relative precision when t is small whereas sin (pi-t) cannot.
    S.C. Reddy, J.A.C. Weideman 1998.  Corrected for MATLAB R13
    by JACW, April 2003.
    """
    # grid points
    xxt = 2*np.pi*np.arange(nfou)/nfou
    # grid spacing
    dhh = 2*np.pi/nfou

    nn1 = np.int(np.floor((nfou-1)/2.))
    nn2 = np.int(np.ceil((nfou-1)/2.))
    if mder == 0:
        # compute first column of zeroth derivative matrix, which is identity
        col1 = np.zeros(nfou)
        col1[0] = 1
        row1 = np.copy(col1)

    elif mder == 1:
        # compute first column of 1st derivative matrix
        col1 = 0.5*np.array([(-1)**k for k in range(1, nfou)], float)
        if nfou%2 == 0:
            topc = 1/np.tan(np.arange(1, nn2+1)*dhh/2)
            col1 = col1*np.hstack((topc, -np.flipud(topc[0:nn1])))
            col1 = np.hstack((0, col1))
        else:
            topc = 1/np.sin(np.arange(1, nn2+1)*dhh/2)
            col1 = np.hstack((0, col1*np.hstack((topc, np.flipud(topc[0:nn1])))))
        # first row
        row1 = -col1

    elif mder == 2:
        # compute first column of 1st derivative matrix
        col1 = -0.5*np.array([(-1)**k for k in range(1, nfou)], float)
        if nfou%2 == 0:
            topc = 1/np.sin(np.arange(1, nn2+1)*dhh/2)**2.
            col1 = col1*np.hstack((topc, np.flipud(topc[0:nn1])))
            col1 = np.hstack((-np.pi**2/3/dhh**2-1/6, col1))
        else:
            topc = 1/np.tan(np.arange(1, nn2+1)*dhh/2)/np.sin(np.arange(1, nn2+1)*dhh/2)
            col1 = col1*np.hstack((topc, -np.flipud(topc[0:nn1])))
            col1 = np.hstack(([-np.pi**2/3/dhh**2+1/12], col1))
        # first row
        row1 = col1

    else:
        # employ FFT to compute 1st column of matrix for mder > 2
        nfo1 = np.int(np.floor((nfou-1)/2.))
        nfo2 = -nfou/2*(mder+1)%2*np.ones((nfou+1)%2)
        mwave = 1j*np.concatenate((np.arange(nfo1+1), nfo2, np.arange(-nfo1, 0)))
        col1 = np.real(np.fft.ifft(mwave**mder*np.fft.fft(np.hstack(([1], np.zeros(nfou-1))))))
        if mder%2 == 0:
            row1 = col1
        else:
            col1 = np.hstack(([0], col1[1:nfou+1]))
            row1 = -col1
    ddm = toeplitz(col1, row1)
    return xxt,ddm