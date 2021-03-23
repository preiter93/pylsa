import numpy as np

# Sort and remove spurious eigenvalues
def print_evals(evals,n=None):
    if n is None:n=len(evals)
    print('{:>4s} largest eigenvalues:'.format(str(n)))
    print('\n'.join('{:4d}: {:10.4e} {:10.4e}j'.format(n-c,np.real(k),np.imag(k)) 
        for c,k in enumerate(evals[-n:])))

def sort_evals(evals,evecs,imag=False):
    if imag:
        idx = np.imag(evals).argsort()
    else:
        idx = np.real(evals).argsort()
    return evals[idx], evecs[:,idx]

def remove_evals(evals,evecs,cut=1000, imag=False):
    if imag:
        evecs=evecs[:,np.imag(evals)<=cut]
        evals=evals[np.imag(evals)<=cut]
    else:
        evecs=evecs[:,np.abs(evals)<=cut]
        evals=evals[np.abs(evals)<=cut]
    return evals,evecs