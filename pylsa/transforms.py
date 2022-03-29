import numpy as np

# Transform is necessary if numerical domain /= physical domain
# See: https://en.wikipedia.org/wiki/Chain_rule#Higher_derivatives
# or Faà di Bruno's formula
def cheb_coord_transform(D1, D2, d1, d2):
    D1_Z = np.diag(d1) @ D1
    D2_Z = np.diag(d2) @ D1 + np.diag(d1 ** 2) @ D2
    return D1_Z, D2_Z


# [-1,1] to [0,L]
def zerotoL_transform(x, L):
    z = (1 + x) / 2 * L
    diff1 = 2 / L + 0 * z
    diff2 = 0 * z
    return z, diff1, diff2


# [-1,1] to [0,1]
def zerotoone_transform(x):
    z = (1 + x) / 2
    diff1 = 2 + 0 * z
    diff2 = 0 * z
    return z, diff1, diff2


def chebder_transform(x, D1, D2, fun_transform, **kwargs):
    z, d1, d2 = fun_transform(x, **kwargs)
    D1z, D2z = cheb_coord_transform(D1, D2, d1, d2)
    return z, D1z, D2z
