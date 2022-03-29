import numpy as np

# Sort and remove spurious eigenvalues
def print_evals(evals, n=None):
    if n is None:
        n = len(evals)
    print("{:>4s} largest eigenvalues:".format(str(n)))
    print(
        "\n".join(
            "{:4d}: {:10.4e} {:10.4e}j".format(n - c, np.real(k), np.imag(k))
            for c, k in enumerate(evals[-n:])
        )
    )


def sort_evals(evals, evecs, which="M"):
    assert which in ["M", "I", "R"]
    if which == "I":
        idx = np.imag(evals).argsort()
    if which == "R":
        idx = np.real(evals).argsort()
    if which == "M":
        idx = np.abs(evals).argsort()
    return evals[idx], evecs[:, idx]


def remove_evals(evals, evecs, lower=-np.inf, higher=1000, which="M"):
    assert which in ["M", "I", "R"]
    if which == "I":
        arg = np.where(
            np.logical_and(lower <= np.imag(evals), np.imag(evals) <= higher)
        )[0]
    if which == "R":
        arg = np.where(
            np.logical_and(lower <= np.real(evals), np.real(evals) <= higher)
        )[0]
    if which == "M":
        arg = np.where(np.abs(evals) <= higher)[0]
    evecs = evecs[:, arg]
    evals = evals[arg]
    return evals, evecs
