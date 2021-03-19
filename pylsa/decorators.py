from pylsa.utils import print_evals
from time import time

def io_decorator(func):
    def wrapper(*args,**kwargs):
        print("Input Parameter:")
        for k, v in kwargs.items(): print(k,":", v)
        start = time()
        result = func(*args,**kwargs)
        evals = result[0]
        print_evals(evals, n=3)
        end = time(); print("Time used: {:8.2f} s.".format(end-start))
        return result
    return wrapper