import sys
sys.path.insert(0, "..")

from pylsa.rbc1d import solve_rbc1d,solve_rbc1d_neutral

# Parameters
Ny    = 51
alpha = 3.1
Ra    = 1708
Pr    = 1.0

# Find the growth rates for given Ra
evals,evecs = solve_rbc1d(Ny=Ny,Ra=Ra,Pr=Pr,
	alpha=alpha,plot=True)

# Find Rac where the growth rate is zero
evals,evecs = solve_rbc1d_neutral(Ny=Ny,Pr=Pr,
	alpha=alpha,plot=True)
