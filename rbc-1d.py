from pylsa.rbc1d import solve_rbc1d

# Parameters
Ny    = 51
alpha = 3.1
Ra    = 1708
Pr    = 1.0

evals,evecs = solve_rbc1d(Ny=Ny,Ra=Ra,Pr=Pr,alpha=alpha,plot=True)