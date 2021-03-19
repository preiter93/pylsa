from pylsa.rbc2d import solve_rbc2d

# Parameters
Gamma = 2.0
Ny    = 15
Nx    = int(Ny*Gamma)
Ra    = 2200
Pr    = 1.0

evals,evecs,x,z = solve_rbc2d(Nx=Nx,Ny=Ny,Ra=Ra,Pr=Pr,
	aspect=Gamma,directsolver=False,plot=True)