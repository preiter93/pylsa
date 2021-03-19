from pylsa.rbc2d import solve_rbc2d

# Parameters
Gamma = 0.5
Ny    = 21
Nx    = int(max(1.0*Ny,Ny*Gamma) )
Ra    = 12.5e3
Pr    = 1.0

evals,evecs,x,z = solve_rbc2d(Nx=Nx,Ny=Ny,Ra=Ra,Pr=Pr,
	aspect=Gamma,directsolver=False,plot=True)