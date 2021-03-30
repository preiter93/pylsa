from pylsa.rbc2d import solve_rbc2d,solve_rbc2d_neutral

# Parameters
Gamma = 1.0
Ny    = 25
Nx    = int(max(1.0*Ny,Ny*Gamma) )
Ra    = 2.6e3
Pr    = 1.0

# Find the growth rates for given Ra
evals,evecs,x,z = solve_rbc2d(Nx=Nx,Ny=Ny,Ra=Ra,Pr=Pr,aspect=Gamma,
	sidewall="adiabatic",directsolver=False,plot=True)

# Find Rac where the growth rate is zero
evals,evecs,x,z = solve_rbc2d_neutral(Nx=Nx,Ny=Ny,Pr=Pr,aspect=Gamma,
	sidewall="adiabatic",directsolver=False,plot=True)