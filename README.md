# pylsa

Python code collection to solve the linear stabilty problem arising in convection problems.
rbc-1d.py and rbc-2d.py contain the special case of Rayleigh-Benard convection.

### Theoretical Background
The basis is the dimensionless Navier--Stokes equation:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Cpartial%7B%5Cbf%20u%7D/%5Cpartial%20t&plus;%7B%5Cbf%20u%7D%5Ccdot%20%5Cnabla%20%7B%5Cbf%20u%7D%26%3D-%5Cnabla%20%7Bp%7D&plus;%20Pr%20%5Cnabla%5E2%20%7B%5Cbf%20u%7D&plus;%20PrRa%7B%5Ctheta%7D%7B%5Cbf%20e%7D_z%20%5Cnonumber%2C%5C%5C%20%5Cpartial%7B%5Ctheta%7D/%5Cpartial%20t&plus;%7B%5Cbf%20u%7D%5Ccdot%20%5Cnabla%20%7B%5Ctheta%7D%26%3D%20%5Cnabla%5E2%20%7B%5Ctheta%7D%20%5Cnonumber%2C%20%5C%5C%20%5Cquad%20%5Cnabla%20%5Ccdot%20%7B%5Cbf%20u%7D%20%3D0.%20%5Cend%7Balign*%7D)

Using the modal ansatz for velocity, pressure and temperature fluctuations ![equation](https://latex.codecogs.com/gif.latex?%5Cphi%5Cequiv%5Cbegin%7Bbmatrix%7D%20u%5E%5Cprime%2C%20w%5E%5Cprime%2C%20p%5E%5Cprime%20%2C%5Ctheta%5E%5Cprime%20%5Cend%7Bbmatrix%7D%5ET)

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Cphi%20%26%3D%20%5Chat%7B%5Cphi%7D%28z%29%20e%5E%7Bi%20%5Calpha%20x-i%5Comega%20t%7D%20%5Cquad%20%5Ctext%7B%28infinite%20domain%29%7D%2C%5C%5C%20%5Cphi%20%26%3D%20%5Chat%7B%5Cphi%7D%28x%2Cz%29%20e%5E%7B-i%5Comega%20t%7D%20%5Cquad%20%5Ctext%7B%28finite%20domain%29%7D%2C%20%5Cend%7Balign*%7D)

the two dimensional linearized Navier-Stokes equation with thermal forcing can be written in Matrix form

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Cunderbrace%7B%20%5Cbegin%7Bbmatrix%7D%20L_%7B2D%7D%20&plus;%20D_x%20%5Coverline%7Bu%7D%20%26%20D_z%20%5Coverline%7Bu%7D%20%26%20D_x%20%26%200%5C%5C%20D_x%20%5Coverline%7Bw%7D%20%26%20L_%7B2D%7D%20&plus;%20D_z%20%5Coverline%7Bw%7D%20%26%20D_z%20%26%20-RaPr%5C%5C%20D_x%20%26%20D_z%20%26%200%20%26%200%5C%5C%20D_x%20%5Coverline%7B%5Ctheta%7D%20%26%20D_z%20%5Coverline%7B%5Ctheta%7D%20%26%200%20%26%20K_%7B2D%7D%5C%5C%20%5Cend%7Bbmatrix%7D%20%7D_%7B%5Cmathcal%7BA%7D%7D%20%5Cunderbrace%7B%20%5Cbegin%7Bbmatrix%7D%20%5Chat%7Bu%7D%20%5C%5C%20%5Chat%7Bv%7D%20%5C%5C%20%5Chat%7Bp%7D%20%5C%5C%20%5Chat%7B%5Ctheta%7D%20%5Cend%7Bbmatrix%7D%20%7D_%7B%5Chat%7B%5Cphi%7D%7D%20%3D%20i%20%5Comega%20%5Cunderbrace%7B%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%201%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%201%20%5C%5C%20%5Cend%7Bbmatrix%7D%20%7D_%7B%5Cmathcal%7BB%7D%7D%20%5Cunderbrace%7B%20%5Cbegin%7Bbmatrix%7D%20%5Chat%7Bu%7D%20%5C%5C%20%5Chat%7Bv%7D%20%5C%5C%20%5Chat%7Bp%7D%20%5C%5C%20%5Chat%7B%5Ctheta%7D%20%5Cend%7Bbmatrix%7D%20%7D_%7B%5Chat%7B%5Cphi%7D%7D%2C%20%5Clabel%7Beq%3Aevp%7D%20%5Cend%7Balign*%7D)


![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20L_%7B2D%7D%20%26%3D%20%5Coverline%7Bu%7D%20D_x%20&plus;%20%5Coverline%7Bw%7D%20D_z%20&plus;%20%5Csqrt%7BPr/Ra%7D%5Cleft%28-D_x%5E2-D_z%5E2%5Cright%29%20%2C%5C%5C%20K_%7B2D%7D%20%26%3D%20%5Coverline%7Bu%7D%20D_x%20&plus;%20%5Coverline%7Bw%7D%20D_z%20&plus;%201/%5Csqrt%7BRaPr%7D%5Cleft%28-D_x%5E2-D_z%5E2%5Cright%29%2C%20%5Cend%7Balign*%7D)

where overlines denote mean flow quantities and D_x is a suitable differentiation matrix (DM), for example a chebyshev DM. 
This is a generalized Eigenvalue problem (EVP) of the form

![equation](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BA%7D%5Chat%7B%5Cphi%7D%20%3D%20i%5Comega%20%5Cmathcal%7BB%7D%5Chat%7B%5Cphi%7D)

of size ![equation](https://latex.codecogs.com/gif.latex?%5BN_x%20%5Ctimes%20N_z%20%5Ctimes%204%5D) and can solved directly or iteratively.

### Publications
P. Reiter and X. Zhang and R. Stepanov and O. Shishkina, Generation of zonal flows in convective systems by travelling thermal waves, J. Fluid Mech., 913 (2021), A13
