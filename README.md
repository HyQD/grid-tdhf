# grid-tdhf

##
Run simulations by
```bash
grid-tdhf -<input_param_1=param_1> ... -<input_param_k=param_k>
```
Running just
```bash
grid-tdhf 
```
starts a simulation with all input parameters set to default values.

###
List of input parameters
- atom: atomic specie (He, Be, Ne or Ar).  
- r_max: The length of the radial grid.
- N: The order of the underlying Legendre polynomial which determines the number of grid points. 
- l_max: the number of angular momenta in the expansion of the orbitals 
- nL: the cut-off value for angular momentum in the multipole expansion for the Coulomb interaction. 
- E0: The maximum field strength of the time-dependent external electric field.
- omega: The carrier frequency of the time-dependent external electric field.
- ncycles: The duration of the field given in optical cycles ($t_c=\frac{2\pi}{\omega}$).
- ncycles_after_pulse: Simulation time after the pulse.
- integrator_name: which integrator to use (CN or CNCMF2). 
- dt: timestep for integration 
- mask_margin: a mask function is applied from the point $r_m = r_{max}$-mask_margin to avoid reflection at   the grid boundary.
