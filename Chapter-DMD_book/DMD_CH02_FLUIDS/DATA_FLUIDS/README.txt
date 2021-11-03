DATA SET FOR FLUID FLOW PAST A CYLINDER AT RE=100
Steven L. Brunton
May 12, 2016

=======

DATA Folder:

    ALL.mat  -  151 snapshots of u and v velocity as well as vorticity (UALL, VALL, VORTALL).  Also includes an extra snapshot for testing.  (snapshot resolution 449 x 199)

    basis.mat  -  Contains mean vorticity field, unsteady fixed point (vorticity) and leading 41 POD modes (vortPHI).

    ibpmUNSTEADY.plt  -  Tecplot file containing unsteady equilibrium solution

=======

Data is generated using Immersed Boundary Projection Method (IBPM) Code: 
    https://github.com/cwrowley/ibpm

IBPM code is based on immersed boundary method of Taira and Colonius:
    The immersed boundary method: a projection approach
    K. Taira and T. Colonius
    Journal of Computational Physics 225(2):2118—2137, 2007
    http://dx.doi.org/10.1016/j.jcp.2007.03.005

    A fast immersed boundary method using a nullspace approach and multi-domain far-field boundary conditions
    T. Colonius and K. Taira
    Computer Methods in Applied Mechanics and Engineering 197(25):2131-2146, 2008
    http://dx.doi.org/10.1016/j.cma.2007.08.014

