'''
This file is a  working example of an elliptic (time independent) PDE solver.


'''

import PDEFEMCode.pde_utils as pde_utils
import fenics as fc
import numpy as np


# Set up the LHS and RHS forms that are used.
# This problem solves the equation
# -\Delta f + f = u,
# for u = u_max * exp(|x-gamma|^2/ r^2) a Gaussian point source (at gamma).
#   (see functions elliptic_LHS, elliptic_RHS for templates on how to change the PDE.)
LHS = pde_utils.elliptic_LHS
RHS = pde_utils.elliptic_RHS
gamma = np.array([.7,.5])
u_max = 1
# Note: the expression below is using the string format function to build it.
#   Gamma is split into two separate numbers.
u_expression = ('exp(-(pow(x[0] - {},2) + pow(x[1] - {},2))'.format(*gamma))

RHS_fn = fc.Expression('exp(-(pow(x[0] - {},2) + pow(x[1] - {},2))')

# Parameters determining the mesh.
# This is a square size 0.01 mesh on the unit square.
nx, ny = 100,100
P0, P1 = np.array([0,0]), np.array([1,1])


## ------Begin main code.----------

mesh, fn_space = pde_utils.setup_rectangular_function_space(nx,ny,P0,P1)
u_trial = fc.TrialFunction(fn_space)
v_test = fc.TestFunction(fn_space)

# Setup variational formulation, tying the LHS form with the trial function
# and the RHS form with the test functions and the RHS function.
LHS_int, RHS_int = pde_utils.variational_formulation(
    u_trial, v_test,
    LHS,
    RHS, RHS_fn
)


u_sol = pde_utils.solve_vp(fn_space,LHS_int,RHS_int)

# Obtain solution functions.
f = pde_utils.FenicsRectangleLinearInterpolator(nx,ny,P0,P1,u_sol)
grad_f = pde_utils.FenicsRectangleGradInterpolator(nx,ny,P0,P1,u_sol)