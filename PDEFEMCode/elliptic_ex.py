'''
This file is a  working example of an elliptic (time independent) PDE solver.


'''
import PDEFEMCode.interface
import PDEFEMCode.fenics_utils as pde_utils
import PDEFEMCode.interface as pde_IO
import fenics as fc
import numpy as np



# Set up the LHS and RHS forms that are used.
# This problem solves the equation
# -\Delta f + f = u,
# for u = u_max * exp(|x-gamma|^2/ r^2) a Gaussian point source (at gamma).
#   (see functions elliptic_LHS, elliptic_RHS for templates on how to change the PDE.)
LHS = pde_utils.elliptic_LHS
RHS = pde_utils.elliptic_RHS
rhs_expression_params = {   'gamma':np.array([.7,.5]),
                            'u_max' : 1,
                            'r' : .05
                            }
rhs_expression = pde_utils.gaussian_expression_2D
# Note: the expression below is using the string format function to build it.
#   Gamma is split into two separate numbers.

# Parameters determining the mesh.
# This is a square size 0.01 mesh on the unit square.
nx, ny = 100,100
P0, P1 = np.array([0,0]), np.array([1,1])

# File save place
out_file = 'elliptic_nx_100_unit_square_gaussian'
out_folder = ''
## ------Begin main code.----------

mesh, fn_space = pde_utils.setup_rectangular_function_space(nx,ny,P0,P1)
u_trial = fc.TrialFunction(fn_space)
v_test = fc.TestFunction(fn_space)

# Setup variational formulation, tying the LHS form with the trial function
# and the RHS form with the test functions and the RHS function.
u_rhs_expression = rhs_expression(**rhs_expression_params)
RHS_fn = fc.Expression(u_rhs_expression, element = fn_space.ufl_element())
LHS_int, RHS_int = pde_utils.variational_formulation(
    u_trial, v_test,
    LHS,
    RHS, RHS_fn
)

u_sol = pde_utils.solve_vp(fn_space,LHS_int,RHS_int)

# Obtain solution functions, such that they work with numpy.
f = PDEFEMCode.interface.FenicsRectangleLinearInterpolator(nx, ny, P0, P1, u_sol)
u_grad = pde_utils.fenics_grad(mesh,u_sol)
grad_f = PDEFEMCode.interface.FenicsRectangleVecInterpolator(nx, ny, P0, P1, u_grad)

param_save = {'f':f,'grad_f':grad_f}

pde_IO.pickle_save(out_folder,out_file,param_save)

# # Code snippet to build intuition on the objects.
# import matplotlib.pyplot as plt
# coords = np.stack(np.meshgrid(np.linspace(0,1,200),np.linspace(0,1,200)),axis = -1)
# coords_rs = coords.reshape(-1,2)
# f_eval = f(coords_rs)
# plt.imshow(f_eval.reshape(200,200))
# #
# grad_f_eval = grad_f(coords) #grad_f(coords_rs) works fine too
# plt.imshow(grad_f_eval[:,:,0])