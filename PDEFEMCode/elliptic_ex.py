'''
This file is a  working example of an elliptic (time independent) PDE solver.


'''
import PDEFEMCode.fenics_utils
import PDEFEMCode.interface
import PDEFEMCode.fenics_utils as pde_utils
import PDEFEMCode.interface as pde_IO
import fenics as fc
import numpy as np
import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('-y','--yaml_fname',required=True)
args = parser.parse_args()
params_yml = pde_IO.yaml_load(args.yaml_fname)
in_params = PDEFEMCode.fenics_utils.yaml_parse_elliptic(params_yml, args.yaml_fname)


# Set up the LHS and RHS forms that are used.
# This problem solves the equation
# -\Delta f + f = u,
# for u = u_max * exp(|x-gamma|^2/ r^2) a Gaussian point source (at gamma).
#   (see functions elliptic_LHS, elliptic_RHS for templates on how to change the PDE.)
var_form_p = in_params.var_form

# Note: the expression below is using the string format function to build it.
#   Gamma is split into two separate numbers.

# Parameters determining the mesh.
# This is a square size 0.01 mesh on the unit square.
mesh_p = in_params.rect_mesh

# File save place
io_p = in_params.io
## ------Begin main code.----------

mesh, fn_space = pde_utils.setup_rectangular_function_space(mesh_p)
u_trial = fc.TrialFunction(fn_space)
v_test = fc.TestFunction(fn_space)

# Setup variational formulation, tying the LHS form with the trial function
# and the RHS form with the test functions and the RHS function.
if not var_form_p.rhs_expression_str:
    var_form_p.rhs_expression_str = var_form_p.rhs_expression(**var_form_p.rhs_exp_params)


RHS_fn = fc.Expression(var_form_p.rhs_expression_str, element = fn_space.ufl_element())
LHS_int, RHS_int = pde_utils.variational_formulation(
    u_trial, v_test,
    var_form_p.LHS,
    var_form_p.RHS, RHS_fn
)

u_sol = pde_utils.solve_vp(fn_space,LHS_int,RHS_int)

# Obtain solution functions, such that they work with numpy.

f = PDEFEMCode.interface.FenicsRectangleLinearInterpolator(mesh_p, u_sol)
u_grad = pde_utils.fenics_grad(mesh,u_sol)
grad_f = PDEFEMCode.interface.FenicsRectangleVecInterpolator(mesh_p, u_grad)

param_save = {'f':f,'grad_f':grad_f,'params':in_params}

pde_IO.pickle_save(io_p.out_folder,io_p.out_file_prefix,param_save)

# # Code snippet to build intuition on the objects.
# import matplotlib.pyplot as plt
# coords = np.stack(np.meshgrid(np.linspace(0,1,200),np.linspace(0,1,200)),axis = -1)
# coords_rs = coords.reshape(-1,2)
# f_eval = f(coords_rs)
# plt.imshow(f_eval.reshape(200,200))
# #
# grad_f_eval = grad_f(coords) #grad_f(coords_rs) works fine too
# plt.imshow(grad_f_eval[:,:,0])