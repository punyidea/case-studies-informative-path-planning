'''
This file is a working example in the discretization of the heat equation with homogeneous Neumann condition, and zero
initial condition.

# This problem solves the equation
# D_t u  - \alpha \Delta u = f, u(0)=0, homogeneous Neumann BC and initial conditions
# for f(x,y,t) = 3(-1+x+y)-3(-1+x+y)cos(3t)+(3(3x^2-2y^3+(3-2y)y^2 sin(3t))/4

# Set up the LHS and RHS forms that are used.
#   (see functions elliptic_LHS, elliptic_RHS for templates on how to change the PDE.)
'''

import PDEFEMCode.fenics_utils as pde_utils
import PDEFEMCode.interface as pde_interface
import fenics as fc
import numpy as np
import argparse,sys

#load parameters from the file given by the first command line argument.
parser = argparse.ArgumentParser()
parser.add_argument('-y','--yaml_fname',required=True)
args = parser.parse_args()
params_yml = pde_interface.yaml_load(args.yaml_fname)
in_params = pde_utils.yaml_parse_parabolic(params_yml, args.yaml_fname)

var_form_p = in_params.var_form
var_form_fn_handles = pde_utils.VarFormFnHandles(var_form_p)

# Parameters determining the mesh.
# This is a triangular mesh on the unit square, where we have [0,1]^2 divided into 100 x 100 sub-squares, each one
# subdivided into two triangles by the down-left to up-right diagonal. In total there are 2 x 100 x 100 triangles.
mesh_p = in_params.rect_mesh
# Parameters determining the time discretization. The simulation is conducted from time 0 to T_fin, for a total of
# Nt uniformly distributed times.
time_disc_p = in_params.time_disc
# File save place
io_p = in_params.io
# ------Begin main code.----------

# Space discretization
mesh, fn_space = pde_utils.setup_rectangular_function_space(mesh_p)

# Time discretization
dt, times = pde_utils.setup_time_discretization(time_disc_p)

u_ref = fc.Expression('((-3*pow(x[0],2) + 2*pow(x[0],3) + pow(x[1],2)*(-3 + 2*x[1]))*(-1 + cos(3*t)))/4',
                      degree=2, t=0)

if not var_form_p.rhs_expression_str:
    var_form_p.rhs_expression_str = var_form_fn_handles.rhs_expression(**var_form_p.rhs_exp_params)

RHS_fn = fc.Expression(
    var_form_p.rhs_expression_str, degree=2, t=0)

# Setup variational formulation, tying the LHS form with the trial function
# and the RHS form with the test functions and the RHS function.
u_initial = fc.interpolate(fc.Constant(0), fn_space)  # the solution is zero according to our problem formulation
u_trial = fc.TrialFunction(fn_space)
v_test = fc.TestFunction(fn_space)

u_previous = u_initial
LHS_int, RHS_int = pde_utils.variational_formulation(
    u_trial, v_test,
    var_form_fn_handles.LHS,
    var_form_fn_handles.RHS, RHS_fn,
    {'dt': dt},
    {'dt': dt, 'u_previous': u_previous}
)

t = times[0]  # initial time
fenics_list = []  # list of Fenics functions to be passed to the interpolators

# Solving
fc.set_log_active(False)  # disable messages of Fenics
for n in range(time_disc_p.Nt):
    # Time update
    t = times[n + 1]
    RHS_fn.t = t  # NB. This change is also reflected inside LHS_int, RHS_int
    u_ref.t = t
    # Solution at this time
    u_current = pde_utils.solve_vp(fn_space, LHS_int, RHS_int)

    # Update previous solution
    u_previous.assign(u_current)
    fenics_list.append(u_current)

    # Notify that something has happened
    print("Solving PDE, " + str(np.floor(100 * t/time_disc_p.T_fin)) + " % done.")

# Prepending initial condition
fenics_list.insert(0, fc.interpolate(fc.Constant(0), fn_space))

# Obtaining the interpolator for u
u = pde_interface.FenicsRectangleLinearInterpolator(mesh_p, fenics_list, T_fin=time_disc_p.T_fin, Nt=time_disc_p.Nt, time_dependent=True, verbose=True, time_as_indices=True)
grad_u_list = [pde_utils.fenics_grad(mesh,u_fenics) for u_fenics in fenics_list]
grad_u = pde_interface.FenicsRectangleVecInterpolator(mesh_p, grad_u_list, T_fin=time_disc_p.T_fin, Nt=time_disc_p.Nt, time_dependent=True, time_as_indices=True)

param_save = {'f':u,'grad_f':grad_u,'params':in_params}

pde_interface.pickle_save(io_p.out_folder, io_p.out_file_prefix, param_save)
