'''
This file is a working example in the discretization of the heat equation with homogeneous Neumann condition, and zero
initial condition.
'''

import PDEFEMCode.pde_utils as pde_utils
import PDEFEMCode.Object_IO as pde_IO
import fenics as fc
import numpy as np

# This problem solves the equation
# D_t u  - \alpha \Delta u = f, u(0)=0, homogeneous Neumann BC and initial conditions
# for f(x,y,t) = 3(-1+x+y)-3(-1+x+y)cos(3t)+(3(3x^2-2y^3+(3-2y)y^2 sin(3t))/4

# Set up the LHS and RHS forms that are used.
#   (see functions elliptic_LHS, elliptic_RHS for templates on how to change the PDE.)
LHS = pde_utils.heat_eq_LHS
RHS = pde_utils.heat_eq_RHS

# Note: the expression below is using the string format function to build it.
f_expression = '3*(-1 + x[0] + x[1]) - 3*(-1 + x[0] + x[1])*cos(3*t) + (3*(3*pow(x[0],2) - 2*pow(x[0],3) + (3 - 2*x[' \
               '1])*pow(x[1],2))*sin(3*t))/4 '

# Parameters determining the mesh.
# This is a triangular mesh on the unit square, where we have [0,1]^2 divided into 100 x 100 sub-squares, each one
# subdivided into two triangles by the down-left to up-right diagonal. In total there are 2 x 100 x 100 triangles.
N = 40
nx, ny = N, N
P0, P1 = np.array([0, 0]), np.array([1, 1])  # top right, bottom left corner

# Parameters determining the time discretization. The simulation is conducted from time 0 to T, for a total of
# time_steps uniformly distributed times.
T = 1.0  # final time
time_steps = N ** 2

# ------Begin main code.----------

# Space discretization
mesh, fn_space = pde_utils.setup_rectangular_function_space(nx, ny, P0, P1)

# Time discretization
dt, times = pde_utils.setup_time_discretization(T, time_steps)

RHS_fn = fc.Expression(
    '3*(-1 + x[0] + x[1]) - 3*(-1 + x[0] + x[1])*cos(3*t) + (3*(3*pow(x[0],2) - 2*pow(x[0],3) + (3 - '
    '2*x[1])*pow(x[1],2))*sin(3*t))/4', degree=2, t=0)
u_ref = fc.Expression('((-3*pow(x[0],2) + 2*pow(x[0],3) + pow(x[1],2)*(-3 + 2*x[1]))*(-1 + cos(3*t)))/4',
                      degree=2, t=0)

# Setup variational formulation, tying the LHS form with the trial function
# and the RHS form with the test functions and the RHS function.
u_initial = fc.interpolate(fc.Constant(0), fn_space)  # the solution is zero according to our problem formulation
u_trial = fc.TrialFunction(fn_space)
v_test = fc.TestFunction(fn_space)

u_previous = u_initial
LHS_int, RHS_int = pde_utils.variational_formulation(
    u_trial, v_test,
    LHS,
    RHS, RHS_fn,
    {'dt': dt},
    {'dt': dt, 'u_previous': u_previous}
)

t = times[0]  # initial time
fenics_list = []  # list of Fenics functions to be passed to the interpolators

# Solving
fc.set_log_active(False)  # disable messages of Fenics
for n in range(time_steps):
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
    print("Solving PDE, " + str(np.floor(100 * t)) + " % done.")

# Prepending initial condition
fenics_list.insert(0, fc.interpolate(fc.Constant(0), fn_space))

# Obtaining the interpolator for u
u = pde_IO.FenicsRectangleLinearInterpolator(nx, ny, P0, P1, fenics_list, time_dependent=True, verbose=True)
