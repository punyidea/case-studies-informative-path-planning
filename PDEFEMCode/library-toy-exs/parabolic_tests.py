'''
This code runs one out of four test cases of the heat equation, on a 1x1 square mesh, from time t=0 to t=T=1
D different iterations are performed, for varying time/space discretization
These discretizations are chosen in such a way that the error in L2 norm should be at least linear (see end of file)

The error per iteration is saved into error_norm, where norm is infinity, L2 or H1 norm
'''

from fenics import *
import numpy as np
import matplotlib.pyplot as plt
set_log_level(50)

test_case = 4  # 1: bump, 2: polynomial, 3: easy, 4: polynomial easy
err = 'L2'  # error to be outputted ('uni', 'L2', 'H1')

# Discretization parameters
T = 1.0  # final time
D = 10 # how many times do we want to do solve our problem ?
time_array = np.ceil(np.logspace(4, 13, D, base=2.0)).astype(int)  # vector of total time steps, per time we do a time
# discretization
N_array = np.ceil(np.sqrt(time_array)).astype(int)  # vector of mesh sizes

# Test case specific definitions (right hand side, analytical solution)
if test_case == 3:
    f = interpolate(Constant(1), V)
    u_e = Expression('t', degree=2, t=0)
elif test_case == 2:
    f = Expression('3*(-1 + x[0] + x[1]) - 3*(-1 + x[0] + x[1])*cos(3*t) + (3*(3*pow(x[0],2) - 2*pow(x[0],3) + (3 - '
                   '2*x[1])*pow(x[1],2))*sin(3*t))/4', degree=2, t=0)
    u_e = Expression('((-3*pow(x[0],2) + 2*pow(x[0],3) + pow(x[1],2)*(-3 + 2*x[1]))*(-1 + cos(3*t)))/4', degree=2, t=0)
elif test_case == 1:
    fcase1 = '(exp(16 + 1/(-0.0625 + pow(-0.5 + x[0] - cos(t)/4.,2) + pow(-0.5 + x[1] - sin(t)/4.,2)))*sin((3*t)/2.)*(96*cos((3*t)/2.) + ((128*((-1 + 2*x[1])*cos(t) + sin(t) - 2*x[0]*sin(t)))/pow(2 - 4*x[0] + 4*pow(x[0],2) - 4*x[1] + 4*pow(x[1],2) + cos(t) - 2*x[0]*cos(t) + sin(t) - 2*x[1]*sin(t),2) - (7*pow(2 - 4*x[0] + cos(t),2) + pow(2 - 4*x[0] + cos(t),4) + pow(2 - 4*x[0] + cos(t),2)*pow(2 - 4*x[1] + sin(t),2) - 4*pow(2 - 4*x[0] + 4*pow(x[0],2) - 4*x[1] + 4*pow(x[1],2) + cos(t) - 2*x[0]*cos(t) + sin(t) - 2*x[1]*sin(t),2))/pow(-0.0625 + pow(-0.5 + x[0] - cos(t)/4.,2) + pow(-0.5 + x[1] - sin(t)/4.,2),4) - (7*pow(2 - 4*x[1] + sin(t),2) + pow(2 - 4*x[0] + cos(t),2)*pow(2 - 4*x[1] + sin(t),2) + pow(2 - 4*x[1] + sin(t),4) - 4*pow(2 - 4*x[0] + 4*pow(x[0],2) - 4*x[1] + 4*pow(x[1],2) + cos(t) - 2*x[0]*cos(t) + sin(t) - 2*x[1]*sin(t),2))/pow(-0.0625 + pow(-0.5 + x[0] - cos(t)/4.,2) + pow(-0.5 + x[1] - sin(t)/4.,2),4))*sin((3*t)/2.)))/64.'
    fcase2 = '0'
    fcond1 = '2 + 4*pow(x[0],2) + 4*pow(x[1],2) + cos(t) + sin(t) < 2*(2*(x[0] + x[1]) + x[0]*cos(t) + x[1]*sin(t))'
    fexpr = fcond1 + ' ? ' + fcase1 + ' : ' + fcase2
    f = Expression(fexpr, degree=6, t=0)

    ucase1 = '(exp(16 + 1/(-0.0625 + pow(-0.5 + x[0] - cos(t)/4.,2) + pow(-0.5 + x[1] - sin(t)/4.,2)))*pow(sin((3*t)/2.),2))/2.'
    ucase2 = '0'
    ucond1 = '2 + 4*pow(x[0],2) + 4*pow(x[1],2) + cos(t) + sin(t) < 2*(2*(x[0] + x[1]) + x[0]*cos(t) + x[1]*sin(t))'
    uexpr = ucond1 + ' ? ' + ucase1 + ' : ' + ucase2
    u_e = Expression(uexpr, degree=6, t=0)

elif test_case == 4:
    f = Expression('3*pow(x[0],2) - 2*pow(x[0],3) + (3 - 2*x[1])*pow(x[1],2) + 12*t*(-1 + x[0] + x[1])', degree=2, t=0)
    u_e = Expression('t*(3*pow(x[0],2) - 2*pow(x[0],3) + (3 - 2*x[1])*pow(x[1],2))', degree=2, t=0)
else:
    raise Exception('Test case not implemented')

# Errors tensor
err_tot = np.zeros(D)

# For loop computing errors
for disc in range(D):

    # Setting up the problem
    N = N_array[disc]
    mesh = RectangleMesh(Point(0, 0), Point(1, 1), N, N)
    V = FunctionSpace(mesh, 'P', 1)

    u_n = interpolate(Constant(0), V)
    u = TrialFunction(V)
    v = TestFunction(V)

    num_steps = time_array[disc]
    dt = T / num_steps  # time step size
    a = u * v * dx + dt * dot(grad(u), grad(v)) * dx
    L = (u_n + dt * f) * v * dx

    u = Function(V)
    t = 0  # initial time

    # Solving
    for n in range(num_steps):
        # Time update
        t += dt
        u_e.t = t
        f.t = t

        # Solution at this time (no boundary conditions!)
        solve(a == L, u)

        # Update previous solution
        u_n.assign(u)

    # Saving the error
    if err == 'uni':
        # Error in L^\infty norm
        u_ee = interpolate(u_e, V)
        error = np.abs(np.array(u_ee.vector()) - np.array(u.vector())).max()
        print('Step = % .2f' % disc)
    elif err == 'L2':
        error = errornorm(u_e, u, 'L2')
        print('Step = % .2f' % disc)
    else:
        error = errornorm(u_e, u, 'H1')
        print('Step = % .2f' % disc)

    err_tot[disc] = error

# loglog(time_array, err_tot) and loglog(time_array, 1/time_array) should be parallel, in the L2 norm
np.savetxt('error_'+err, err_tot)

