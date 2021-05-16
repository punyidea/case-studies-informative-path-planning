'''
This code runs one out of four test cases of the heat equation, on a 1x1 square mesh, from time t=0 to t=T=1
The error over time is computed, in infinity, L2 or H1 norm
'''

from fenics import *
import numpy as np
import matplotlib.pyplot as plt

test_case = 4  # 1: bump, 2: polynomial, 3: easy, 4: polynomial easy
err = 'uni'  # error to be outputted ('uni', 'L2', 'H1')

T = 1.0  # final time
num_steps = 100  # number of time steps
dt = T / num_steps  # time step size

# Create mesh and define function space
N = 100
nx = ny = N
mesh = RectangleMesh(Point(0, 0), Point(1, 1), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Null initial condition
u_n = interpolate(Constant(0), V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

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

# Solution computation
a = u * v * dx + dt * dot(grad(u), grad(v)) * dx
L = (u_n + dt * f) * v * dx

u = Function(V)
t = 0  # initial time
err_v = np.zeros(num_steps)  # no initial error

# Solving
for n in range(num_steps):
    # Time update
    t += dt
    u_e.t = t
    f.t = t

    # Solution at this time (no boundary conditions!)
    solve(a == L, u)

    # Printing the error
    if err == 'uni':
        # Error in L^\infty norm
        u_ee = interpolate(u_e, V)
        error = np.abs(np.array(u_ee.vector()) - np.array(u.vector())).max()
        print('t = % .2f: L^\infty error = % .3f' % (t, error))
    elif err == 'L2':
        error = errornorm(u_e, u, 'L2')
        print('t = % .2f: L^2 error = % .3f' % (t, error))
    else:
        error = errornorm(u_e, u, 'H1')
        print('t = % .2f: H^1 error = % .3f' % (t, error))

    err_v[n] = error

    # Update previous solution
    u_n.assign(u)

# Display the error
print("{} error over time: {}".format(err, err_v))