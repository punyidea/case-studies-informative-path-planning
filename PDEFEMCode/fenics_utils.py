'''
Contains functions that (arguably) simplify FENICS' API.
Documentation assumes that fenics 2019.1.0 is used, and imported by
    import fenics as fc
'''
import fenics as fc
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass,field

from PDEFEMCode.interface import RectMeshParams, IOParams, VarFormulationParams, TimeDiscParams, EllipticRunParams, \
  ParabolicRunParams

@dataclass
class VarFormFnHandles:
    '''
    A class which stores all function handles used in the script.
    This is stored separately from other dataclasses to avoid
        a dependency on FEniCS when opening the pkl file.
    '''
    LHS: 'typing.Callable' # Function handle to the LHS form used in setting up the PDE.
    RHS: 'typing.Callable' # Function handle to the RHS form used in setting up the PDE.
    rhs_expression: 'typing.Callable' #Function handle to a function which generates a valid FEniCS expression, when necessary.
    def __init__(self,var_form_p):
        '''
        Get the function handles from a VarFormulationParams object.
        :param var_form_p: a VarFormulationParams object.
        '''

        self.LHS = get_function_from_str(var_form_p.LHS_form_str)
        self.RHS = get_function_from_str(var_form_p.RHS_form_str)

        if not var_form_p.rhs_expression_str:
            self.rhs_expression = get_function_from_str(var_form_p.rhs_expression_fn_str)
        if var_form_p.rhs_expression_str and var_form_p.rhs_expression_fn_str:
            Warning(
                'Both a fenics string and string specifying which function to '
                'use to construct the FEniCS expression string are defined.\n'
                'Function string is ignored.')
        if not (var_form_p.rhs_expression_str or var_form_p.rhs_expression_fn_str):
            raise Exception('No RHS function was provided, either in string generating '
                            'function or FEniCS expression string.')

def get_function_from_str(fname):
    '''
    See
    https://web.archive.org/web/20210614165007/https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string
    :param fname: A string version of the function which is inside of this file.
    :return: The function as a function object.
    '''
    try:
        fn = globals().copy()[fname]
        return fn
    except KeyError:
        raise ValueError('The function {} was not defined in the fenics_utils file.'.format(fname))


def post_process_parabolic_run_params(parabolic_p):
    '''
    Post processing on the parabolic run parameters. Default behavior is
    :param parabolic_p:  a ParabolicRunParams object.
    :return:
    '''
    if parabolic_p.time_disc.Nt == -1:
        parabolic_p.time_disc.Nt = parabolic_p.rect_mesh.nx*parabolic_p.rect_mesh.ny
    return parabolic_p

def yaml_parse_elliptic(par_obj,in_fname):
    '''
    Parses the YAML object given by PYYaml (a dictionary)
        to place it in the correct data structures.
    :param par_obj: The object returned by PYYAML library
    :param in_fname: The name of the yaml file it was read from. (stored in IOParams)
    :return: The (parsed and partially validated, through variable names) EllipticRunParams structure which allows elloptic_ex.py to run.
    '''

    par_prep ={}
    par_prep['var_form'] = VarFormulationParams(**par_obj['var_form'])
    par_prep['io'] = IOParams(in_file = in_fname, **par_obj['io'])
    par_prep['rect_mesh'] = RectMeshParams(**par_obj['rect_mesh'])

    ret_params = EllipticRunParams(**par_prep)
    return ret_params

def yaml_parse_parabolic(par_obj,in_fname):
    '''

    Parses the YAML object given by PYYaml (a dictionary)
        to place it in the correct data structures.
    :param par_obj: The object returned by PYYAML library
    :param in_fname: The name of the yaml file it was read from. (stored in IOParams)
    :return: The (parsed and partially validated, through variable names) ParabolicRunParams structure which allows elloptic_ex.py to run.

    :param par_obj:
    :param in_fname:
    :return:
    '''

    par_prep ={}
    par_prep['var_form'] = VarFormulationParams(**par_obj['var_form'])
    par_prep['io'] = IOParams(in_file=in_fname, **par_obj['io'])
    par_prep['rect_mesh'] = RectMeshParams(**par_obj['rect_mesh'])
    par_prep['time_disc'] = TimeDiscParams(**par_obj['time_disc'])

    ret_params = ParabolicRunParams(**par_prep)
    ret_params = post_process_parabolic_run_params(ret_params)
    return ret_params

def setup_unitsquare_function_space(n):
    """
    Sets up the function space on the unit square, with n subdivisions.
    This includes preparing the mesh and basis functions on the mesh.
    See https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos/built-in-meshes/demo_built-in-meshes.py.html
    for documentation on rectangular meshes.

    :return:
        - mesh, a fenics mesh on the unit square.
            The mesh is a unit square mesh with n sub-squares.
            Triangular subdivisions are obtained by going diagonally up and right,
            see UnitSquareMesh function documentation.
        - fn_space,  The FENICS function space on this mesh. The "hat function" is used as a basis.
    """
    # Create mesh and define function space
    mesh = fc.UnitSquareMesh(n, n)
    fn_space = fc.FunctionSpace(mesh, 'Lagrange', 1)
    return mesh,fn_space

def setup_rectangular_function_space(rmesh_p):
    """
    Sets up the dicrete function space on a rectangle with lower left corner P0, upper right corner P1.
    This includes preparing the mesh and basis functions on the mesh.

    :param rmesh_p: The RectMeshParams object which describes the structure of the rectangular mesh.
    :return:
        - mesh, a mesh on the rectangle with lower left corner P0, upper right corner P1.
            nx means a nx+1 uniform subdivision in the x direction, analogously for ny
            Hence we have a discretization of nx*ny rectangular cells, of 2*nx*ny triangles
            Triangular subdivisions are obtained by going diagonally up and right
            https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos/built-in-meshes/demo_built-in-meshes.py.html
        - fn_space, the FENICS function space of  linear FE on this mesh. The "hat functions" are used as a basis.
    """
    # Create mesh and define function space
    mesh = fc.RectangleMesh(fc.Point(rmesh_p.P0[0], rmesh_p.P0[1]),
                            fc.Point(rmesh_p.P1[0], rmesh_p.P1[1]),
                            rmesh_p.nx, rmesh_p.ny)
    fn_space = fc.FunctionSpace(mesh, 'Lagrange', 1)
    return mesh, fn_space


def setup_time_discretization(time_disc_p):
    """
    :param time_disc_p: The TimeDiscParams object which describes how the time space is discretised.
    :return:
        - dt: dt  = time_disc_p.T_fin/time_disc_p.Nt
        - times: a vector of Nt+1 evenly spaced time instants 0, dt, 2*dt, ... time_disc_p.T_fin
    """
    # Create mesh and define function space
    dt = 1 / time_disc_p.Nt
    times = np.linspace(0, time_disc_p.T_fin, time_disc_p.Nt + 1)
    return dt, times

def variational_formulation(u_trial, v_test, LHS, RHS, RHS_fn, LHS_args=None, RHS_args=None):
    '''

    :param u_trial: The trial function evaluated on the FENICS function space.
    :param v_test: The test function evaluated on the FENICS function space.
    :param LHS: a function that accepts at least 2 FENICS functions.
        If you have trial and test functions as defined below,
        it is evaluated on both. For an example see elliptic_LHS().
        If you need extra (named) arguments to your function,
            supply these through the RHS_args parameter, as a dictionary.
            (search "**kwargs" to see what this means)

    :param RHS: a function that accepts at least 2 FENICS functions,
        the functional form of the test functions, and some function RHS_fn.
        For an example, see elliptic_RHS().
        Extra arguments are supplied through RHS_args
    :param RHS_fn: a function defined on the FENICS function space, could be
            fc.Expression()
            fc.Constant()
    :param LHS/RHS_args: Dictionary containing extra argument names for the LHS and RHS functions.
    :return:
    '''
    LHS_args = LHS_args if LHS_args is not None else {}
    RHS_args = RHS_args if RHS_args is not None else {}

    # set up integrals using these forms.
    LHS_int = LHS(u_trial, v_test, **LHS_args)
    RHS_int = RHS(v_test, RHS_fn, **RHS_args)
    return LHS_int, RHS_int

def solve_vp(Fn_space, LHS_int, RHS_int, bc=None):
    '''
    Generates a FENICS solution function on the function space,
        then solves the PDE given in variational form by LHS_int, RHS_int, and boundary conditions bc.
    :param Fn_space: the function space of the PDE, according to FENICS.
    :param LHS_int: the variational integral form of the PDE's LHS evaluated on the function space
    :param RHS_int: the variational integral form of the PDE's RHS evaluated on the function space
        (see sample LHS and RHS functions in this file for the elliptic equation)
    :param bc: Boundary conditions of the PDE on the given function space/mesh.
    :return:
    '''
    u_sol = fc.Function(Fn_space)
    fc.solve(LHS_int == RHS_int, u_sol, bcs=bc)
    return u_sol


## BEGIN SAMPLE LHS and RHS functions.
def elliptic_LHS(u_trial, v_test, **kwargs):
    '''
    returns the LHS of the elliptic problem provided in the project handout:
    -\Delta u + u = f  => \int (grad(u) dot grad(v)  + u*x) dx = \int f*v dx
    :param u_trial: The trial function space defined on the FENICS Fn space.
        (You obtain this by calling u_trial = fc.TrialFunction(V))
    :param v_test: The test function space defined on the FENICS Fn space.
        (You obtain this by calling u_trial = fc.TestFunction(V))
    :return: an integral form of the equation, ready to be used in solve_pde.
    '''
    return (fc.dot(fc.grad(u_trial), fc.grad(v_test)) + u_trial * v_test) * fc.dx

def elliptic_RHS(v_test, RHS_fn, **kwargs):
    '''
    returns the RHS of the elliptic problem provided in the project handout:
    \Delta u + u = f  => \int (grad(u) dot grad(v)  + u*x) dx = \int f*v dx
    :param v_test: The test function space defined on the FENICS Fn space.
        (You obtain this by calling u_test = fc.TestFunction(V))
    :param RHS_fn: a FENICS function evaluated on the function space.
        (obtained by calling fc.
    :return: an integral form of the equation, ready to be used in solve_pde.
    '''
    return RHS_fn * v_test * fc.dx

def heat_eq_LHS(u_trial, v_test, dt=1, alpha=1):
    '''
    returns the LHS a(u_next, v) of the parabolic problem provided in the project handout:
    D_t u  - \alpha \Delta u = f, u(0)=0, homogeneous Neumann BC and initial conditions
    as discretized by means of the implicit Euler's method =>
    a(u_next, v) = \int u_trial * v dx + \int dt * alpha * dot(grad(u_trial), grad(v)) dx
    L(v) = (u_previous + dt * f) * v * dx

    If alpha=dt=1 returns the LHS of the elliptic problem provided in the project handout:
    -\Delta u + u = f  => \int (grad(u) dot grad(v)  + u*x) dx = \int f*v dx

    :param u_trial: The trial function space defined on the FENICS Fn space.
        (You obtain this by calling u_trial = fc.TrialFunction(V))
        Note: it is the solution of the current time step
    :param v_test: The test function space defined on the FENICS Fn space.
        (You obtain this by calling v_trial = fc.TestFunction(V))
    :param dt: time discretization mesh size. The default is 1, so that a stationary PDE can also be approximated
    :param alpha: a constant, default 1

    :return: an integral form of the equation, ready to be used in solve_pde.
    '''

    return (dt * alpha * fc.dot(fc.grad(u_trial), fc.grad(v_test)) + u_trial * v_test) * fc.dx

def heat_eq_RHS(v_test, RHS_fn, dt=1, u_previous=0):
    '''
    returns the RHS L(v) = (u_previous + dt * f) * v * dx of the parabolic problem provided in the project handout:
    D_t u  - \alpha \Delta u = f, u(0)=0, homogeneous Neumann BC and initial conditions
    as discretized by means of the implicit Euler's method =>
    a (u_next, v) = \int u_trial * v dx + \int dt * alpha * dot(grad(u_trial), grad(v)) dx
    L (v) = (u_previous + dt * f) * v * dx

    If dt=1, u_previous=0, returns the RHSof the elliptic problem provided in the project handout:
    \Delta u + u = f  => \int (grad(u) dot grad(v)  + u*x) dx = \int f*v dx

    :param u_previous: the FEM solution at the last time step (whereas u_triaL is at current time step). Default: 0
        (You obtain this by calling u_trial = fc.TestFunction(V))
    :param dt: time discretization mesh size. Default: 1
    :param v_test: The test function space defined on the FENICS Fn space.
        (You obtain this by calling v_trial = fc.TestFunction(V))
    :param RHS_fn: a FENICS function evaluated on the function space.
        (obtained by calling fc.
    :return: an integral form of the equation, ready to be used in solve_pde.
    '''

    return (dt * RHS_fn + u_previous) * v_test * fc.dx



def gaussian_expression_2D(gamma,u_max,r):
    '''
    Returns a fenics string which computes the gaussian in 2D.
    :param gamma: shape (2,) the center of the gaussian point source
    :param u_max: Height scaling parameter.
    :param r: radius parameter.
    :return: the string representation
    '''
    return '{} * exp(-(pow(x[0] - {},2) + pow(x[1] - {},2)) /pow({},2))'.format(u_max,gamma[0],gamma[1], r)

def sum_of_2d_gaussians(gamma_list,u_max_list,r_list):
    '''
    Returns a fenics expression that is the sum of gaussians
    :param gamma_list: a list of
    :param u_max_list: list of the scaling constants multiplying each gaussian
    :param r_list: radius parameter of each gaussian.
    :return: Example: sum_of_2d_gaussians([[0,0],[0,1], [.3,2],[0.05,0.1])
        returns the sum of: t
        -- 0.3 *(the gaussian point source centered at the origin with radius 0.05)
        -- 2 * (the gaussian point source centered at [0,1] with radius 0.1)

    '''
    assert(gamma_list.__len__()==u_max_list.__len__())
    assert(u_max_list.__len__()==r_list.__len__())
    return ' + '.join(gaussian_expression_2D(gamma,u_max,r) for gamma,u_max,r in zip(gamma_list,u_max_list,r_list))

def parabolic_double_bump_expr():
    '''
    Insert docstring here.
    This string is used for a parabolic RHS, since it includes parameter t.
     Generates a moving Gaussian bump with a non-differentiable path.
    :return: double_bump test case.
    '''
    fcase1 = '(exp(16)*(exp(-t + 1/(-0.0625 + pow(-0.5 + x[0] + cos(t)/4.,2) + pow(-0.5 + x[1] + sin(t)/4.,' \
             '2)))*(-1 + exp(t))*pow(cos((3*t)/2.),2) + exp(1/(-0.0625 + pow(-0.5 + x[0] - cos(t)/4.,' \
             '2) + pow(-0.5 + x[1] - sin(t)/4.,2)))*pow(sin((3*t)/2.),2)))/2. '
    fcase2 = '(exp(16 + 1/(-0.0625 + pow(-0.5 + x[0] - cos(t)/4.,2) + pow(-0.5 + x[1] - sin(t)/4.,2)))*pow(sin((' \
             '3*t)/2.),2))/2. '
    fcase3 = '(exp(16 - t + 1/(-0.0625 + pow(-0.5 + x[0] + cos(t)/4.,2) + pow(-0.5 + x[1] + sin(t)/4.,2)))*(-1 + ' \
             'exp(t))*pow(cos((3*t)/2.),2))/2. '
    fcase4 = '0'

    fcond11 = '2 + 4*pow(x[0],2) + 4*pow(x[1],2) + cos(t) + sin(t) < 2*(2*(x[0] + x[1]) + x[0]*cos(t) + x[1]*sin(' \
              't)) '
    fcond12 = '2*(1 + 2*pow(x[0],2) + 2*pow(x[1],2) + x[0]*cos(t) + x[1]*sin(t)) < 4*(x[0] + x[1]) + cos(t) + ' \
              'sin(t) '

    fcond1 = fcond11 + ' && ' + fcond12
    fcond2 = '2 + 4*pow(x[0],2) + 4*pow(x[1],2) + cos(t) + sin(t) < 2*(2*(x[0] + x[1]) + x[0]*cos(t) + x[1]*sin(t))'
    fcond3 = '2*(1 + 2*pow(x[0],2) + 2*pow(x[1],2) + x[0]*cos(t) + x[1]*sin(t)) < 4*(x[0] + x[1]) + cos(t) + sin(t)'


    fexpr = fcond1 + ' ? ' + fcase1 + ' :  (' + fcond2 + ' ? ' + fcase2 + ' :  ( ' + fcond3 + ' ? ' + fcase3 + ' : ' + fcase4 + ' ) )'
    return fexpr


def parabolic_non_smooth_expr():
    '''
    Insert docstring here.
    This string is used for a parabolic RHS, since it includes parameter t.
     Generates a moving Gaussian bump with a non-differentiable path.
    :return: non_smooth test case.
    '''
    xp1 = '0.375'
    xc1 = 't<=1||t>4'
    xp2 = '.671875'
    xc2 = '2<t && t<=4'
    xp3 = '.125 * (1+2 * t)'

    yp1 = '.671875'
    yc1 = '1<t && t<=2'
    yp2 = '.125 * (9-2 * t)'
    yc2 = '2<t && t<=3'
    yp3 = '.125 * (-3+2 * t)'
    yc3 = '3<t && t<=4'
    yp4 = '.125 * (3+2 * t)'

    y_traj = '(' + yc1 + ' ? ' + yp1 + ' :  (' + yc2 + ' ? ' + yp2 + ' :  ( ' + yc3 + ' ? ' + yp3 + ' : ' + yp4 + ' )))'
    x_traj = '(' + xc1 + ' ? ' + xp1 + ' :  ( ' + xc2 + ' ? ' + xp2 + ' : ' + xp3 + ' ))'

    b = 'exp(16 + 1/(-0.0625 + pow(x[0] - ' + x_traj + ',2) + pow(x[1] - ' + y_traj + ',2)))'
    c = 'pow(x[0] - ' + x_traj + ',2) + pow(x[1] - ' + y_traj + ',2) < 0.0625'

    fexpr = c + ' ? ' + b + ' : ' + '0'
    return fexpr


# Compute error in L2 norm
def error_L2(u_ref, u_sol):
    '''
    Returns the error between two FENICS objects.
    :param u_ref: the reference solution on the function space.
    :param u_sol:
    :return:
    '''
    return fc.errornorm(u_ref, u_sol, 'L2')

def error_H1(u_ref, u_sol):
    '''
    Returns the H1 error between two FENICS objects.
    :param u_ref: the reference solution on the function space.
    :param u_sol:
    :return:
    '''
    return fc.errornorm(u_ref, u_sol, 'H1')

def error_LInf_piece_lin(u_ref, u_sol, mesh):
    '''
    Computes the L infinity norm between reference and solution functions,
    in the case that the function is linear. (bc we can simply evaluate function at mesh points.)
    :param u_ref:
    :param u_sol:
    :param mesh:
    :return:
    '''
    # Compute maximum error at vertices
    vertex_values_u_D = u_ref.compute_vertex_values(mesh)
    vertex_values_u = u_sol.compute_vertex_values(mesh)
    return np.max(np.abs(vertex_values_u_D - vertex_values_u))


## Interpolating Functions

def fenics_unit_square_function_wrap(mesh, n, u_fenics):
    '''
    Wraps a fenics function object so that it may be called by a function which supplies numpy arrays.
    The wrapper performs bilinear interpolation between the given points, using scipy's RegularGridInterpolator.
    :param mesh: the fenics mesh object that we used.
    :param n: the number given to define the mesh size.
    :param u_fenics: the function to wrap in an interpolator
    :return: a function, which when evaluated,
        gives the function evaluated at coordinates.
    '''
    coords = mesh.coordinates().reshape((n + 1, n + 1, -1), order='F')
    interpolator_coords = coords[:, 0, 0], coords[0, :, 1]
    fn_vals = u_fenics.compute_vertex_values(mesh).reshape((n + 1, n + 1), order='F')
    return RegularGridInterpolator(interpolator_coords, fn_vals, method='linear')


## Convenient Fenics evaluation wrappers

def fenics_grad(mesh, u_fenics):
    '''
        Wraps a fenics function object so that it may be called by a function which supplies numpy arrays.
        :return: Fenics function which computes the gradient of the provided function (a discontinuous mesh)
    '''
    gradspace = fc.VectorFunctionSpace(mesh,'DG',0) #discontinuous lagrange.
    grad_fc = fc.project(fc.grad(u_fenics),gradspace)
    return grad_fc


