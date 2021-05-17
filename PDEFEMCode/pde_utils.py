'''
Contains functions that (arguably) simplify FENICS' API.
Documentation assumes that fenics 2019.1.0 is used, and imported by
    import fenics as fc
'''
import fenics as fc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def setup_function_space(n):
    """
    Sets up the function space on the unit square, with n subdivisions.
    This includes preparing the mesh and basis functions on the mesh.

    :return:
        - mesh, a fenics mesh on the unit square. 
            The mesh is a unit square mesh with n sub-squares.
            Triangular subdivisions are obtained by going diagonally up and right,
            see UnitSquareMesh function documentation.
        - fn_space,  The FENICS function space on this mesh. The "hat function" is used as a basis.
    """
    # Create mesh and define function space
    mesh = fc.UnitSquareMesh(n, n)
    fn_space = fc.FunctionSpace(mesh, 'P', 1)
    return mesh,fn_space

# # Define boundary condition
# u_D = Constant('-6')
#
# def boundary(x, on_boundary):
#     return on_boundary
#
# bc = DirichletBC(V, u_D, boundary)

def variational_formulation(u_trial,v_test,LHS, RHS,RHS_fn, LHS_args = None, RHS_args=None):
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
        TODO (if need be): change RHS definition to include u_form if need be?
    :param RHS_fn: a function defined on the FENICS function space, could be
            fc.Expression()
            fc.Constant()
    :param LHS/RHS_args: Dictionary containing extra argument names for the LHS and RHS functions.
    :return:
    '''
    LHS_args = LHS_args if LHS_args is not None else {}
    RHS_args = RHS_args if RHS_args is not None else {}

    #set up integrals using these forms.
    LHS_int = LHS(u_trial,v_test, **LHS_args)
    RHS_int = RHS(v_test,RHS_fn, **RHS_args)
    return LHS_int,RHS_int


def solve_pde(Fn_space,LHS_int,RHS_int,bc=None):
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
    fc.solve(LHS_int == RHS_int,u_sol,bcs=bc)
    return u_sol

## BEGIN SAMPLE LHS and RHS functions.
def elliptic_LHS(u_trial,v_test,**kwargs):
    '''
    returns the LHS of the elliptic problem provided in the project handout:
    -\Delta u + u = f  => \int (grad(u) dot grad(v)  + u*x) dx = \int f*v dx
    :param u_trial: The trial function space defined on the FENICS Fn space.
        (You obtain this by calling u_trial = fc.TrialFunction(V))
    :param v_test: The test function space defined on the FENICS Fn space.
        (You obtain this by calling u_trial = fc.TestFunction(V))
    :return: an integral form of the equation, ready to be used in solve_pde.
    '''
    return (fc.dot(fc.grad(u_trial), fc.grad(v_test)) + u_trial*v_test)*fc.dx

def elliptic_RHS(v_test,RHS_fn,**kwargs):
    '''
    returns the RHSof the elliptic problem provided in the project handout:
    \Delta u + u = f  => \int (grad(u) dot grad(v)  + u*x) dx = \int f*v dx
    :param v_test: The test function space defined on the FENICS Fn space.
        (You obtain this by calling u_trial = fc.TestFunction(V))
    :param RHS_fn: a FENICS function evaluated on the function space.
        (obtained by calling fc.
    :return: an integral form of the equation, ready to be used in solve_pde.
    '''
    return RHS_fn*v_test*fc.dx


# Compute error in L2 norm
def error_L2(u_ref,u_sol):
    '''
    Returns the error between two FENICS objects.
    :param u_ref: the reference solution on the function space.
    :param u_sol:
    :return:
    '''
    return fc.errornorm(u_ref, u_sol, 'L2')

def error_H1(u_ref,u_sol):
    '''
    Returns the H1 error between two FENICS objects.
    :param u_ref: the reference solution on the function space.
    :param u_sol:
    :return:
    '''
    return fc.errornorm(u_ref,u_sol,'H1')

def error_LInf_piece_lin(u_ref,u_sol,mesh):
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


def fenics_unit_square_function_wrap(mesh,n,u_fenics):
    '''
    Wraps a fenics function object so that it may be called by a function which supplies numpy arrays.
    The wrapper performs bilinear interpolation between the given points.
    :param mesh: the fenics mesh object that we used.
    :param n: the number given to define the mesh size.
    :param u_fenics: the function to wrap in an interpolator
    :return: a function, which when evaluated,
        gives the function evaluated at coordinates.
    '''
    coords = mesh.coordinates().reshape((n+1,n+1,-1),order='F')
    interpolator_coords = coords[:,0,0],coords[0,:,1]
    fn_vals = u_fenics.compute_vertex_values(mesh).reshape((n+1,n+1),order='F')
    return RegularGridInterpolator(interpolator_coords,fn_vals,method='linear')


def fenics_grad_wrap(u_fenics):
    '''
    Wraps a fenics function object so that it may be called by a function which supplies numpy arrays.
    :return: a function, which when evaluated,
        gives the gradient of the function evaluated at coordinates.
    TODO: document function shape.
    '''
    pass
# # Print errors
# print('error_L2  =', error_L2)
# print('error_max =', error_max)
#
# # Hold plot
# plt.show()