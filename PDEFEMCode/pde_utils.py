'''
Contains functions that (arguably) simplify FENICS' API.
Documentation assumes that fenics 2019.1.0 is used, and imported by
    import fenics as fc
'''
import fenics as fc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


### To remove?
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
    return mesh, fn_space


def setup_rectangular_function_space(nx, ny, P0, P1):
    """
    Sets up the dicrete function space on a rectangle with lower left corner P0, upper right corner P1.
    This includes preparing the mesh and basis functions on the mesh.

    :param P0: np.array of two coordinates, indicating the lower left rectangle of the mesh
    :param P1: np.array of two coordinates, indicating the upper right rectangle of the mesh

    :return:
        - mesh, a mesh on the rectangle with lower left corner P0, upper right corner P1.
            nx means a nx+1 uniform subdivision in the x direction, analogously for ny
            Hence we have a discretization of nx*ny rectangular cells, of 2*nx*ny triangles
            Triangular subdivisions are obtained by going diagonally up and right
            https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos/built-in-meshes/demo_built-in-meshes.py.html
        - fn_space, the FENICS function space of  linear FE on this mesh. The "hat functions" are used as a basis.
    """
    # Create mesh and define function space
    mesh = fc.RectangleMesh(fc.Point(P0[0], P0[1]), fc.Point(P1[0], P1[1]), nx, ny)
    fn_space = fc.FunctionSpace(mesh, 'P', 1)
    return mesh, fn_space


# # Define boundary condition
# u_D = Constant('-6')
#
# def boundary(x, on_boundary):
#     return on_boundary
#
# bc = DirichletBC(V, u_D, boundary)

def setup_time_discretization(T, Nt):
    """

    :param T: because we study the PDE in [0,T]
    :param Nt: for time discretization, we only look at a uniform dicretization of [0,T] of Nt+1 instants

    :return:
        - dt: dt  = T/Nt
        - times: a vector of Nt+1 evenly spaced time instants 0, dt, 2*dt, ... T
    """
    # Create mesh and define function space
    dt = 1 / Nt
    times = np.linspace(0, T, Nt + 1)
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
        TODO (if need be): change RHS definition to include u_form if need be?
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


def solve_pde(Fn_space, LHS_int, RHS_int, bc=None):
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
    returns the RHSof the elliptic problem provided in the project handout:
    \Delta u + u = f  => \int (grad(u) dot grad(v)  + u*x) dx = \int f*v dx
    :param v_test: The test function space defined on the FENICS Fn space.
        (You obtain this by calling u_trial = fc.TestFunction(V))
    :param RHS_fn: a FENICS function evaluated on the function space.
        (obtained by calling fc.
    :return: an integral form of the equation, ready to be used in solve_pde.
    '''
    return RHS_fn * v_test * fc.dx


def general_LHS(u_trial, v_test, dt=1, alpha=1):
    '''
    returns the LHS a(u_next, v) of the parabolic problem provided in the project handout:
    D_t u  - \alpha \Delta u = f, u(0)=0, Neumann BC
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


def general_RHS(v_test, RHS_fn, dt=1, u_previous=0):
    '''
    returns the RHS L(v) = (u_previous + dt * f) * v * dx of the parabolic problem provided in the project handout:
    D_t u  - \alpha \Delta u = f, u(0)=0, Neumann BC
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


def fenics_unit_square_function_wrap(mesh, n, u_fenics):
    '''
    Wraps a fenics function object so that it may be called by a function which supplies numpy arrays.
    The wrapper performs bilinear interpolation between the given points.
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


class fenics_rectangle_function_wrap():
    '''
    Wraps a fenics function object so that it may be called by a function which supplies numpy arrays.
    It is memory inefficient but runtime efficient: by means of linear interpolation, the query value is computed in
    8 operations
    :param mesh: the fenics mesh object that we used.
    :param nx, ny: number of side rectangular cells in x and y directions
    :param time_dependent. If True, we return the interpolator for the parabolic equation, for the elliptic otherwise.
        It's False by default.
    :param fem_data: the function to wrap in an interpolator if time_dependent = False,
        the ordered list of all such functions (as time varies), if time_dependent = True
    :return: a function, which when evaluated,
        gives the function evaluated at coordinates.
    '''

    def __init__(self, nx, ny, P0, P1, fem_data, time_dependent=False, verbose = False):
        self.nx = nx
        self.ny = ny
        self.x0 = P0[0]
        self.y0 = P0[1]
        self.x1 = P1[0]
        self.y1 = P1[1]
        self.hx = (P1[0] - P0[0]) / nx
        self.hy = (P1[1] - P0[1]) / ny
        self.D = self.hx * self.hy
        self.slope = self.hy / self.hx
        self.pad_v = np.array([self.nx - 1, self.ny - 1])
        self.zero = np.array([0, 0])
        self.time_dependent = time_dependent

        if not time_dependent:
            self.mesh = fem_data.function_space().mesh()
            u = fem_data
            self.T, self.Txy = self.pre_computations(u)
        else:
            self.mesh = fem_data[0].function_space().mesh()  # the mesh doesn't change with time
            # Let's make a list of the same tensors we built in the time independent case
            T_glo = []
            Txy_glo = []
            for i in range(len(fem_data)):

                u = fem_data[i]
                T, Txy = self.pre_computations(u)

                T_glo.append(T)
                Txy_glo.append(Txy)

                if verbose:
                    print('Building interpolator, ', 100 * (i + 1) / len(fem_data), ' % done.')

            self.T_glo = np.array(T_glo)
            self.Txy_glo = np.array(Txy_glo)

    def pre_computations(self, u):

        # Nodes of the mesh and nodal values of u. For i=0,...,nx and j=0,...,ny, we have the l-th entry l=i+j(nx+1)
        coords = self.mesh.coordinates()  # a 1+nx+ny(nx+1) x 2 matrix
        nodal_vals = u.compute_vertex_values()

        nx = self.nx
        ny = self.ny

        # Some grids
        self.x = np.linspace(self.x0, self.x1, nx + 1)
        self.y = np.linspace(self.y0, self.y1, ny + 1)

        # For triangles dw (dw = facing down)
        #    3
        #  2 1
        n_1_dw = 0  # Counter for 1 nodes of dw triangles
        n_2_dw = 0
        n_3_dw = 0
        u_1_dw = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))  # Nodal 1 vadwes of u for dw triangles
        u_2_dw = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))
        u_3_dw = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))
        xv_dw = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))  # Vertex coordinates for dw triangles
        yv_dw = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))

        # For triangles up (up = facing up)
        #  1 2
        #  3

        n_1_up = 0  # Counter for 1 nodes of up triangles
        n_2_up = 0
        n_3_up = 0
        u_1_up = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))  # Nodal 1 values of u for up triangles
        u_2_up = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))
        u_3_up = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))
        xv_up = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))  # Vertex coordinates for up triangles
        yv_up = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))
        self.U = np.zeros((nx + 1, ny + 1))  # array of U values

        for j in range(ny + 1):  # Pythonics for j=0...ny
            for i in range(nx + 1):  # Pythonics for i=0...nx

                # print([self.x0+i*self.hx, self.y0+j*self.hy], coords[i + j * (nx + 1)], nodal_vals[i + j * (nx + 1)], self.u(coords[i + j * (nx + 1)]))

                u_ij = nodal_vals[i + j * (nx + 1)]
                self.U[i, j] = u_ij

                if i > 0 and j < ny:  # Python wants and !!
                    u_1_dw[n_1_dw] = u_ij
                    n_1_dw += 1
                if i < nx and j < ny:
                    u_2_dw[n_2_dw] = u_ij
                    xv_dw[n_2_dw] = self.x0 + i * self.hx
                    yv_dw[n_2_dw] = self.y0 + j * self.hy
                    n_2_dw += 1
                    u_3_up[n_3_up] = u_ij
                    xv_up[n_3_up] = self.x0 + i * self.hx
                    yv_up[n_3_up] = self.y0 + j * self.hy
                    n_3_up += 1
                if i > 0 and j > 0:
                    u_3_dw[n_3_dw] = u_ij
                    n_3_dw += 1
                    u_2_up[n_2_up] = u_ij
                    n_2_up += 1
                if i < nx and j > 0:
                    u_1_up[n_1_up] = u_ij
                    n_1_up += 1

        # For dw triangles like
        #    3
        #  2 1

        T_dw = self.hx * self.hy * u_2_dw - self.hy * u_1_dw * xv_dw + self.hy * u_2_dw * xv_dw + self.hx * u_1_dw * yv_dw - self.hx * u_3_dw * yv_dw
        Tx_dw = self.hy * u_1_dw - self.hy * u_2_dw
        Ty_dw = - self.hx * u_1_dw + self.hx * u_3_dw

        # For triangles up (up = facing up)
        #  1 2
        #  3

        T_up = self.hx * self.hy * u_3_up + self.hy * u_1_up * xv_up - self.hy * u_2_up * xv_up - self.hx * u_1_up * yv_up + self.hx * u_3_up * yv_up
        Tx_up = - self.hy * u_1_up + self.hy * u_2_up
        Ty_up = self.hx * u_1_up - self.hx * u_3_up

        # All together
        T = np.array([T_dw, T_up]).T
        Tx = np.array([Tx_dw, Tx_up]).T
        Ty = np.array([Ty_dw, Ty_up]).T
        Txy = np.array([Tx, Ty])  # In Pythonics, these is a tensor. The first index let's us switch between Px, Py

        return T, Txy

    def get_interpolator_elliptic(self, M):
        '''
        It takes in a matrix of N points (Nx2)
        For every point it computes the triangle type and triangle index
        We build an Nx2 matrix, the first column is for the index, the second is 0 for dw triangle, 1 for up triangles
        We finally compute and return the interpolated values
        '''

        if len(np.shape(M)) == 1:
            M = np.array([M])

        index_raw, type_raw = np.divmod(M - [self.x0, self.y0], [self.hx, self.hy])
        # Ensuring the query index is admissible (and at the same time: being able to input any point we want)
        index_raw = np.maximum(np.minimum(index_raw, self.pad_v), self.zero)

        index_def = np.squeeze((index_raw[:, 0] + index_raw[:, 1] * self.nx)).astype(int)
        type_def = np.squeeze(type_raw[:, 0] * self.slope < type_raw[:, 1]).astype(int)  # If 0, dw triangle

        # Interpolation (only this has to be repeated in case of a time dependent pde)
        Pt = self.Txy[:, index_def, type_def]
        P = np.sum(Pt * M, axis=1)
        N = self.T[index_def, type_def] + P

        return N / self.D

    def get_interpolator_parabolic(self, M, It):
        '''
        It takes in a matrix of N points (Nx2) and an array of time indices (Tx1). It is a subset of 0:len(fem_data)
        For every point and every time it computes the triangle type and triangle index
        We build a Nx2 matrix, the first column is for the index, the second is 0 for dw triangle, 1 for up triangles
        We return a matrix that is NxT (N spatial interpolations at every time)
        '''

        if len(np.shape(M)) == 1:
            M = np.array([M])

        index_raw, type_raw = np.divmod(M - [self.x0, self.y0], [self.hx, self.hy])
        # Ensuring the query index is admissible (and at the same time: being able to input any point we want)
        index_raw = np.maximum(np.minimum(index_raw, self.pad_v), self.zero)

        index_def = np.squeeze((index_raw[:, 0] + index_raw[:, 1] * self.nx)).astype(int)
        type_def = np.squeeze(type_raw[:, 0] * self.slope < type_raw[:, 1]).astype(int)  # If 0, dw triangle

        # Interpolation (only this has to be repeated in case of a time dependent pde)
        Pt = self.Txy_glo[It, :, index_def, type_def]
        P = np.sum(Pt * M, axis=-1) # the last axis
        N = self.T_glo[It, index_def, type_def] + P

        return N / self.D

    def get_interpolator(self):
        if not self.time_dependent:
            return self.get_interpolator_elliptic
        else:
            return self.get_interpolator_parabolic

    def get_scipy_interpolator(self):
        return RegularGridInterpolator((self.x, self.y), self.U)

# # Print errors
# print('error_L2  =', error_L2)
# print('error_max =', error_max)
#
# # Hold plot
# plt.show()
