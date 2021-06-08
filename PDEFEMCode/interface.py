import os, pickle

import numpy as np


def pickle_save(out_path, fname, obj_save, ext='.pkl'):
    '''
    Saves parameter
    :param fname:name of the file to save to. IGNORES EXTENSION.
    :param obj_save: the Python object to save. Prefer that this is a dictionary with objects to save.

    '''
    fname, _ = os.path.splitext(fname)  # only keep root part
    with open(os.path.join(out_path, fname + ext), 'wb') as f_obj:
        pickle.dump(obj_save, f_obj)


def pickle_load(fname, def_ext='.pkl'):
    '''

    :param fname: full path of the file to load using pickle.
    :param def_ext: default file extension used to save files.
    :return: the Python object. This should be a dictionary if it has more than one parameter.
    '''
    fname, ext = os.path.splitext(fname)
    if not ext:
        ext = def_ext
    with open(fname + ext, 'rb') as f_obj:
        return pickle.load(f_obj)


class RectangleInterpolator():
    '''
    This is an abstract class that stores common parameters of the gradient interpolator.
    '''

    def __init__(self, nx, ny, P0, P1, fem_data, T=None, Nt=None, time_dependent=False, for_optimization=True, verbose=False):
        # Some geometric initializations
        pass
        self.nx = nx
        self.ny = ny
        self.x0 = P0[0]
        self.y0 = P0[1]
        self.x1 = P1[0]
        self.y1 = P1[1]
        self.hx = (P1[0] - P0[0]) / nx
        self.hy = (P1[1] - P0[1]) / ny

        # Some temporal initializations
        if time_dependent and T is None or Nt is None:
            raise Exception('Please insert the final time and the number of time intervals.')
        else:
            self.T=T
            self.Nt=Nt

        # Variable deciding if the returned interpolator will have some additional functionalities or not
        self.for_optimization = for_optimization

        # Now, we enlarge the mesh by one cell in every direction, to handle boundary points well
        self.enx = self.nx + 2
        self.eny = self.ny + 2
        self.ex0 = P0[0]-self.hx
        self.ey0 = P0[1]-self.hy
        self.ex1 = P1[0]+self.hx
        self.ey1 = P1[1]+self.hy

        # To differentiate which interpolator to return. We now generate helping variables that will make interpolation
        # faster later on
        self.time_dependent = time_dependent
        self.verbose = verbose

        self.pre_computations(fem_data)

    def pre_computations(self):
        pass

    def __call__(self, coords):
        raise Exception('Abstract class. No call method implemented.')


class FenicsRectangleLinearInterpolator(RectangleInterpolator):
    '''
    Given the fenics solution to the time (in)dependent PDE of the project, it wraps a function object around it.
    This new interpolator function can be later called without the need of having fenics installed.
    This code can more generally work on a single fenics function, or a "time" list of fenics functions.


    NOTE: DEPENDENT ON USING RECTANGULAR MESH WITH DOWN/RIGHT diagonals.
    ASSUMES THE FENICS FUNCTION(s) ARE PIECEWISE AFFINE ON MESH ELEMENTS
    IF A QUERY POINT (x,y) IS OUTSIDE [a,b]x[c,d], (x,y) IS TRANSFORMED INTO (clip(x,a,c), clip(y,c,d))

    Use case:
        my_interp_time = pde_IO.FenicsRectangleLinearInterpolator(nx, ny, P0, P1, fenics_list, time_dependent=True)
        my_interp_stat = pde_IO.FenicsRectangleLinearInterpolator(nx, ny, P0, P1, fenics_list, time_dependent=False)
    my_interp now accepts an Nx2 numpy array of query points, where the solution of the pde is evaluated. If the
    solution is time dependent, a time array can also be specified: the solution will be interpolated at the query
    points for every desired time instant. The time array is an ordered list of indices (0 indicates the initial
    time, 1 the time immediately after and so on).
        my_interp_time(np.array([0,3]),[0,2,7])
        my_interp_stat(np.array([0,3]))
    In general, the output of my_interp_time is an NxNt matrix, of N evaluations in space for every Nt time instants
    TODO (leo): fix documentation
    Input variables.
    :param mesh: the rectangular fenics mesh object that we used. It is supposed to be composed of uniform rectangular
    cells, divided into triangles by the down-left to up-right diagonal
    :param P0, P1: two 2d numpy arrays, specifying upper right and lower left corners of the rectangular domain
    :param nx, ny: number of side rectangular cells in x and y directions
    :param time_dependent. If True, we return the interpolator for the parabolic equation, for the elliptic otherwise.
        It's False by default.
    :param fem_data: the fenics function to wrap in the interpolator if time_dependent = False,
        the ordered list of all such functions (as time varies), if time_dependent = True
    :param verbose: if True, it displays the building status of the time dependent interpolator
    :return: a function, which when evaluated, gives the function evaluated at the desired space-time coordinates.
    '''

    def pre_computations(self, fem_data):

        # All possible time indices
        if self.time_dependent:
            self.t_ind_all = np.arange(0, self.Nt+1).tolist()
            self.dt = 1 / self.Nt

        # Other temporary variables for the interpolation process
        self.D = self.hx * self.hy
        self.slope = self.hy / self.hx

        # Used to handle query points outside the original domain
        self.max_pad = np.array([self.enx - 2, self.eny - 2])
        self.min_pad = np.array([1, 1])

        # To differentiate which interpolator to return. We now generate helping variables that will make interpolation
        # faster later on
        if not self.time_dependent:
            u = fem_data
            self.T, self.Tx, self.Ty = self.pre_computations_mat(
                u)  # some helping variables to speed up the interpolation later on
        else:
            # Let's make a list of the helping tensors we built in the time independent case
            T_glo = []
            Tx_glo = []
            Ty_glo = []
            for i in range(len(fem_data)):

                u = fem_data[i]
                T, Tx, Ty = self.pre_computations_mat(u)

                T_glo.append(T)
                Tx_glo.append(Tx)
                Ty_glo.append(Ty)

                if self.verbose:
                    print('Building function interpolator, ', str(np.floor(100 * (i + 1) / len(fem_data))), ' % done.')

            self.T_glo = np.array(T_glo)
            self.Tx_glo = np.array(Tx_glo)
            self.Ty_glo = np.array(Ty_glo)

    def pre_computations_mat(self, u):
        '''
        It receives a fenics (non time dependent) function and computes some related helping variables to speed up
        the interpolation process later on.
        These are T and Tx, Ty.
        For detailed functioning, refer to the pdf of the tecnical documentation.
        '''

        # For not writing self every time. Working with the enlarged mesh.
        nx = self.enx
        ny = self.eny

        # For triangles dw (dw = facing down)
        #    3
        #  2 1
        n_1_dw = 0  # Counter for 1 nodes of dw triangles
        n_2_dw = 0
        n_3_dw = 0
        u_1_dw = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))  # Nodal 1 values of u for dw triangles
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

        # This will build different arrays useful for computing T, Txy. See the technical documentation for details
        for j in range(ny + 1):  # Pythonics for j=0...ny
            for i in range(nx + 1):  # Pythonics for i=0...nx

                x = self.ex0 + i*self.hx
                y = self.ey0 + j*self.hy

                # If my query point is outside the original mesh, set u to zero. This won't affect the final result,
                # as everything outside the original mesh is squashed onto the rectangle boundary. The interpolation
                # then becomes a line interpolation of known u values
                if x > self.x1+1e-12 or x < self.x0-1e-12 or y < self.y0-1e-12 or y > self.y1+1e-12:
                    u_ij = 0
                else:
                    u_ij = u([x, y])

                if i > 0 and j < ny:
                    u_1_dw[n_1_dw] = u_ij
                    n_1_dw += 1
                if i < nx and j < ny:
                    u_2_dw[n_2_dw] = u_ij
                    xv_dw[n_2_dw] = self.ex0 + i * self.hx
                    yv_dw[n_2_dw] = self.ey0 + j * self.hy
                    n_2_dw += 1
                    u_3_up[n_3_up] = u_ij
                    xv_up[n_3_up] = self.ex0 + i * self.hx
                    yv_up[n_3_up] = self.ey0 + j * self.hy
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

        # All together: computing T, Tx, Ty
        T = np.array([T_dw, T_up]).T
        Tx = np.array([Tx_dw, Tx_up]).T
        Ty = np.array([Ty_dw, Ty_up]).T

        return T, Tx, Ty

    def get_interpolator_elliptic(self, M):
        '''
        It takes in a numpy 2d array of N points (Nx2) (or also a single 2D point, a numpy 1d array)
        It returns the interpolated values at the query points.
        For details about the functioning refer to the technical documentation.
        '''

        # If a single query point
        if len(np.shape(M)) == 1:
            M = np.array([M])

        # Getting the index of the rectangular cell and the type of the triangle, while also ensuring the query index
        # is admissible (and at the same time: being able to input any point we want)
        index_raw, type_raw = np.divmod(M - [self.ex0, self.ey0], [self.hx, self.hy])
        index_raw = np.clip(index_raw, self.min_pad, self.max_pad)
        index_def = np.squeeze((index_raw[:, 0] + index_raw[:, 1] * self.enx)).astype(int)
        type_def = np.squeeze(type_raw[:, 0] * self.slope < type_raw[:, 1]).astype(int)  # If 0, dw triangle

        # Interpolation
        Px = self.Tx[index_def, type_def] * M[:, 0]
        Py = self.Ty[index_def, type_def] * M[:, 1]
        N = self.T[index_def, type_def] + Px + Py

        return np.squeeze(N / self.D)

    def get_interpolator_parabolic_dev(self, M, times=None, optimization_mode=False):
        '''
        TODO (leo); add documentation
        '''

        # If a single query point
        M=np.array(M, ndmin=2)
        if times is None:
            # We condider exactly all the timestamps from the heat equation simulation
            t_ind = self.t_ind_all # all time indices
        else:
            # If a single query time
            times = np.array(times, ndmin=1)
            t_ind = np.clip(np.round(times/self.dt), 0, self.Nt).astype(int).tolist()

        # Getting the index of the rectangular cell and the type of the triangle, while also ensuring the query index
        # is admissible (and at the same time: being able to input any point we want)
        index_raw, type_raw = np.divmod(M - [self.ex0, self.ey0], [self.hx, self.hy])
        index_raw = np.clip(index_raw, self.min_pad, self.max_pad)
        index_def = np.squeeze((index_raw[:, 0] + index_raw[:, 1] * self.enx)).astype(int)
        type_def = np.squeeze(type_raw[:, 0] * self.slope < type_raw[:, 1]).astype(int)  # If 0, dw triangle

        # Interpolation (time dependent version)
        if optimization_mode:
            # If this interpolator is run in optimization mode, then M must be of size times and we
            # evaluate at timestamp i, the interpolator at point i
            row_indices = t_ind
        else:
            row_indices = np.array(t_ind)[:, None]
        Px = self.Tx_glo[row_indices, index_def, type_def] * M[:, 0]
        Py = self.Ty_glo[row_indices, index_def, type_def] * M[:, 1]
        N = self.T_glo[row_indices, index_def, type_def] + Px + Py

        return np.squeeze(N / self.D)

    def get_interpolator_parabolic_opt(self, M, t_ind=None):
        '''
        TODO (leo); add documentation
        '''

        # If a single query point
        if len(np.shape(M)) == 1:
            M = np.array([M])

        # Getting the index of the rectangular cell and the type of the triangle, while also ensuring the query index
        # is admissible (and at the same time: being able to input any point we want)
        index_raw, type_raw = np.divmod(M - [self.ex0, self.ey0], [self.hx, self.hy])
        index_raw = np.clip(index_raw, self.min_pad, self.max_pad)
        index_def = np.squeeze((index_raw[:, 0] + index_raw[:, 1] * self.enx)).astype(int)
        type_def = np.squeeze(type_raw[:, 0] * self.slope < type_raw[:, 1]).astype(int)  # If 0, dw triangle

        # Interpolation (time dependent version)
        if t_ind is None:
            row_indices = self.t_ind_all
            # And M must have Nt+1 rows
        else:
            # Handling a single time
            if isinstance(t_ind, int):
                t_ind = [t_ind]
            row_indices = np.array(t_ind)[:, None]
        Px = self.Tx_glo[row_indices, index_def, type_def] * M[:, 0]
        Py = self.Ty_glo[row_indices, index_def, type_def] * M[:, 1]
        N = self.T_glo[row_indices, index_def, type_def] + Px + Py
        return np.squeeze(N / self.D)

    def __call__(self, *args, **kwargs):
        '''
        Returns the interpolator function, either time dependent or independent.
        '''
        if not self.time_dependent:
            return self.get_interpolator_elliptic(*args, **kwargs)
        else:
            if self.for_optimization:
                return self.get_interpolator_parabolic_opt(*args, **kwargs)
            else:
                return self.get_interpolator_parabolic_dev(*args, **kwargs)


class FenicsRectangleVecInterpolator(RectangleInterpolator):
    '''
    Wraps a fenics vector function object (gradient of function) so that it may be called by a
    function which supplies numpy arrays.
    It is memory inefficient but runtime efficient.
    NOTE: DEPENDENT ON USING RECTANGULAR MESH WITH UP/RIGHT diagonals.
    ASSUMES THE FUNCTION IS PIECEWISE CONSTANT ON MESH ELEMENTS (i.e. fn_space, 'DG',0),
        like the gradient of a function on Lagrangian "hat" elements.
    TODO: (Victor) Mention behavior outside of bounds.

    Variables.
    :param mesh: the fenics mesh object that we used.
    :param nx, ny: number of side rectangular cells in x and y directions
    :param P0: Minimum coordinates (x, then y) of the bounding box of the mesh
    :param P1: Maximum coordinates (x, then y) of the bounding box
    :param u_fenics: the  fenics Vector function (piecewise constant) to wrap in an interpolator.
    :return: an object, that when called, computes the gradient of the provided function (a discontinuous mesh)


    Above, coords is shape (don't care x 2), to imitate shape of native fenics caller.
    #CALL VARS (coords). See __call__()

    Use case:
        u_grad      = fenics_grad(mesh,u_fenics)
        grad_fn     = FenicsRectangleGradInterpolator(nx, ny, P0, P1, u_grad)
        grad_eval   = grad_fn(coords)
    '''

    def pre_computations(self, vec_u):
        nx, ny = self.nx, self.ny
        self.x = np.linspace(self.x0, self.x1, nx + 1)
        self.y = np.linspace(self.y0, self.y1, ny + 1)
        grad_u = vec_u

        # compute min coords of each cell
        low_cell_X, low_cell_Y = np.meshgrid(self.x[:-1], self.y[:-1], indexing='ij')
        low_cell_coords = np.stack((low_cell_X, low_cell_Y), axis=-1)

        # upper triangle: ("top left" of each cell)
        # Lower triangle: ("bottom right" of each cell)
        # example of UL:
        #  1 2
        #  3
        BR_coords = low_cell_coords + [.5 * self.hx, .25 * self.hy]
        UL_coords = low_cell_coords + [.25 * self.hx, .5 * self.hy]
        T_coords = np.stack((BR_coords, UL_coords), axis=-2)

        self.T_grads = native_fenics_eval_vec(grad_u, T_coords)

    def __call__(self, coords):
        '''
        TODO: (Victor) add time dependent solution, with a simple round.
        When called, returns the gradient of the point on the mesh.
        :param coords: coordinates on which we'd like to shape (don't care) by 2
        :return: gradient of the interpolator, shape  (coords.shape[:-1} x2)
        '''
        coords_shape = coords.shape
        coords_rs = np.reshape(coords, (-1, coords_shape[-1]))

        index_float = (coords_rs - [self.x0, self.y0]) / [self.hx, self.hy]
        eps = 1e-1
        index_float = np.clip(index_float, [0, 0], [self.nx - eps, self.ny - eps])
        # assert((index_float<=[self.nx,self.ny]).all())# are we in bounds?
        # assert((index_float>=0).all()) #are we in bounds?

        index_int, frac_index = np.divmod(index_float, 1)
        index_int = index_int.astype(int)

        # if type_def is 0, it is a BR_triangle
        type_def = np.squeeze(frac_index[:, 0] < frac_index[:, 1]).astype(int)  # If 0, dw triangle

        out_arr = self.T_grads[index_int[:, 0], index_int[:, 1], type_def, :]
        out_shape = coords_shape[:-1] + (2,)
        return out_arr.reshape(out_shape)


def native_fenics_eval_scalar(u_fenics, coords):
    '''
    Natively evaluate a function using fenics' evaluator.
    :param u_fenics: fenics function.
    :param coords: points such that shape.coords[-1] is the number of dimensions of the function space.
        NOTE! Coords is not in separate (X,Y) form.
    :return:
    '''
    out_arr_shape = coords.shape[:-1]
    coords_reshape = coords.reshape((-1, coords.shape[-1]))
    out_arr = np.empty(coords_reshape.shape[0])
    for ind, coord in enumerate(coords_reshape):
        u_fenics.eval(out_arr[ind:ind + 1], coord)  # slice used here for 1d case.
    return out_arr.reshape(out_arr_shape)


def native_fenics_eval_vec(vec_fenics, coords):
    '''
    Natively evaluate a function using fenics' evaluator.
    :param vec_fenics: fenics vector function with output dimension dim_out.
    :param dim_out: a
    :param coords: points such that shape.coords[-1] is the number of dimensions of the function space.
        NOTE! Coords is not in separate (X,Y) form.
    :return: a shape with the same dimension as coords, but where the last index contains all dims of the fenics output.
    '''
    dim_out = vec_fenics.value_dimension(0)
    out_arr_shape = coords.shape[:-1] + (dim_out,)  # plus is Python concatenation of tuples.
    coords_reshape = coords.reshape((-1, coords.shape[-1]))
    out_arr = np.empty((coords_reshape.shape[0], dim_out))
    for ind, coord in enumerate(coords_reshape):
        vec_fenics.eval(out_arr[ind], coord)
    return out_arr.reshape(out_arr_shape)
