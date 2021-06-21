# class FenicsRectangleLinearInterpolator(RectangleInterpolator):
#     '''
#     Given the fenics solution to the time (in)dependent PDE, it wraps a function object around it. This new interolator
#     function can be later called without the need of having fenics installed.
#     To obtain the interpolator, the user must run something like:
#     wrap = pde_utils.fenics_rectangle_function_wrap(nx, ny, P0, P1, u_fenics)
#     my_interp = wrap.get_interpolator()
#     my_interp now accepts an Nx2 numpy array of query points, where the solution of the pde is evaluated. If the
#     solution is time dependent, a time array can also be specified: the solution will be interpolated at the query
#     points for every desired time instance. The time array is an order list of indices (0 indicates the initial
#     time, 1 the time immediately after and so on).
#
#     :param mesh: the rectangular fenics mesh object that we used. It is supposed to be composed of uniform rectangular
#     cells, divided into triangles by the down-left to up-right diagonal
#     :param P0, P1: two 2d numpy arrays, specifying upper right and lower left corners of the rectangular domain
#     :param nx, ny: number of side rectangular cells in x and y directions
#     :param time_dependent. If True, we return the interpolator for the parabolic equation, for the elliptic otherwise.
#         It's False by default.
#     :param fem_data: the fenics function to wrap in the interpolator if time_dependent = False,
#         the ordered list of all such functions (as time varies), if time_dependent = True
#     :param verbose: if True, it displays the building status of the time dependent interpolator
#     :return: a function, which when evaluated, gives the function evaluated at the desired space-time coordinates.
#     '''
#
#     def pre_computations(self, fem_data):
#
#         # Other temporary variables for the interpolation process
#         self.D = self.hx * self.hy
#         self.slope = self.hy / self.hx
#
#         # Used to handle query points outside the domain
#         self.pad_v = np.array([self.nx - 1, self.ny - 1])
#         self.zero = np.array([0, 0])
#
#         # To differentiate which interpolator to return. We now generate helping variables that will make interpolation
#         # faster later on
#
#         if not self.time_dependent:
#             u = fem_data
#             self.T, self.Tx, self.Ty = self.pre_computations_mat(
#                 u)  # some helping variables to speed up the interpolation later on
#         else:
#             # Let's make a list of the helping tensors we built in the time independent case
#             T_glo = []
#             Tx_glo = []
#             Ty_glo = []
#             for i in range(len(fem_data)):
#
#                 u = fem_data[i]
#                 T, Tx, Ty = self.pre_computations_mat(u)
#
#                 T_glo.append(T)
#                 Tx_glo.append(Tx)
#                 Ty_glo.append(Ty)
#
#                 if self.verbose:
#                     print('Building function interpolator, ', str(np.floor(100 * (i + 1) / len(fem_data))), ' % done.')
#
#             self.T_glo = np.array(T_glo)
#             self.Tx_glo = np.array(Tx_glo)
#             self.Ty_glo = np.array(Ty_glo)
#
#     def pre_computations_mat(self, u):
#         '''
#         It receives a fenics (non time dependent) function and computes some related helping variables to speed up
#         the interpolation process later on.
#         These are T and Tx, Ty.
#         For detailed functioning, refer to the pdf of the tecnical documentation.
#         '''
#
#         # Nodal values of u. For i=0,...,nx and j=0,...,ny, we have the l-th entry l=i+j(nx+1)
#         nodal_vals = u.compute_vertex_values()
#
#         # For not writing self every time
#         nx = self.nx
#         ny = self.ny
#
#         # For triangles dw (dw = facing down)
#         #    3
#         #  2 1
#         n_1_dw = 0  # Counter for 1 nodes of dw triangles
#         n_2_dw = 0
#         n_3_dw = 0
#         u_1_dw = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))  # Nodal 1 values of u for dw triangles
#         u_2_dw = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))
#         u_3_dw = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))
#         xv_dw = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))  # Vertex coordinates for dw triangles
#         yv_dw = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))
#
#         # For triangles up (up = facing up)
#         #  1 2
#         #  3
#         n_1_up = 0  # Counter for 1 nodes of up triangles
#         n_2_up = 0
#         n_3_up = 0
#         u_1_up = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))  # Nodal 1 values of u for up triangles
#         u_2_up = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))
#         u_3_up = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))
#         xv_up = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))  # Vertex coordinates for up triangles
#         yv_up = np.zeros(1 + nx - 1 + (ny - 1) * (nx - 1 + 1))
#         # self.U = np.zeros((nx + 1, ny + 1))  # array of u values used to get the scipy interpolator, for debug
#
#         # This will build different arrays useful for computing T, Txy. See the technical documentation for details
#         for j in range(ny + 1):  # Pythonics for j=0...ny
#             for i in range(nx + 1):  # Pythonics for i=0...nx
#
#                 u_ij = nodal_vals[i + j * (nx + 1)]
#                 # self.U[i, j] = u_ij # used to get the scipy interpolator, for debug
#
#                 if i > 0 and j < ny:  # Python wants and !!
#                     u_1_dw[n_1_dw] = u_ij
#                     n_1_dw += 1
#                 if i < nx and j < ny:
#                     u_2_dw[n_2_dw] = u_ij
#                     xv_dw[n_2_dw] = self.x0 + i * self.hx
#                     yv_dw[n_2_dw] = self.y0 + j * self.hy
#                     n_2_dw += 1
#                     u_3_up[n_3_up] = u_ij
#                     xv_up[n_3_up] = self.x0 + i * self.hx
#                     yv_up[n_3_up] = self.y0 + j * self.hy
#                     n_3_up += 1
#                 if i > 0 and j > 0:
#                     u_3_dw[n_3_dw] = u_ij
#                     n_3_dw += 1
#                     u_2_up[n_2_up] = u_ij
#                     n_2_up += 1
#                 if i < nx and j > 0:
#                     u_1_up[n_1_up] = u_ij
#                     n_1_up += 1
#
#         # For dw triangles like
#         #    3
#         #  2 1
#         T_dw = self.hx * self.hy * u_2_dw - self.hy * u_1_dw * xv_dw + self.hy * u_2_dw * xv_dw + self.hx * u_1_dw * yv_dw - self.hx * u_3_dw * yv_dw
#         Tx_dw = self.hy * u_1_dw - self.hy * u_2_dw
#         Ty_dw = - self.hx * u_1_dw + self.hx * u_3_dw
#
#         # For triangles up (up = facing up)
#         #  1 2
#         #  3
#         T_up = self.hx * self.hy * u_3_up + self.hy * u_1_up * xv_up - self.hy * u_2_up * xv_up - self.hx * u_1_up * yv_up + self.hx * u_3_up * yv_up
#         Tx_up = - self.hy * u_1_up + self.hy * u_2_up
#         Ty_up = self.hx * u_1_up - self.hx * u_3_up
#
#         # All together: computing T, Txy
#         T = np.array([T_dw, T_up]).T
#         Tx = np.array([Tx_dw, Tx_up]).T
#         Ty = np.array([Ty_dw, Ty_up]).T
#
#         return T, Tx, Ty
#
#     def get_interpolator_elliptic(self, M):
#         '''
#         It takes in a numpy 2d array of N points (Nx2) (or also a single 2D point, a numpy 1d array)
#         It returns the interpolated values at the query points.
#         For details about the functioning refer to the technical documentation.
#         '''
#
#         # If a single query point
#         if len(np.shape(M)) == 1:
#             M = np.array([M])
#
#         # Getting the index of the rectangular cell and the type of the triangle, while also ensuring the query index
#         # is admissible (and at the same time: being able to input any point we want)
#         index_raw, type_raw = np.divmod(M - [self.x0, self.y0], [self.hx, self.hy])
#         index_raw = np.clip(index_raw, self.zero, self.pad_v)
#         index_def = np.squeeze((index_raw[:, 0] + index_raw[:, 1] * self.nx)).astype(int)
#         type_semidef = np.squeeze(type_raw[:, 0] * self.slope < type_raw[:, 1]).astype(int)  # If 0, dw triangle
#         correction = np.isclose(M, [self.x1, self.y1], 1e-10, 0).astype(int)
#         type_def = np.clip(type_semidef + np.sum(correction * [-1, 1], axis=1), 0, 1)
#
#         # Interpolation
#         Px = self.Tx[index_def, type_def] * M[:, 0]
#         Py = self.Ty[index_def, type_def] * M[:, 1]
#         N = self.T[index_def, type_def] + Px + Py
#
#         return np.squeeze(N / self.D)
#
#     def get_interpolator_parabolic(self, M, It):
#         '''
#         It takes in a matrix of N points (Nx2) and an array of time indices (Tx1), which is a subset of 0:len(fem_data)
#         It returns a matrix that is NxT (N spatial interpolations at every time)
#         '''
#
#         # If a single query point / single time
#         if len(np.shape(M)) == 1:
#             M = np.array([M])
#         if not type(It) == list:
#             It = [It]
#
#         # Getting the index of the rectangular cell and the type of the triangle, while also ensuring the query index
#         # is admissible (and at the same time: being able to input any point we want)
#         index_raw, type_raw = np.divmod(M - [self.x0, self.y0], [self.hx, self.hy])
#         index_raw = np.clip(index_raw, self.zero, self.pad_v)
#         index_def = np.squeeze((index_raw[:, 0] + index_raw[:, 1] * self.nx)).astype(int)
#         type_semidef = np.squeeze(type_raw[:, 0] * self.slope < type_raw[:, 1]).astype(int)  # If 0, dw triangle
#         correction = np.isclose(M, [self.x1, self.y1], 1e-10, 0).astype(int)
#         type_def = np.clip(type_semidef + np.sum(correction * [-1, 1], axis=1), 0, 1)
#
#         # Interpolation (time dependent version)
#         row_indices = np.array(It)[:, None]
#         Px = self.Tx_glo[row_indices, index_def, type_def] * M[:, 0]
#         Py = self.Ty_glo[row_indices, index_def, type_def] * M[:, 1]
#         N = self.T_glo[row_indices, index_def, type_def] + Px + Py
#
#         return np.squeeze(N / self.D)
#
#     def __call__(self, *args, **kwargs):
#         '''
#         Returns the interpolator function, either time dependent or independent.
#         '''
#         if not self.time_dependent:
#             return self.get_interpolator_elliptic(*args, **kwargs)
#         else:
#             return self.get_interpolator_parabolic(*args, **kwargs)
#
#     # def get_scipy_interpolator(self):
#     #    return RegularGridInterpolator((self.x, self.y), self.U)