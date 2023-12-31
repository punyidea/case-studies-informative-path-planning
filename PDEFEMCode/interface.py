import argparse
import os, pickle

import numpy as np
import typing
import yaml
from dataclasses import dataclass, field
import warnings



def yaml_load(fname):
    '''
    Loads a file given by YAML filename.
    :param fname:
    :return:
    '''

    with open(fname,'r') as filep:
        par_obj = yaml.load(filep)
    return par_obj
def parse_args_cli():
    '''
    Adds functionality to a file, to read in parameters from a yaml file.
    Adds requirement to file, that it is executed with either "-y FNAME" or "--yaml_fname FNAME,"
        where FNAME is the name of the YAML parameter file where parameters are stored
    :return: params_yml: the native output of PYYAML after
            args: list of all arguments provided to the script
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-y','--yaml_fname',required=True)
    args = parser.parse_args()
    params_yml = yaml_load(args.yaml_fname)
    return params_yml,args

@dataclass
class RectMeshParams:
    '''
    Data structure which contains information about the rectangular mesh.
    '''
    nx: int                 # number of divisions of the unit interval in x, included in the mesh.
                                # There are nx + 1 distinct x coordinates in the mesh. nx = 100 on unit square
                                # corresponds to dx of 0.01
    ny: int                 # same as above, for y.
    P0: 'typing.Iterable'   # Coordinates of the bottom left of the rectangular mesh. length 2. Ex. unit square: [0,0]
    P1: 'typing.Iterable'   # Coordinates of the top right of the rectangular mesh. length 2    Ex. unit square: [1,1]


@dataclass
class IOParams:
    '''
    Data structure which stores information for saving the file
    '''
    out_file_prefix:   str  # The prefix of the save filename, without .pkl extension included.
    in_file:    str         # the name of the file that made the struct (for datakeeping)
    out_folder: str = ''    # The folder path where we would like to save

@dataclass
class VarFormulationParams:
    '''
    Parameters which store the
    '''
    LHS_form_str: str           # The function name, in string form, used to construct the left hand integral in
                                    # the variational form setup. See elliptic_LHS, heat_eq_LHS for examples.
    RHS_form_str: str           # The function name, in string form, used to construct the left hand integral in
                                    # the variational form setup. See elliptic_RHS, heat_eq_RHS for examples.

    rhs_exp_params: dict # Extra keyword arguments used in the construction of the RHS expression by the rhs_expression.
    rhs_expression_fn_str: str = '' # The function name, in string form, used to generate a FEniCS
    rhs_expression_str: str = ''    # If rhs_expression_fn_str is not set: This string will be used as the valid FEniCS expression for the RHS.

@dataclass
class TimeDiscParams:
    '''
    Parameters which store time discretization.
    '''
    T_fin: float #Ending time
    Nt: int = -1 # If -1, the default is nx*ny + 1 distinct timesteps.



@dataclass
class EllipticRunParams:
    '''
    All necessary parameters used to run an FEM simulation in elliptic_ex.py.
    '''
    rect_mesh: RectMeshParams
    io: IOParams
    var_form: VarFormulationParams


@dataclass
class ParabolicRunParams:
    '''
    All necessary parameters used to run an FEM simulation in parabolic_ex.py.
    '''
    rect_mesh: RectMeshParams
    time_disc: TimeDiscParams
    io: IOParams
    var_form: VarFormulationParams



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
    Loads the object that was stored in the .pkl file.
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
    This is an abstract class that stores common parameters of the both custom interpolators.

    Initialization parameters:
    :param rmesh_p: Parameters which define this  rectangular mesh, RectMeshParams object.
        The mesh is supposed to be composed of uniform rectangular cells, divided into triangles by the down-left to
        up-right diagonal
    :param fem_data: the fenics function to wrap in the interpolator if time_dependent = False,
        the ordered list of all such functions (as time varies), if time_dependent = True
    :param time_dependent. If True, we return the interpolator for the parabolic equation, for the elliptic otherwise.
        It's False by default.
    :param T_fin. Final time of PDE simulation. Needed if time_dependent = True
    :param Nt. Number of time intervals in which [0,T_fin] is subdivided for the simulation. Needed if time_dependent
        = True. fem_data[i] is the PDE at time 0 + T_fin/Nt * i. We therefore have a Nt+1 times discretization.
    :param time_as_indices. If time_dependent, it lets us switch between modes 2) and 3) of the interpolator.
        See the actual implementation at FenicsRectangleLinearInterpolator(RectangleInterpolator), for details.
        It is used for optimization: if True, a time-dependent interpolator will accept time indices corresponding to
        the different time steps instead of computing nearest neighbor interpolation on actual times.
        True by default.
    :param verbose: if True, it displays the building status of the time dependent interpolator.

    '''

    def __init__(self, rmesh_p, fem_data, T_fin=None, Nt=None, time_dependent=False, time_as_indices=False, verbose=False):
        # Some geometric initializations
        pass
        self.nx = rmesh_p.nx
        self.ny = rmesh_p.ny
        self.x0 = rmesh_p.P0[0]
        self.y0 = rmesh_p.P0[1]
        self.x1 = rmesh_p.P1[0]
        self.y1 = rmesh_p.P1[1]
        self.hx = (self.x1- self.x0) / self.nx
        self.hy = (self.y1 - self.y0) / self.ny

        # Some temporal initializations
        if time_dependent:
            if (T_fin is None or Nt is None):
                raise Exception('Please insert the final time and the number of time intervals.')
            if len(fem_data) != Nt + 1:
                raise Exception('There must be Nt + 1 functions supplied for the Nt + 1 timestamps.')

        self.T_fin=T_fin
        self.Nt=Nt


        # Variable deciding if the returned interpolator will have some additional functionalities or not
        self.time_as_indices = time_as_indices

        # Now, we enlarge the mesh by one cell in every direction, to handle boundary points well
        self.enx = self.nx + 2
        self.eny = self.ny + 2
        self.ex0 = self.x0-self.hx
        self.ey0 = self.y0-self.hy
        self.ex1 = self.x1+self.hx
        self.ey1 = self.y1+self.hy

        # To differentiate which interpolator to return. We now generate helping variables that will make interpolation
        # easier later on
        self.time_dependent = time_dependent
        self.verbose = verbose

        self.pre_computations(fem_data)

    def pre_computations(self,u):
        pass

    def __call__(self, coords):
        raise Exception('Abstract class. No call method implemented.')


class FenicsRectangleLinearInterpolator(RectangleInterpolator):
    '''
    Given the fenics solution to the time (in)dependent PDE of the project, it wraps a function object around it.

    This new interpolator function can be later called without the need of having fenics installed.
    This code can more generally work on any single fenics function, or a "time" list of fenics functions.


    NOTE: DEPENDENT ON USING RECTANGULAR MESH WITH DOWN/RIGHT diagonals.
    IDEALLY, THE FENICS FUNCTION(s) ARE PIECEWISE AFFINE ON MESH ELEMENTS, BUT THIS IS NOT NECESSARY
    IF A QUERY POINT (x,y) IS OUTSIDE [a,b]x[c,d], (x,y) IS TRANSFORMED INTO (clip(x,a,c), clip(y,c,d))

    Use case and examples
    Three different version of interpolators can be returned.

    1) an interpolator of a function not depending on time. Given an Nx2 matrix P of query points, it returns an N
    vector containing the evaluation of the fenics function at every point. It also supports evaluation on one point
    only, however, for efficiency, consider inputting an Nx2 matrix of points, instead of doing N times a single point
    evaluation.
        u = FenicsRectangleLinearInterpolator(rmesh_p, fenics_function) # rmesh_p is generated by RectMeshParams
        P = np.array([[42, 42], [0, 1]])
        print(u(P))  # returns a vector of length 2, containing fenics_function([42,42]), fenics_function([0,1])
        Q = np.array([1, 2])
        print(u(Q))  # returns a real number

    2) an interpolator of a function depending on time, with some nice functionalities. The query points are still a
    matrix P of size Nx2 or a single point. The times are an array times of length N_times (or a single real number).
    A matrix of size N_times x N is returned. Times that are not exactly in the correct times on which the PDE was
    simulated get mapped to the closest correct time (see example below).
    This version is to be used during development, it is advised to use 3) in the final version of the code. It is
    activated by time_as_indices = False, the default option.
        T_fin = 1   # the time simulation runs from time 0 to time 1
        Nt = 2  # the time interval [0,T_fin] is subdivided into 0,t_1,t_2,...,t_Nt+1, where t_i+1-t_i=T_fin/Nt
        list_fenics = [fenics_function1, fenics_function2, fenics_function3]    # fenics function at times 0, .5, 1

        2.1) Standard version, i.e. evaluation of a time dependent function, at all points in P.
        This is repeated for every timestamp in times.
            u = FenicsRectangleLinearInterpolator(rmesh_p, list_fenics, T_fin=T_fin, Nt=Nt, time_dependent=True, verbose=True, time_as_indices=False)
                # list_fenics[i] corresponds to [0, .5, 1][i]
            P = np.array([[5.1, 22], [10, 18], [8, 23]])
            times = [-100, 1, .4, .1]   # vector of times, which differ quite a lot from [0, .5, 1]
            u(P, times) # return u(points in P, t_i) for all t_i of times, a 4x3 matrix
            u(P) # return u(points in P, t_i) for all t_i in the times of the PDE simulation [0,.5,1],thus a 3x3 matrix

        2.2) Trajectory version. The i-th point in P is interpreted as the only query point at time i. Therefore, P
        and times must have the same length, and a vector of this length is outputted. It is fenics_function(P_i,t_i) at
        every index i. Activated by trajectory_mode, = False by default
            u = FenicsRectangleLinearInterpolator(rmesh_p, list_fenics, T_fin=T_fin, Nt=Nt, time_dependent=True, verbose=True, time_as_indices=False)
            P = np.array([[5.1, 22], [10, 18], [8, 23]])
            times = [-100, 1, .4]   # vector of times, same length of P
            u(P, times, trajectory_mode = True) # return a 3 vector. The user is notified that a time is out of bounds
            u(P, trajectory_mode = True) # luckily, [0,.5,1], the times of the PDE simulation, also has length 3, so a 3 vector is returned.
                 # It contains fenics_functioni(P_i) at every index i

    3) almost the same interpolator of 2.2), with the difference that instead of real times, now we must input a list
    of integer time indices, contained in [0,...,Nt]. No time interpolations/clipping are performed here, to get a
    faster version (NB. A list!). It is activated by time_as_indices = True.
        u = FenicsRectangleLinearInterpolator(rmesh_p, list_fenics, T_fin=T_fin, Nt=Nt, time_dependent=True, verbose=True, time_as_indices = True)
        P = np.array([[5.1, 22], [10, 18], [8, 23]])
        # times = [-100, 1, .4]   # vector of times, same length of P -> it will raise an error
        time_indices = [0, 1]
        u(P[[0,1],:], time_indices) # for evaluation of P_i at [0, .5, 1][i], i=0,1
        u(P) # for evaluation of P_i at [0, .5, 1][i], for all i

    Input variables. See Abstract RectangleInterpolator class.
    :return: a function, which when evaluated, gives the function evaluated at the desired space-time coordinates.
    '''

    def pre_computations(self, fem_data):

        # All possible time indices
        if self.time_dependent:
            self.t_ind_all = np.arange(0, self.Nt+1).tolist()
            self.dt = self.T_fin / self.Nt

        # Other temporary variables for the interpolation process
        self.D = self.hx * self.hy
        self.slope = self.hy / self.hx

        # Used to handle query points outside the original domain
        self.max_pad = np.array([self.x1, self.y1])
        self.min_pad = np.array([self.x0, self.y0])

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
                if x > self.x1+self.hx/2 or x < self.x0-self.hx/2 or y < self.y0-self.hy/2 or y > self.y1+self.hy/2:
                    u_ij = 0
                else:
                    # Next two lines for robustness against rounding errors
                    x = np.clip(x, self.x0, self.x1)
                    y = np.clip(y, self.y0, self.y1)
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

        # If a single query point
        if len(np.shape(M)) == 1:
            M = np.array([M])

        # Getting the index of the rectangular cell and the type of the triangle, while also ensuring the query index
        # is admissible (and at the same time: being able to input any point we want)
        M = np.clip(M,self.min_pad, self.max_pad)
        index_raw, type_raw = np.divmod(M - [self.ex0, self.ey0], [self.hx, self.hy])
        index_def = np.squeeze((index_raw[:, 0] + index_raw[:, 1] * self.enx)).astype(int)
        type_def = np.squeeze(type_raw[:, 0] * self.slope < type_raw[:, 1]).astype(int)  # If 0, dw triangle

        # Interpolation
        Px = self.Tx[index_def, type_def] * M[:, 0]
        Py = self.Ty[index_def, type_def] * M[:, 1]
        N = self.T[index_def, type_def] + Px + Py

        return np.squeeze(N / self.D)

    def get_interpolator_parabolic_dev(self, M, times=None, trajectory_mode=False):

        # If a single query point
        M=np.array(M, ndmin=2)
        if times is None:
            # We condider exactly all the timestamps from the heat equation simulation
            t_ind = self.t_ind_all # all time indices
        else:
            # If a single query time
            times = np.array(times, ndmin=1)
            # Victor made this change:
            eps = 1e-5
            time_ind = times / self.dt
            if np.any(np.logical_or(time_ind < 0, time_ind > self.Nt + eps)):
                warnings.warn('Warning: a time supplied was out of bounds. Check that times are in interval [0,{}]'.format(self.T_fin))
            t_ind = np.clip(np.round(time_ind), 0, self.Nt).astype(int).tolist()
            # if not trajectory_mode:
            #     eps = 1e-5
            #     time_ind = times / self.dt
            #     if np.any(np.logical_or(time_ind < 0, time_ind > self.Nt + eps)):
            #         print('Warning: a time supplied was out of bounds. Check that times are in interval [0,{}]'.format(self.T_fin))
            #     t_ind = np.round(time_ind).astype(int).tolist()
            # else:
            #     t_ind = np.clip(np.round(times / self.dt), 0, self.Nt).astype(int).tolist()

        # Getting the index of the rectangular cell and the type of the triangle, while also ensuring the query index
        # is admissible (and at the same time: being able to input any point we want)
        M = np.clip(M, self.min_pad, self.max_pad)
        index_raw, type_raw = np.divmod(M - [self.ex0, self.ey0], [self.hx, self.hy])
        index_def = np.squeeze((index_raw[:, 0] + index_raw[:, 1] * self.enx)).astype(int)
        type_def = np.squeeze(type_raw[:, 0] * self.slope < type_raw[:, 1]).astype(int)  # If 0, dw triangle

        # Interpolation (time dependent version)
        if trajectory_mode:
            # If this interpolator is run in optimization mode, then M must be of size times and we
            # evaluate at timestamp i, the interpolator at point i
            row_indices = t_ind
        else:
            row_indices = np.array(t_ind)[:, None]
        Px = self.Tx_glo[row_indices, index_def, type_def] * M[:, 0]
        Py = self.Ty_glo[row_indices, index_def, type_def] * M[:, 1]
        N = self.T_glo[row_indices, index_def, type_def] + Px + Py

        return np.squeeze(N / self.D)

    def get_interpolator_parabolic(self, M, t_ind=None):

        # If a single query point
        if len(np.shape(M)) == 1:
            M = np.array([M])

        # Getting the index of the rectangular cell and the type of the triangle, while also ensuring the query index
        # is admissible
        M = np.clip(M, self.min_pad, self.max_pad)
        index_raw, type_raw = np.divmod(M - [self.ex0, self.ey0], [self.hx, self.hy])
        index_def = np.squeeze((index_raw[:, 0] + index_raw[:, 1] * self.enx)).astype(int)
        type_def = np.squeeze(type_raw[:, 0] * self.slope < type_raw[:, 1]).astype(int)  # If 0, dw triangle

        # Interpolation (time dependent version)
        if t_ind is None:
            t_ind = self.t_ind_all
            # And M must have Nt+1 rows
        Px = self.Tx_glo[t_ind, index_def, type_def] * M[:, 0]
        Py = self.Ty_glo[t_ind, index_def, type_def] * M[:, 1]
        N = self.T_glo[t_ind, index_def, type_def] + Px + Py
        return np.squeeze(N / self.D)

    def __call__(self, *args, **kwargs):
        '''
        Returns the desired interpolator function, either time dependent or independent.
        '''
        if not self.time_dependent:
            return self.get_interpolator_elliptic(*args, **kwargs)
        else:
            if self.time_as_indices:
                return self.get_interpolator_parabolic(*args, **kwargs)
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
    When out of bounds, the function sets the gradient in the normal direction to the boundary to be 0.
        This is consistent with homogeneous Neumann boundary conditions.

    Variables. See Abstract RectangleInterpolator class.
    :return: an object, that when called, computes the gradient of the provided function (a discontinuous mesh)


    Above, coords is shape (don't care x 2), to imitate shape of native fenics caller.
    #CALL VARS (coords). See __call__()

    Use case (time independent):
        u_grad      = fenics_grad(mesh,u_fenics)
        grad_interp     = FenicsRectangleGradInterpolator(rmesh_p, u_grad)
        grad_eval   = grad_fn(coords)

    Use case (time dependent):
        u_grad_list = [fenics_grad(mesh, u_fenics) for u_fenics in u_fenics_list]
        grad_fn     = FenicsRectangleGradInterpolator(rmesh_p, u_grad_list,
                                                time_dependent=True, Nt = len(u_grad_list)-1,T_fin = 1)
        grad_eval   = grad_fn(coords,t)
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

        if not self.time_dependent:
            ndims = grad_u.value_dimension(0)
            self.T_grads = native_fenics_eval_vec(grad_u, T_coords)
        else:
            self.t_ind_all = np.arange(0, self.Nt + 1).tolist()
            vec_u_list = vec_u # we are actually given a list of functions.
            ndims = vec_u_list[0].value_dimension(0)
            self.T_grads = np.zeros((len(vec_u),) + T_coords.shape)
            for ind,vec_u in enumerate(vec_u_list):
                if vec_u.value_dimension(0)!= ndims:
                    raise ValueError(
                        'The fenics functions in the list have mismatching '
                        'dimensions between them (detected at index {}).'.format(ind))
                self.T_grads[ind,...] = native_fenics_eval_vec(vec_u,T_coords)
            self.dt = self.T_fin / self.Nt
        assert (ndims>=2) # we are in 2D, so the gradient (vector) function better have more than 1d!

    def __call__(self, coords, times = None):
        '''
        When called, returns the gradient of the point on the mesh.
        If time-dependent, performs nearest-neighbor-in-time interpolation.

        :param coords: coordinates on which we'd like to shape (don't_care_coords) by 2
        :param times: (if time dependent) must be either:
            - shape (,), or a scalar.
                Then all time is assumed to be the same.
            - the same shape as coords.shape[:-1]
                Then each coordinate point has its unique time.
                evaluate grad_f(coords[i,j,k,...]) at time[i,j,k,...]
            - If no times are given, and the interpolator is time-dependent, then
                the time_as_indices flag assumes all times are requested.
            BELOW FEATURE IS NOT EXPECTED TO BE USED THAT MUCH
             below line written in tuple form. + means concatenation
            v v v v
            - shape = (don't_care_time) + tuple(np.ones_like(coords.shape[:-1))
            In this case, gradient is evaluated at each coordinate AND each time, separately.
            resulting out array is of shape
            (don't_care_time) from time + coords.shape
        :return: gradient of the interpolator, shape  (coords.shape[:-1] + (2,) ).
            If time dependent, times may impact size of array.
        '''
        # coords_shape = coords.shape
        # coords_rs = np.reshape(coords, (-1, coords_shape[-1]))

        index_float = (coords - [self.x0, self.y0]) / [self.hx, self.hy]

        # keep track of which points were out of bounds and in which index. Then clip.
        oob = np.logical_or(index_float < [0,0], index_float > [self.nx,self.ny])
        index_float = np.clip(index_float, [0, 0], [self.nx, self.ny])

        # generate an array of indices, where in the last dim, the i-th coordinate is the index in the i-th dimension
        index_int, frac_index = np.divmod(index_float, 1)
        index_int = index_int.astype(int)

        # handle boundary cases gracefully
        frac_index[index_float == [self.nx, self.ny]] = 1
        index_int[index_float == [self.nx,self.ny]] -= 1

        # if type_def is 0, it is a BR_triangle.
        # type_def is has shape frac_index.shape[:-1]
        type_def = (frac_index[..., 0] < frac_index[..., 1]).astype(int)

        if not self.time_dependent:
            out_arr = self.T_grads[index_int[..., 0], index_int[..., 1], type_def, :]

        else:

            if self.time_as_indices:
                if times is None:
                    time_ind = self.t_ind_all
                else:
                    time_ind = times #.astype(int)
            else:
                if times is None:
                    raise ValueError('Interpolation was said to be time dependent. No time was provided.')
                eps = 1e-5
                time_ind = times/self.dt
                if np.any(np.logical_or(time_ind < 0, time_ind > self.Nt + eps)):
                    warnings.warn('Warning: a time supplied was out of bounds. Check that times are in interval [0,{}]'.format(self.T_fin))
                time_ind = np.clip(np.round(time_ind),0,self.Nt).astype(int)

            out_arr = self.T_grads[time_ind,index_int[..., 0], index_int[..., 1], type_def, :]

        out_arr[..., oob] = 0 # set gradient to be 0 in out of bounds dimension.
        #out_shape = coords_shape[:-1] + (2,)
        return out_arr


def native_fenics_eval_scalar(u_fenics, coords):
    '''
    Natively evaluate a function using fenics' evaluator.
    :param u_fenics: fenics function.
    :param coords: points such that shape.coords[-1] is the number of dimensions of the function space.
        NOTE! Coords is not in separate (X,Y) form.
    :return: shape.coords[-1] object where the fenics function has been evaluated at all coordinates.
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


