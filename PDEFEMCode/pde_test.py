import unittest
from unittest import TestCase

import PDEFEMCode.interface
import PDEFEMCode.fenics_utils as pde_utils
import PDEFEMCode.interface as pde_IO
import fenics as fc
import numpy as np
import os


## Helper functions used in testing only.###

def grad_hat_ref(X, Y):
    X_center, Y_center = 0.5 - X, 0.5 - Y
    Y_geq_X = np.abs(Y_center) >= np.abs(X_center)
    in_b = np.logical_not( np.logical_or(np.abs(X_center)>0.5,np.abs(Y_center)>0.5) )
    return np.array([0, 2]) * (np.logical_and(Y_geq_X,in_b) * np.sign(Y_center))[..., np.newaxis] + \
           np.array([2, 0]) * (np.logical_and(np.logical_not(Y_geq_X),in_b) * np.sign(X_center))[..., np.newaxis]


def eval_hat(X, Y):
    return 1 - 2 * np.maximum(np.abs(X - .5), np.abs(Y - .5))


## Begin Testing code.
class TestEllipticSolver(TestCase):
    @classmethod
    def setUpClass(self):
        self.n = 100
        self.setup_function_mesh_cls(self.n)

    @classmethod
    def setup_function_mesh_cls(cls, n):
        cls.mesh, cls.fn_space = pde_utils.setup_unitsquare_function_space(n)
        cls.LHS = staticmethod(pde_utils.elliptic_LHS)
        cls.RHS = staticmethod(pde_utils.elliptic_RHS)

        cls.u_trial = fc.TrialFunction(cls.fn_space)
        cls.v_test = fc.TestFunction(cls.fn_space)

    def setup_function_mesh(self, n):
        self.mesh, self.fn_space = pde_utils.setup_unitsquare_function_space(n)
        self.LHS = pde_utils.elliptic_LHS
        self.RHS = pde_utils.elliptic_RHS

        self.u_trial = fc.TrialFunction(self.fn_space)
        self.v_test = fc.TestFunction(self.fn_space)

    def solve_obtain_error(self, RHS_fn, u_ref):
        '''
        helper function for test cases below.
        '''
        LHS_int, RHS_int = pde_utils.variational_formulation(
            self.u_trial, self.v_test,
            self.LHS,
            self.RHS, RHS_fn
        )
        u_sol = pde_utils.solve_vp(self.fn_space, LHS_int, RHS_int)

        error_L2 = pde_utils.error_L2(u_ref, u_sol)
        error_LInf = pde_utils.error_LInf_piece_lin(u_ref, u_sol, self.mesh)
        return error_L2, error_LInf, u_sol

    def test_PDE_solve_constant(self):
        # Constant function is supplied.
        # The solver is expected to return the same constant.

        RHS_fn = fc.Constant(3.0)
        u_ref = RHS_fn

        error_L2, error_LInf, _ = self.solve_obtain_error(RHS_fn, u_ref)
        print('Constant function error_L2  =', error_L2)
        print('Constant function error_Linf =', error_LInf)
        np.testing.assert_almost_equal(error_L2, 0, decimal=10)
        np.testing.assert_almost_equal(error_LInf, 0, decimal=10)

    def test_PDE_solve_sines(self):
        # Constant function is supplied.
        # The solver is expected to return the same constant.

        RHS_fn = fc.Expression('(2*pi*pi + 1)*sin(pi*x[0] + pi/2)*sin(pi*x[1]+pi/2)',
                               element=self.fn_space.ufl_element())
        u_ref = fc.Expression('sin(pi*x[0] + pi/2)*sin(pi*x[1]+pi/2)',
                              element=self.fn_space.ufl_element())

        error_L2, error_LInf, _ = self.solve_obtain_error(RHS_fn, u_ref)
        print('product sines function error_L2  =', error_L2)
        print('product sines function error_Linf =', error_LInf)
        np.testing.assert_almost_equal(error_L2, 0, decimal=3)
        np.testing.assert_almost_equal(error_LInf, 0, decimal=2)
        # 5e-4 error reported in test MATLAB code.

    def testConvergenceOrder(self):
        RHS_fn = fc.Expression('(2*pi*pi + 1)*sin(pi*x[0] + pi/2)*sin(pi*x[1]+pi/2)',
                               element=self.fn_space.ufl_element())
        u_ref = fc.Expression('sin(pi*x[0] + pi/2)*sin(pi*x[1]+pi/2)',
                              element=self.fn_space.ufl_element())

        n_list = np.asarray(np.ceil(np.logspace(5, 9, 10, base=2)),
                            dtype=np.int32)
        error_L2_list, error_LInf = np.empty((2, 10))
        for ind, n_grid in enumerate(n_list):
            self.setup_function_mesh(n_grid)
            self.solve_obtain_error(RHS_fn, u_ref)
            error_L2_list[ind], error_LInf[ind], _ = self.solve_obtain_error(RHS_fn, u_ref)

        OOC = -(np.log(error_L2_list[1:]) - np.log(error_L2_list[:-1])) / \
              (np.log(n_list[1:]) - np.log(n_list[:-1]))
        # check that we are within 0.01 of the desired order of convergence
        np.testing.assert_allclose(OOC, 2, atol=.01)

    def testPickleLoad(self):
        rmesh_p = pde_IO.RectMeshParams(nx = self.n, ny = self.n, P0=[0,0],P1 = [1,1])

        RHS_fn = fc.Expression('(2*pi*pi + 1)*sin(pi*x[0] + pi/2)*sin(pi*x[1]+pi/2)',
                               element=self.fn_space.ufl_element())
        u_ref = fc.Expression('sin(pi*x[0] + pi/2)*sin(pi*x[1]+pi/2)',
                              element=self.fn_space.ufl_element())
        error_L2, error_LInf, u_sol = self.solve_obtain_error(RHS_fn, u_ref)

        save_file = 'pde_test_sine_sol'
        out_dir = 'test-files'
        f = PDEFEMCode.interface.FenicsRectangleLinearInterpolator(rmesh_p, u_sol)
        u_grad = pde_utils.fenics_grad(self.mesh, u_sol)
        grad_f = PDEFEMCode.interface.FenicsRectangleVecInterpolator(rmesh_p, u_grad)

        save_params = {'f': f, 'grad_f': grad_f}
        pde_IO.pickle_save(out_dir, save_file, save_params)

        fname = os.path.join(out_dir, save_file)
        load_params = pde_IO.pickle_load(fname)

        coords = np.stack(np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20)), axis=-1)

        f_save_eval = f(coords.reshape(-1, 2))
        f_load_eval = load_params['f'](coords.reshape(-1, 2))
        np.testing.assert_array_equal(f_load_eval, f_save_eval)
        grad_f_save_eval = grad_f(coords)
        grad_f_load_eval = load_params['grad_f'](coords)
        np.testing.assert_array_equal(grad_f_save_eval, grad_f_load_eval)


class TestPDEParabolicSolver(TestCase):
    @classmethod
    def setUpClass(self):
        self.setup_preliminary()

    @classmethod
    def setup_preliminary(cls):
        fc.set_log_active(False)  # disable messages of Fenics

        # Bottom right and top left corner of rectangular domain
        cls.P0 = np.array([0, 0])
        cls.P1 = np.array([1, 1])

        # Discretizations parameters
        cls.T = 1.0  # final time

        cls.LHS = staticmethod(pde_utils.heat_eq_LHS)
        cls.RHS = staticmethod(pde_utils.heat_eq_RHS)

    def ooc_arrays(self, D, ip, fp):
        '''
        Returns time and space discretization parameters, if we want to test the order of convergence of the parabolic
        solver

        :param D: number of times we'll solve the heat equation
        :param ip: first time discretization (uniform, with 2^ip subdivisions)
        :param fp: lst time discretization (uniform, with 2^fp subdivisions)

        It computes:
            time_array, i.e. a logspace array from 2^ip, to 2^fp, of D entries
            N_array, i.e. sqrt(time_array)
        '''

        self.D = D  # how many times do we want to do solve our problem ?
        initial_power = ip
        final_power = fp
        self.time_array = np.ceil(np.logspace(initial_power, final_power, D, base=2.0)).astype(
            int)  # vector of total time steps, per time we do a time discretization
        self.N_array = np.ceil(np.sqrt(self.time_array)).astype(int)  # vector of mesh sizes

    def time_independence_arrays(self, D, ip, fp, Nt):
        '''
        Returns time and space discretization parameters, if we want to test that the computed error is not influenced
        by the time discretization

        :param D: number of times we'll solve the heat equation
        :param ip: first space discretization (with 2^ip subdivisions per side)
        :param fp: lst space discretization (with 2^fp subdivisions per side)
        :param Nt: only one temporal size

        It computes:
            N_array, i.e. a logspace array from 2^ip, to 2^fp, of D entries
            time_array = repeated Nt to match the length of N_array
        '''

        self.D = D  # how many times do we want to do solve our problem ?
        initial_power = ip
        final_power = fp
        self.N_array = np.ceil(np.logspace(initial_power, final_power, D, base=2.0)).astype(int)
        self.time_array = np.ceil(np.repeat([Nt], self.D)).astype(int)  # vector of time discretizations

    def compute_ooc(self, err, verbose=True):
        times = self.time_array
        ooc = - (np.log(err[1:]) - np.log(err[:-1])) / (np.log(times[1:]) - np.log(times[:-1]))
        if verbose:
            print('Order of convergence: ', ooc)
        return ooc

    def compute_ooc_space(self, err, verbose=True):
        spaces = self.N_array
        ooc = - (np.log(err[1:]) - np.log(err[:-1])) / (np.log(spaces[1:]) - np.log(spaces[:-1]))
        if verbose:
            print('Order of convergence: ', ooc)
        return ooc

    def solve_obtain_error(self, RHS_fn, u_ref, t_err, err_type, verbose=False):
        '''
        Helper function for test cases below. It runs a series of tests with different discretizations, and it then
        returns the corresponding errors.
        The error is only computed around the time t_err
        '''

        # Errors tensor
        err_tot = np.zeros(self.D)

        # For loop computing errors
        for current_discr in range(self.D):

            # Space discretization
            N = self.N_array[current_discr]
            rmesh_p = pde_IO.RectMeshParams(nx= N, ny=N, P0 = self.P0,P1 = self.P1)
            mesh, V = pde_utils.setup_rectangular_function_space(rmesh_p)

            # Time discretization
            curr_time_steps = self.time_array[current_discr]
            time_disc_p = pde_utils.TimeDiscParams(self.T,curr_time_steps)
            dt, times = pde_utils.setup_time_discretization(time_disc_p)

            # Variational problem
            u_previous = fc.interpolate(fc.Constant(0), V)  # the solution at initial time is zero
            u_trial = fc.TrialFunction(V)
            v_test = fc.TestFunction(V)
            LHS_int, RHS_int = pde_utils.variational_formulation(
                u_trial, v_test,
                self.LHS,
                self.RHS, RHS_fn,
                {'dt': dt},
                {'dt': dt, 'u_previous': u_previous}
            )

            t = times[0]  # initial time

            computed_error = False

            # Solving
            for n in range(curr_time_steps):
                # Time update
                t = times[n + 1]
                u_ref.t = t
                RHS_fn.t = t  # NB. This change is also reflected inside LHS_int, RHS_int

                # Solution at this time (no BC!) (NB. It's not very explicit that time changed, but it did above)
                u_current = pde_utils.solve_vp(V, LHS_int, RHS_int)

                # Update previous solution
                u_previous.assign(u_current)

                # Saving the error only at the middle timestep
                if t > t_err and not computed_error:
                    if err_type == 'uni':
                        error = pde_utils.error_LInf_piece_lin(u_ref, u_current, mesh)
                    elif err_type == 'L2':
                        error = fc.errornorm(u_ref, u_current, 'L2')
                    else:
                        error = fc.errornorm(u_ref, u_current, 'H1')

                    err_tot[current_discr] = error

                    computed_error = True

                if verbose:
                    print('Discretization: ', 1 + current_discr, '/', self.D, '; ', np.ceil(n / curr_time_steps * 100),
                          ' % done.')

        return err_tot

    def test_easy_polynomial(self):

        '''
        Expected behaviour: the L2 error should be O(h^2).
        '''

        RHS_fn = fc.Expression('3*pow(x[0],2) - 2*pow(x[0],3) + (3 - 2*x[1])*pow(x[1],2) + 12*t*(-1 + x[0] + x[1])',
                               degree=2, t=0)
        u_ref = fc.Expression('t*(3*pow(x[0],2) - 2*pow(x[0],3) + (3 - 2*x[1])*pow(x[1],2))', degree=2, t=0)

        # Discretization parameters, the time discretization won't change
        self.time_independence_arrays(4, 5, 7, 20)

        # Computing the error at time t_err
        err_type = 'L2'  # error to be outputted ('uni', 'L2', 'H1')
        t_err = 0.5
        err_tot = self.solve_obtain_error(RHS_fn, u_ref, t_err, err_type, verbose=True)

        # Quadratic convergence?
        # Testing the order of convergence
        ooc = self.compute_ooc_space(err_tot)
        np.testing.assert_almost_equal(ooc[-1], 2, 0.05)

        # Now we also let the time vary
        # Setting up a test for the order of convergence
        self.ooc_arrays(4, 7, 10)

        # Computing the error at time t_err
        err_type = 'L2'  # error to be outputted ('uni', 'L2', 'H1')
        t_err = 0.5
        err_tot = self.solve_obtain_error(RHS_fn, u_ref, t_err, err_type, verbose=True)

        # Testing the order of convergence
        ooc = self.compute_ooc_space(err_tot)
        np.testing.assert_almost_equal(ooc[-1], 2, 0.05)

    def test_constant(self):

        '''
        Expected behaviour: the exact solution (up to machine precision) is computed.
        '''

        RHS_fn = fc.Expression('1', degree=2)
        u_ref = fc.Expression('t', degree=2, t=0)

        # Setting up the test
        self.ooc_arrays(4, 7, 12)
        err_type = 'uni'  # error to be outputted ('uni', 'L2', 'H1')
        err_tot = self.solve_obtain_error(RHS_fn, u_ref, 0.5, err_type, verbose=True)

        # No error?
        np.testing.assert_allclose(err_tot, 0, atol=1e-12, rtol=0)

    def test_moving_bump(self):
        '''
        Expected behaviour: loglog(time_array, err_tot) and loglog(time_array, 1/time_array) should be parallel, in the
        L2 norm (at least, asymptotically). I.e. we want order of convergence asymptotically close to 1
        '''

        fcase1 = '(exp(16 + 1/(-0.0625 + pow(-0.5 + x[0] - cos(t)/4.,2) + pow(-0.5 + x[1] - sin(t)/4.,2)))*sin((' \
                 '3*t)/2.)*(96*cos((3*t)/2.) + ((128*((-1 + 2*x[1])*cos(t) + sin(t) - 2*x[0]*sin(t)))/pow(2 - 4*x[0] ' \
                 '+ 4*pow(x[0],2) - 4*x[1] + 4*pow(x[1],2) + cos(t) - 2*x[0]*cos(t) + sin(t) - 2*x[1]*sin(t),' \
                 '2) - (7*pow(2 - 4*x[0] + cos(t),2) + pow(2 - 4*x[0] + cos(t),4) + pow(2 - 4*x[0] + cos(t),' \
                 '2)*pow(2 - 4*x[1] + sin(t),2) - 4*pow(2 - 4*x[0] + 4*pow(x[0],2) - 4*x[1] + 4*pow(x[1],2) + cos(t) ' \
                 '- 2*x[0]*cos(t) + sin(t) - 2*x[1]*sin(t),2))/pow(-0.0625 + pow(-0.5 + x[0] - cos(t)/4.,' \
                 '2) + pow(-0.5 + x[1] - sin(t)/4.,2),4) - (7*pow(2 - 4*x[1] + sin(t),2) + pow(2 - 4*x[0] + cos(t),' \
                 '2)*pow(2 - 4*x[1] + sin(t),2) + pow(2 - 4*x[1] + sin(t),4) - 4*pow(2 - 4*x[0] + 4*pow(x[0],' \
                 '2) - 4*x[1] + 4*pow(x[1],2) + cos(t) - 2*x[0]*cos(t) + sin(t) - 2*x[1]*sin(t),2))/pow(-0.0625 + ' \
                 'pow(-0.5 + x[0] - cos(t)/4.,2) + pow(-0.5 + x[1] - sin(t)/4.,2),4))*sin((3*t)/2.)))/64. '
        fcase2 = '0'
        fcond1 = '2 + 4*pow(x[0],2) + 4*pow(x[1],2) + cos(t) + sin(t) < 2*(2*(x[0] + x[1]) + x[0]*cos(t) + x[1]*sin(t))'
        fexpr = fcond1 + ' ? ' + fcase1 + ' : ' + fcase2
        RHS_fn = fc.Expression(fexpr, degree=6, t=0)

        ucase1 = '(exp(16 + 1/(-0.0625 + pow(-0.5 + x[0] - cos(t)/4.,2) + pow(-0.5 + x[1] - sin(t)/4.,2)))*pow(sin((' \
                 '3*t)/2.),2))/2. '
        ucase2 = '0'
        ucond1 = '2 + 4*pow(x[0],2) + 4*pow(x[1],2) + cos(t) + sin(t) < 2*(2*(x[0] + x[1]) + x[0]*cos(t) + x[1]*sin(t))'
        uexpr = ucond1 + ' ? ' + ucase1 + ' : ' + ucase2
        u_ref = fc.Expression(uexpr, degree=6, t=0)

        # Setting up a test for the order of convergence
        self.ooc_arrays(4, 7, 12)

        # Computing the error at time t_err
        err_type = 'L2'  # error to be outputted ('uni', 'L2', 'H1')
        t_err = 0.5
        err_tot = self.solve_obtain_error(RHS_fn, u_ref, t_err, err_type, verbose=True)

        # Testing the order of convergence
        ooc = self.compute_ooc(err_tot)
        np.testing.assert_almost_equal(ooc[-1], 1, 0.05)

    def test_moving_bumps(self):

        '''
        Expected behaviour: the code runs without errors. No analytical solution is provided here, this is just a test
        for conditional expressions.
        '''
        fexpr = pde_utils.parabolic_double_bump_expr()

        RHS_fn = fc.Expression(fexpr, degree=6, t=0)

        # ucase1 = '(exp(16 + 1/(-0.0625 + pow(-0.5 + x[0] - cos(t)/4.,2) + pow(-0.5 + x[1] - sin(t)/4.,2)))*pow(sin((' \
        #          '3*t)/2.),2))/2. '
        # ucase2 = '0'
        # ucond1 = '2 + 4*pow(x[0],2) + 4*pow(x[1],2) + cos(t) + sin(t) < 2*(2*(x[0] + x[1]) + x[0]*cos(t) + x[1]*sin(t))'
        # uexpr = ucond1 + ' ? ' + ucase1 + ' : ' + ucase2
        u_ref = fc.Expression('0', degree=6, t=0)   # No exact solution available, this is dummy

        # Setting up a test for the order of convergence
        self.ooc_arrays(4, 7, 9)
        err_type = 'L2'  # error to be outputted ('uni', 'L2', 'H1')
        _ = self.solve_obtain_error(RHS_fn, u_ref, 0.5, err_type, verbose=True)

    def test_polynomial(self):

        '''
        Expected behaviour: loglog(time_array, err_tot) and loglog(time_array, 1/time_array) should be parallel, in the
        L2 norm (at least, asymptotically). I.e. we want order of convergence asymptotically close to 1
        '''

        RHS_fn = fc.Expression(
            '3*(-1 + x[0] + x[1]) - 3*(-1 + x[0] + x[1])*cos(3*t) + (3*(3*pow(x[0],2) - 2*pow(x[0],3) + (3 - '
            '2*x[1])*pow(x[1],2))*sin(3*t))/4', degree=2, t=0)
        u_ref = fc.Expression('((-3*pow(x[0],2) + 2*pow(x[0],3) + pow(x[1],2)*(-3 + 2*x[1]))*(-1 + cos(3*t)))/4',
                              degree=2, t=0)

        # Setting up a test for the order of convergence
        self.ooc_arrays(4, 7, 12)

        # Computing the error at time t_err
        err_type = 'L2'  # error to be outputted ('uni', 'L2', 'H1')
        t_err = 0.5
        err_tot = self.solve_obtain_error(RHS_fn, u_ref, t_err, err_type, verbose=True)

        # Testing the order of convergence
        ooc = self.compute_ooc(err_tot)
        np.testing.assert_almost_equal(ooc[-1], 1, 0.05)

    def test_non_smooth(self):
        '''
        Expected behaviour: the code runs without errors. No analytical solution is provided here, this is just a test
        for conditional expressions.
        '''

        fexpr = pde_utils.parabolic_non_smooth_expr()

        RHS_fn = fc.Expression(fexpr, degree=6, t=0)

        # ucase1 = '(exp(16 + 1/(-0.0625 + pow(-0.5 + x[0] - cos(t)/4.,2) + pow(-0.5 + x[1] - sin(t)/4.,2)))*pow(sin((' \
        #          '3*t)/2.),2))/2. '
        # ucase2 = '0'
        # ucond1 = '2 + 4*pow(x[0],2) + 4*pow(x[1],2) + cos(t) + sin(t) < 2*(2*(x[0] + x[1]) + x[0]*cos(t) + x[1]*sin(t))'
        # uexpr = ucond1 + ' ? ' + ucase1 + ' : ' + ucase2
        u_ref = fc.Expression('0', degree=6, t=0)

        # Setting up a test for the order of convergence
        self.ooc_arrays(4, 7, 9)
        err_type = 'L2'  # error to be outputted ('uni', 'L2', 'H1')
        _ = self.solve_obtain_error(RHS_fn, u_ref, 0.5, err_type, verbose=True)

class TestInterpolators(unittest.TestCase):
    '''
    Tests the non-native interpolators.
    '''

    def setUp(self):
        self.n = 50
        self.mesh, self.fn_space = pde_utils.setup_unitsquare_function_space(self.n)

    def test_fenics_interpolate_bilin(self):

        affine_fc = fc.Expression('1 + 3*x[0] + 4*x[1]',
                                  element=self.fn_space.ufl_element())
        affine_np = pde_utils.fenics_unit_square_function_wrap(self.mesh, self.n, affine_fc)
        X, Y = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40), indexing='ij')
        eval_ref = 1 + 3 * X + 4 * Y
        eval_wrap = affine_np(np.stack((X, Y), axis=-1))
        np.testing.assert_almost_equal(eval_ref, eval_wrap)

    def test_parabolic_interp(self):
        rmesh_p = pde_IO.RectMeshParams(nx = 5,
                                        ny = 6,
                                        P0 = np.array([4, 1]),
                                        P1 = np.array([10, 23]))
        mesh, fn_space = pde_utils.setup_rectangular_function_space(rmesh_p)
        T_fin = 1
        Nt = 2

        F1 = fc.interpolate(fc.Expression('x[0]+pow(x[1],2)', degree=1), fn_space)
        F2 = fc.interpolate(fc.Expression('3*x[0]+pow(x[1],2)+cos(100*x[0])', degree=1), fn_space)
        F3 = fc.interpolate(fc.Expression('x[0] * x[1]', degree=1), fn_space)
        list_fenics = [F1, F2, F3]

        # Note, for very 'high' functions, the difference between me and Fenics is O(1e-6), instead of O(1e-13)
        wrap = pde_IO.FenicsRectangleLinearInterpolator(rmesh_p, list_fenics, T_fin=T_fin, Nt=Nt, time_dependent=True,
                                                        time_as_indices=False, verbose=True)

        P = np.array([[5.1, 22], [10, 18], [8, 23], [9.5, 1.1], [10, 2.5], [10, 23]])

        # A test on single time-space evaluations
        not_mine = np.zeros((np.shape(P)[0], len(list_fenics)))
        mine = np.zeros(len(list_fenics))
        for i in range(np.shape(P)[0]):

            for j in range(len(list_fenics)):
                not_mine[i, j] = list_fenics[j](P[i, :])
                mine[j] = wrap(P[i, :], [0, .5, 1][j])

            delta = mine - not_mine[i, :]
            np.testing.assert_almost_equal(np.max(np.abs(delta)), 0, decimal=8)

        # A test on vector evaluation
        mine = wrap(P)
        np.testing.assert_almost_equal(np.max(np.abs(mine.T - not_mine)), 0, decimal=8)

        times = [.4, .99]
        mine = wrap(P, times)
        np.testing.assert_almost_equal(np.max(np.abs(mine.T - not_mine[:, [1, 2]])), 0, decimal=8)

        # Testing the optimization mode
        # Pointwise test
        tr_len = 3
        not_mine = np.zeros(tr_len)
        for i in range(tr_len):
            not_mine[i] = list_fenics[i](P[i, :])
            np.testing.assert_almost_equal(not_mine[i] - wrap(P[i, :], [[-1, .6, 1][i]], trajectory_mode=True), 0,
                                           decimal=8)

        # Vector test
        mine = wrap(P[[0, 2], :], [0, 1], trajectory_mode=True)
        np.testing.assert_almost_equal(np.max(np.abs(mine - not_mine[[0, 2]])), 0, decimal=8)
        mine = wrap(P[[0, 1, 2], :], trajectory_mode=True)
        np.testing.assert_almost_equal(np.max(np.abs(mine - not_mine)), 0, decimal=8)

        # Now a test on the optimization version
        wrapO = pde_IO.FenicsRectangleLinearInterpolator(rmesh_p, list_fenics, T_fin=T_fin, Nt=Nt, time_dependent=True,
                                                         time_as_indices=True, verbose=True)
        # Test without explicit indices
        mineO = wrapO(P[[0, 1, 2], :])
        np.testing.assert_almost_equal(np.max(np.abs(mineO - not_mine)), 0, decimal=8)

        # Test with explicit indices
        mineO = wrapO(P[[2,0,0], :], [2, 0, 0])
        np.testing.assert_almost_equal(np.max(np.abs(mineO - not_mine[[2,0,0]])), 0, decimal=8)

    def test_elliptic_interp(self):
        rmesh_p = pde_IO.RectMeshParams(nx = 2,
                                        ny = 2,
                                        P0 = np.array([1, 1]),
                                        P1 = np.array([2, 2]))
        mesh, fn_space = pde_utils.setup_rectangular_function_space(rmesh_p)

        u_fenics = fc.interpolate(fc.Expression('x[0]*x[1]', degree=1), fn_space)

        wrap = pde_IO.FenicsRectangleLinearInterpolator(rmesh_p, u_fenics, time_dependent=False)
        # my_interp = wrap.get_interpolator()

        P = np.array([[1, 1], [2, 2], [1.5, 1], [1.5, 2], [1, 1.5], [2, 1.5], [1.3, 1.7]])

        # Pointwise test
        not_mine = np.zeros(np.shape(P)[0])
        for i in range(np.shape(P)[0]):
            not_mine[i] = u_fenics(P[i,:])
            np.testing.assert_almost_equal(not_mine[i], wrap(P[i,:]))

        # Vector test
        np.testing.assert_almost_equal(not_mine, wrap(P))


    def test_fenics_lin_interpolator_rectangle_right(self):
        rmesh_p = pde_IO.RectMeshParams(nx = 2,
                                        ny = 2,
                                        P0 = np.array([0, 0]),
                                        P1 = np.array([1, 1]))
        mesh, fn_space = pde_utils.setup_rectangular_function_space(rmesh_p)

        u_fenics = fc.interpolate(fc.Expression('x[0]*x[1]', degree=1), fn_space)
        wrap = PDEFEMCode.interface.FenicsRectangleLinearInterpolator(rmesh_p, u_fenics)
        # my_interp = wrap.get_interpolator

        P = np.array([[.25, 1], [1, 1]])

        # np.testing.assert_almost_equal(wrap(P) - u_fenics(P), 0, decimal=10)
        print(wrap(P))

    def test_fenics_grad_interpolator_rectangle_right(self):
        rmesh_p = pde_IO.RectMeshParams(nx = 5,
                                        ny = 6,
                                        P0 = np.array([4, 1]),
                                        P1 = np.array([10, 23]))
        mesh, fn_space = pde_utils.setup_rectangular_function_space(rmesh_p)

        u_fenics = fc.interpolate(fc.Expression('x[0]+pow(x[0],3)/42+pow(x[1],2)', degree=1), fn_space)
        u_fenics_grad = pde_utils.fenics_grad(mesh, u_fenics)
        grad_approxim = PDEFEMCode.interface.FenicsRectangleVecInterpolator(rmesh_p, u_fenics_grad)

        X, Y = np.meshgrid(np.linspace(4.01, 9.995, 24),
                           np.linspace(2.01, 22.995, 13), indexing='ij')
        coords = np.stack((X, Y), axis=-1)
        np.testing.assert_almost_equal(grad_approxim(coords) -
                                       PDEFEMCode.interface.native_fenics_eval_vec(u_fenics_grad, coords),
                                       0, decimal=6)

        coords = np.stack((np.linspace(4.01, 9.995, 24), np.linspace(2.01, 9.995, 24)), axis=-1)
        np.testing.assert_almost_equal(grad_approxim(coords) -
                                       PDEFEMCode.interface.native_fenics_eval_vec(u_fenics_grad, coords),
                                       0, decimal=6)
        u_fenics_affine =  fc.interpolate(fc.Expression('1 + 3*x[0] + 4*x[1]', degree=1), fn_space)
        u_fenics_aff_grad = pde_utils.fenics_grad(mesh, u_fenics_affine)
        native_eval_grad =  lambda coords: np.ones_like(coords) * np.array([3, 4]) * \
                                np.logical_and(coords >= rmesh_p.P0,coords <= rmesh_p.P1)
        affine_approxim = PDEFEMCode.interface.FenicsRectangleVecInterpolator(rmesh_p,u_fenics_aff_grad)
        # test 1d random vector, with some out of bounds
        coords = np.random.uniform(rmesh_p.P0 - 1, rmesh_p.P1 + 1, (100, 2))
        np.testing.assert_almost_equal(affine_approxim(coords),native_eval_grad(coords), decimal=6)


    def test_fenics_grad_interpolator_rectangle_right_parabolic(self):
        rmesh_p = pde_IO.RectMeshParams(nx = 5,
                                        ny = 6,
                                        P0 = np.array([4, 1]),
                                        P1 = np.array([10, 23]))

        Nt = 1
        mesh, fn_space = pde_utils.setup_rectangular_function_space(rmesh_p)

        u_fenics_list = [fc.interpolate(fc.Expression('x[0]+pow(x[0],3)/42+pow(x[1],2)', degree=1), fn_space),
                         fc.interpolate(fc.Expression('x[0]+pow(x[0],2)/42+pow(x[1],3)', degree=1), fn_space)
                         ]
        u_fenics_grad_list = [pde_utils.fenics_grad(mesh, u_fenics) for u_fenics in u_fenics_list]
        grad_approxim = PDEFEMCode.interface.FenicsRectangleVecInterpolator(rmesh_p, u_fenics_grad_list,
                                                                            time_dependent=True, Nt=Nt,T_fin=1)

        #2D coords.
        X, Y = np.meshgrid(np.linspace(4.01, 9.995, 24),
                           np.linspace(2.01, 22.995, 13), indexing='ij')
        coords = np.stack((X, Y), axis=-1)
        np.testing.assert_almost_equal(grad_approxim(coords,0) -
                                       PDEFEMCode.interface.native_fenics_eval_vec(u_fenics_grad_list[0], coords),
                                       0, decimal=6)

        coords = np.stack((np.linspace(4.01, 9.995, 24), np.linspace(2.01, 9.995, 24)), axis=-1)
        np.testing.assert_almost_equal(grad_approxim(coords,.75) -
                                       PDEFEMCode.interface.native_fenics_eval_vec(u_fenics_grad_list[1], coords),
                                       0, decimal=6)
        #invalid time
        try:
            grad_approxim(coords,1.2)
        except ValueError:
            pass
        else:
            raise Exception('Did not raise exception with bad time.')

        #1d coords
        eps = 1e-4
        n_test = 29
        coords = np.random.uniform(rmesh_p.P0+eps,rmesh_p.P1-eps,(n_test,2))
        t = np.linspace(0,1,n_test)

        approx_grad_eval = grad_approxim(coords,t)

        np.testing.assert_almost_equal(approx_grad_eval[t <=.5,:],
PDEFEMCode.interface.native_fenics_eval_vec(u_fenics_grad_list[0], coords[t <=.5,:]),
                                       )

        np.testing.assert_almost_equal(approx_grad_eval[t > .5, :],
                                       PDEFEMCode.interface.native_fenics_eval_vec(u_fenics_grad_list[1],
                                                                                   coords[t > .5, :])
                                       )

        #three points, four times. Check we are out of bounds (therefore 0) in the correct places.
        coords = np.array([[2,15],[4,0],[3,200]])
        times = np.random.uniform(0,1,(4,1))
        approx_grad_eval = grad_approxim(coords,times)
        np.testing.assert_array_equal(approx_grad_eval[:,1:,1],0)
        np.testing.assert_array_equal(approx_grad_eval[:,[0,2],0],0)

        #test new time_as_indices indexing
        # vector of coordinates.
        grad_approxim_tai = PDEFEMCode.interface.FenicsRectangleVecInterpolator(rmesh_p, u_fenics_grad_list,
                                                                                  time_dependent=True, Nt=Nt, T_fin=1, time_as_indices=True)
        coords = np.random.uniform(rmesh_p.P0 + eps, rmesh_p.P1 - eps, (3, 2))
        times_optim = np.array([0,1,1]).astype(int)
        approx_grad_optim_eval  = grad_approxim_tai(coords,times_optim)
        np.testing.assert_almost_equal(approx_grad_optim_eval[times_optim == 0, :],
                                       PDEFEMCode.interface.native_fenics_eval_vec(u_fenics_grad_list[0],
                                                                                   coords[times_optim==0, :]))
        np.testing.assert_almost_equal(approx_grad_optim_eval[times_optim == 1, :],
                                       PDEFEMCode.interface.native_fenics_eval_vec(u_fenics_grad_list[1],
                                                                                   coords[times_optim==1, :]))

        #test single coordinate.
        coord = np.random.uniform(rmesh_p.P0 + eps, rmesh_p.P1 - eps, (2,))
        time_optim = 1
        approx_grad_optim_eval = grad_approxim_tai(coord, time_optim)
        np.testing.assert_almost_equal(approx_grad_optim_eval,
                                      PDEFEMCode.interface.native_fenics_eval_vec(u_fenics_grad_list[time_optim],
                                                                                  coord))
        # test no time given.
        approx_grad_optim_allt_eval = grad_approxim_tai(coord)
        np.testing.assert_almost_equal(approx_grad_optim_allt_eval[0,:],
                                       PDEFEMCode.interface.native_fenics_eval_vec(u_fenics_grad_list[0],
                                                                                   coord))
        np.testing.assert_almost_equal(approx_grad_optim_allt_eval[1,:],
                                       PDEFEMCode.interface.native_fenics_eval_vec(u_fenics_grad_list[1],
                                                                                   coord))






class TestFenicsFnWrap(unittest.TestCase):
    '''
    Tests the wrappers of fenics built-in functions.
    '''

    def setUp(self):
        self.mesh = fc.UnitSquareMesh(1, 1, "crossed")
        self.fn_space = fc.FunctionSpace(self.mesh, 'Lagrange', 1)

    def test_fenics_fn_wrap(self):
        def test_xy(X, Y):
            ref_sol = affine_ref_f(X, Y)
            coords = np.stack((X, Y), axis=-1)
            calc_sol = PDEFEMCode.interface.native_fenics_eval_scalar(affine_fc, coords)
            np.testing.assert_array_almost_equal(calc_sol, ref_sol)

        affine_ref_f = lambda X, Y: 1 + 3 * X + 4 * Y
        affine_exp = fc.Expression('1 + 3*x[0] + 4*x[1]',
                                   element=self.fn_space.ufl_element())
        affine_fc = fc.interpolate(affine_exp, self.fn_space)

        # test random vectors
        X, Y = np.random.uniform(0, 1, (2, 5))
        test_xy(X, Y)

        # test 2d grid.
        X, Y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 18), indexing='ij')
        test_xy(X, Y)

        # test point
        X, Y = np.array([0.3, .2])
        test_xy(X, Y)

    def test_fenics_grad_wrap_affine(self):
        def test_xy(X, Y):
            ref_sol = affine_grad_ref_f(X, Y)
            coords = np.stack((X, Y), axis=-1)
            calc_sol = PDEFEMCode.interface.native_fenics_eval_vec(grad_affine_fc, coords)
            np.testing.assert_array_almost_equal(calc_sol, ref_sol)

        affine_grad_ref_f = lambda X, Y: np.ones_like(X)[..., np.newaxis] * np.array([3, 4]) * \
                            np.logical_and(np.abs(X[...,np.newaxis]-0.5) <= .5,np.abs(Y[...,np.newaxis]-0.5) <= .5 )
        affine_exp = fc.Expression('1 + 3*x[0] + 4*x[1]',
                                   element=self.fn_space.ufl_element())
        affine_fc = fc.interpolate(affine_exp, self.fn_space)
        grad_affine_fc = pde_utils.fenics_grad(self.mesh, affine_fc)
        # test random vectors
        X, Y = np.random.uniform(.001, 0.95, (2, 50))
        test_xy(X, Y)

        # test 2d grid.
        X, Y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 18), indexing='ij')
        test_xy(X, Y)

        # test point
        X, Y = np.array([0.3, .2])
        test_xy(X, Y)



    def test_fenics_grad_wrap_hat(self):
        def test_xy(X, Y):
            ref_sol = grad_hat_ref(X, Y)
            coords = np.stack((X, Y), axis=-1)
            calc_sol = PDEFEMCode.interface.native_fenics_eval_vec(grad_hat_fc, coords)
            np.testing.assert_array_almost_equal(calc_sol, ref_sol)

        # hat function
        hat_fc = fc.Expression('1 - 2*max(abs(x[0]-.5),abs(x[1]-.5))',
                               degree=1)
        hat_fc_fenics = fc.interpolate(hat_fc, self.fn_space)
        grad_hat_fc = pde_utils.fenics_grad(self.mesh, hat_fc_fenics)
        # test random vectors
        X, Y = np.random.uniform(0, 1, (2, 5))
        test_xy(X, Y)

        # test 2d grid, without points on the boundary of each gradient jump.
        X, Y = np.meshgrid(np.linspace(0.01, .995, 20), np.linspace(0.01, .992, 18), indexing='ij')
        test_xy(X, Y)

        # test point
        X, Y = np.array([0.3, .2])
        test_xy(X, Y)

class TestYAMLInterface(unittest.TestCase):
    test_fname = 'elliptic_params.yaml'

    @staticmethod
    def load_file(fname):
        obj_param = pde_IO.yaml_load(fname)
        return obj_param

    def test_load(self):
        self.load_file(self.test_fname)

    def test_elliptic_parse(self):
        obj_param = self.load_file(self.test_fname)
        elliptic_params = PDEFEMCode.fenics_utils.yaml_parse_elliptic(obj_param, self.test_fname)
