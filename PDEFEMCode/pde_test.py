import unittest
from unittest import TestCase
import PDEFEMCode.pde_utils as pde_utils
import PDEFEMCode.Object_IO as pde_IO
import fenics as fc
import numpy as np
import os


## Helper functions used in testing only.###

def grad_hat_ref(X, Y):
    X_center, Y_center = 0.5 - X, 0.5 - Y
    Y_geq_X = np.abs(Y_center) >= np.abs(X_center)
    return np.array([0, 2]) * ((Y_geq_X) * np.sign(Y_center))[..., np.newaxis] + \
           np.array([2, 0]) * ((np.logical_not(Y_geq_X)) * np.sign(X_center))[..., np.newaxis]

def eval_hat(X, Y):
    return 1 - 2 * np.maximum(np.abs(X - .5), np.abs(Y - .5))

## Begin Testing code.
class TestEllipticSolver(TestCase):
    @classmethod
    def setUpClass(self):
        self.n=100
        self.setup_function_mesh_cls(self.n)

    @classmethod
    def setup_function_mesh_cls(cls,n):
        cls.mesh,cls.fn_space = pde_utils.setup_function_space(n)
        cls.LHS = staticmethod(pde_utils.elliptic_LHS)
        cls.RHS = staticmethod(pde_utils.elliptic_RHS)

        cls.u_trial = fc.TrialFunction(cls.fn_space)
        cls.v_test = fc.TestFunction(cls.fn_space)


    def setup_function_mesh(self, n):
        self.mesh, self.fn_space = pde_utils.setup_function_space(n)
        self.LHS = pde_utils.elliptic_LHS
        self.RHS = pde_utils.elliptic_RHS

        self.u_trial = fc.TrialFunction(self.fn_space)
        self.v_test = fc.TestFunction(self.fn_space)

    def solve_obtain_error(self,RHS_fn,u_ref):
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
        return error_L2,error_LInf, u_sol

    def test_PDE_solve_constant(self):
        # Constant function is supplied.
        # The solver is expected to return the same constant.

        RHS_fn = fc.Constant(3.0)
        u_ref = RHS_fn

        error_L2, error_LInf,_ = self.solve_obtain_error(RHS_fn, u_ref)
        print('Constant function error_L2  =', error_L2)
        print('Constant function error_Linf =', error_LInf)
        np.testing.assert_almost_equal(error_L2,0,decimal=10)
        np.testing.assert_almost_equal(error_LInf,0,decimal=10)

    def test_PDE_solve_sines(self):
        # Constant function is supplied.
        # The solver is expected to return the same constant.

        RHS_fn = fc.Expression('(2*pi*pi + 1)*sin(pi*x[0] + pi/2)*sin(pi*x[1]+pi/2)',
            element = self.fn_space.ufl_element())
        u_ref = fc.Expression('sin(pi*x[0] + pi/2)*sin(pi*x[1]+pi/2)',
                element = self.fn_space.ufl_element())

        error_L2, error_LInf,_ = self.solve_obtain_error(RHS_fn, u_ref)
        print('product sines function error_L2  =', error_L2)
        print('product sines function error_Linf =', error_LInf)
        np.testing.assert_almost_equal(error_L2,0, decimal=3)
        np.testing.assert_almost_equal(error_LInf,0, decimal=2)
        # 5e-4 error reported in test MATLAB code.

    def testConvergenceOrder(self):
        RHS_fn = fc.Expression('(2*pi*pi + 1)*sin(pi*x[0] + pi/2)*sin(pi*x[1]+pi/2)',
            element = self.fn_space.ufl_element())
        u_ref = fc.Expression('sin(pi*x[0] + pi/2)*sin(pi*x[1]+pi/2)',
                element = self.fn_space.ufl_element())

        n_list = np.asarray(np.ceil(np.logspace(5,9,10, base=2)),
                                dtype= np.int32)
        error_L2_list,error_LInf = np.empty((2,10))
        for ind,n_grid in enumerate(n_list):
            self.setup_function_mesh(n_grid)
            self.solve_obtain_error(RHS_fn,u_ref)
            error_L2_list[ind], error_LInf[ind],_ = self.solve_obtain_error(RHS_fn,u_ref)

        OOC = -(np.log(error_L2_list[1:])-np.log(error_L2_list[:-1])) / \
                (np.log(n_list[1:])-np.log(n_list[:-1]))
        #check that we are within 0.01 of the desired order of convergence
        np.testing.assert_allclose(OOC,2,atol=.01)

    def testPickleLoad(self):
        nx = ny = self.n
        P0 = [0,0]
        P1 = [1,1]
        RHS_fn = fc.Expression('(2*pi*pi + 1)*sin(pi*x[0] + pi/2)*sin(pi*x[1]+pi/2)',
                               element=self.fn_space.ufl_element())
        u_ref = fc.Expression('sin(pi*x[0] + pi/2)*sin(pi*x[1]+pi/2)',
                              element=self.fn_space.ufl_element())
        error_L2, error_LInf, u_sol = self.solve_obtain_error(RHS_fn, u_ref)

        save_file = 'pde_test_sine_sol'
        out_dir = 'test-files'
        f = pde_utils.FenicsRectangleLinearInterpolator(nx, ny, P0, P1, u_sol)
        u_grad = pde_utils.fenics_grad(self.mesh, u_sol)
        grad_f = pde_utils.FenicsRectangleVecInterpolator(nx, ny, P0, P1, u_grad)

        save_params = {'f':f, 'grad_f':grad_f}
        pde_IO.pickle_save(out_dir,save_file,save_params)

        fname = os.path.join(out_dir,save_file)
        load_params = pde_IO.pickle_load(fname)

        coords = np.stack(np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20)), axis=-1)

        f_save_eval = f.get_interpolator(coords.reshape(-1,2))
        f_load_eval = load_params['f'].get_interpolator(coords.reshape(-1,2))
        np.testing.assert_array_equal(f_load_eval,f_save_eval)
        grad_f_save_eval = grad_f(coords)
        grad_f_load_eval = load_params['grad_f'](coords)
        np.testing.assert_array_equal(grad_f_save_eval,grad_f_load_eval)


class TestPDEParabolicSolver(TestCase):

    fc.set_log_active(False)    # disable messages of Fenics

    LHS = staticmethod(pde_utils.general_LHS)
    RHS = staticmethod(pde_utils.general_RHS)

    # Bottom right and top left corner of rectangular domain
    P0 = np.array([0, 0])
    P1 = np.array([1, 1])

    # Discretizations parameters
    T = 1.0  # final time
    D = 2  # how many times do we want to do solve our problem ?
    initial_power = 7
    final_power = 8
    time_array = np.ceil(np.logspace(initial_power, final_power, D, base=2.0)).astype(
        int)  # vector of total time steps, per time we do a time discretization
    N_array = np.ceil(np.sqrt(time_array)).astype(int)  # vector of mesh sizes

    err = 'uni'  # error to be outputted ('uni', 'L2', 'H1')

    def solve_obtain_error(self,RHS_fn,u_ref):
        '''
        helper function for test cases below. It runs a series of tests with different discretizations, and it then
        returns the corresponding errors. The error is only computed at the final time of every time discretization.
        '''

        # Errors tensor
        err_tot = np.zeros(self.D)

        # For loop computing errors
        for current_discr in range(self.D):

            # Space discretization
            N = self.N_array[current_discr]
            mesh, V = pde_utils.setup_rectangular_function_space(N, N, self.P0, self.P1)

            # Time discretization
            curr_time_steps = self.time_array[current_discr]
            dt, times = pde_utils.setup_time_discretization(self.T, curr_time_steps)

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

            # Solving
            for n in range(curr_time_steps):

                # Time update
                t = times[n+1]
                u_ref.t = t
                RHS_fn.t = t     # NB. This change is also reflected inside LHS_int, RHS_int

                # Solution at this time (no BC!) (NB. It's not very explicit that time changed, but it did above)
                u_current = pde_utils.solve_vp(V, LHS_int, RHS_int)

                # Update previous solution
                u_previous.assign(u_current)

            # Saving the error only at the last timestep
            if self.err == 'uni':
                error = pde_utils.error_LInf_piece_lin(u_ref, u_current, mesh)
            elif self.err == 'L2':
                error = fc.errornorm(u_ref, u_current, 'L2')
            else:
                error = fc.errornorm(u_ref, u_current, 'H1')

            print('Step = % .2f' % current_discr)

            err_tot[current_discr] = error

        return err_tot

    def test_easy_polynomial(self):

        '''
        Expected behaviour: the exact solution (up to machine precision) is computed
        '''

        RHS_fn = fc.Expression('3*pow(x[0],2) - 2*pow(x[0],3) + (3 - 2*x[1])*pow(x[1],2) + 12*t*(-1 + x[0] + x[1])',
                       degree=2, t=0)
        u_ref = fc.Expression('t*(3*pow(x[0],2) - 2*pow(x[0],3) + (3 - 2*x[1])*pow(x[1],2))', degree=2, t=0)
        err_tot = self.solve_obtain_error(RHS_fn, u_ref)
        raise Exception('Error test not implemented')

    def test_constant(self):

        '''
        Expected behaviour: the exact solution (up to machine precision) is computed
        '''

        RHS_fn = fc.Expression('1', degree=2)
        u_ref = fc.Expression('t', degree=2, t=0)

        err_tot = self.solve_obtain_error(RHS_fn, u_ref)
        raise Exception('Error test not implemented')

    def test_moving_bump(self):

        '''
        Expected behaviour: loglog(time_array, err_tot) and loglog(time_array, 1/time_array) should be parallel, in the
        L2 norm
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

        err_tot = self.solve_obtain_error(RHS_fn, u_ref)
        raise Exception('Error test not implemented')

    def test_polynomial(self):

        '''
        Expected behaviour: loglog(time_array, err_tot) and loglog(time_array, 1/time_array) should be parallel, in the
        L2 norm
        '''

        RHS_fn = fc.Expression(
            '3*(-1 + x[0] + x[1]) - 3*(-1 + x[0] + x[1])*cos(3*t) + (3*(3*pow(x[0],2) - 2*pow(x[0],3) + (3 - '
            '2*x[1])*pow(x[1],2))*sin(3*t))/4', degree=2, t=0)
        u_ref = fc.Expression('((-3*pow(x[0],2) + 2*pow(x[0],3) + pow(x[1],2)*(-3 + 2*x[1]))*(-1 + cos(3*t)))/4',
                            degree=2, t=0)

        err_tot = self.solve_obtain_error(RHS_fn, u_ref)
        raise Exception('Error test not implemented')
class TestInterpolators(unittest.TestCase):
    '''
    Tests the non-native interpolators.
    '''
    def setUp(self):
        self.n = 50
        self.mesh, self.fn_space = pde_utils.setup_function_space(50)

    def test_fenics_interpolate_bilin(self):

        affine_fc = fc.Expression('1 + 3*x[0] + 4*x[1]',
            element=self.fn_space.ufl_element())
        affine_np = pde_utils.fenics_unit_square_function_wrap(self.mesh,self.n,affine_fc)
        X,Y = np.meshgrid(np.linspace(0,1,40),np.linspace(0,1,40),indexing='ij')
        eval_ref = 1 + 3*X + 4*Y
        eval_wrap = affine_np(np.stack((X,Y),axis=-1))
        np.testing.assert_almost_equal(eval_ref,eval_wrap)

    def test_fenics_lin_interpolator_rectangle_right(self):
        nx = 5
        ny = 6
        P0 = np.array([4, 1])
        P1 = np.array([10, 23])
        mesh, fn_space = pde_utils.setup_rectangular_function_space(nx, ny, P0, P1)

        u_fenics = fc.interpolate(fc.Expression('x[0]+pow(x[1],2)', degree=1), fn_space)
        wrap = pde_utils.FenicsRectangleLinearInterpolator(nx, ny, P0, P1, u_fenics)
        my_interp = wrap.get_interpolator

        P = np.array([6.41, 7.71])

        np.testing.assert_almost_equal(my_interp(P) - u_fenics(P), 0, decimal=10)

    def test_fenics_grad_interpolator_rectangle_right(self):
        nx = 5
        ny = 6
        P0 = np.array([4, 1])
        P1 = np.array([10, 23])
        mesh, fn_space = pde_utils.setup_rectangular_function_space(nx, ny, P0, P1)

        u_fenics = fc.interpolate(fc.Expression('x[0]+pow(x[0],3)/42+pow(x[1],2)', degree=1), fn_space)
        u_fenics_grad = pde_utils.fenics_grad(mesh, u_fenics)
        grad_approxim = pde_utils.FenicsRectangleVecInterpolator(nx, ny, P0, P1, u_fenics_grad)

        X,Y =np.meshgrid(np.linspace(4.01,9.995,24),
                        np.linspace(2.01,22.995,13),indexing='ij')
        coords = np.stack((X,Y),axis=-1)
        np.testing.assert_almost_equal(grad_approxim(coords) -
                                       pde_utils.native_fenics_eval_vec(u_fenics_grad,coords),
                                       0, decimal=6)

        coords = np.stack((np.linspace(4.01,9.995,24),np.linspace(2.01,9.995,24)),axis=-1)
        np.testing.assert_almost_equal(grad_approxim(coords) -
                                       pde_utils.native_fenics_eval_vec(u_fenics_grad,coords),
                                       0, decimal=6)


class TestFenicsFnWrap(unittest.TestCase):
    '''
    Tests the wrappers of fenics built-in functions.
    '''


    def setUp(self):
        self.mesh = fc.UnitSquareMesh(1,1,"crossed")
        self.fn_space = fc.FunctionSpace(self.mesh,'Lagrange',1)

    def test_fenics_fn_wrap(self):
        def test_xy(X,Y):
            ref_sol = affine_ref_f(X,Y)
            coords = np.stack((X,Y),axis=-1)
            calc_sol = pde_utils.native_fenics_eval_scalar(affine_fc,coords)
            np.testing.assert_array_almost_equal(calc_sol,ref_sol)

        affine_ref_f = lambda X,Y: 1 + 3*X + 4*Y
        affine_exp = fc.Expression('1 + 3*x[0] + 4*x[1]',
                                  element=self.fn_space.ufl_element())
        affine_fc = fc.interpolate(affine_exp,self.fn_space)

        # test random vectors
        X, Y = np.random.uniform(0, 1, (2, 5))
        test_xy(X, Y)

        #test 2d grid.
        X,Y = np.meshgrid(np.linspace(0,1,20),np.linspace(0,1,18),indexing='ij')
        test_xy(X,Y)

        #test point
        X,Y = np.array([0.3,.2])
        test_xy(X,Y)



    def test_fenics_grad_wrap_affine(self):
        def test_xy(X, Y):
            ref_sol = affine_grad_ref_f(X, Y)
            coords = np.stack((X, Y), axis=-1)
            calc_sol = pde_utils.native_fenics_eval_vec(grad_affine_fc, coords)
            np.testing.assert_array_almost_equal(calc_sol, ref_sol)

        affine_grad_ref_f = lambda X, Y: np.ones_like(X)[..., np.newaxis] * np.array([3, 4])
        affine_exp = fc.Expression('1 + 3*x[0] + 4*x[1]',
                                   element=self.fn_space.ufl_element())
        affine_fc = fc.interpolate(affine_exp, self.fn_space)
        grad_affine_fc = pde_utils.fenics_grad(self.mesh,affine_fc)
        # test random vectors
        X, Y = np.random.uniform(0, 1, (2, 5))
        test_xy(X, Y)

        #test 2d grid.
        X,Y = np.meshgrid(np.linspace(0,1,20),np.linspace(0,1,18),indexing='ij')
        test_xy(X,Y)

        #test point
        X,Y = np.array([0.3,.2])
        test_xy(X,Y)

    def test_fenics_grad_wrap_hat(self):
        def test_xy(X, Y):
            ref_sol = grad_hat_ref(X, Y)
            coords = np.stack((X, Y), axis=-1)
            calc_sol = pde_utils.native_fenics_eval_vec(grad_hat_fc, coords)
            np.testing.assert_array_almost_equal(calc_sol, ref_sol)

        # hat function
        hat_fc = fc.Expression('1 - 2*max(abs(x[0]-.5),abs(x[1]-.5))',
           degree=1)
        hat_fc_fenics = fc.interpolate(hat_fc,self.fn_space)
        grad_hat_fc = pde_utils.fenics_grad(self.mesh,hat_fc_fenics)
        # test random vectors
        X, Y = np.random.uniform(0, 1, (2, 5))
        test_xy(X, Y)

        #test 2d grid, without points on the boundary of each gradient jump.
        X,Y = np.meshgrid(np.linspace(0.01,.995,20),np.linspace(0.01,.992,18),indexing='ij')
        test_xy(X,Y)

        #test point
        X,Y = np.array([0.3,.2])
        test_xy(X,Y)