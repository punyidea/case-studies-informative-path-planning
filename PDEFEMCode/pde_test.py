import unittest
from unittest import TestCase
import PDEFEMCode.pde_utils as pde_utils
import fenics as fc
import numpy as np

class TestPDESolver(TestCase):
    @classmethod
    def setUpClass(self):
        n=100
        self.setup_function_mesh_cls(n)

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
        u_sol = pde_utils.solve_pde(self.fn_space, LHS_int, RHS_int)

        error_L2 = pde_utils.error_L2(u_ref, u_sol)
        error_LInf = pde_utils.error_LInf_piece_lin(u_ref, u_sol, self.mesh)
        return error_L2,error_LInf

    def test_PDE_solve_constant(self):
        # Constant function is supplied.
        # The solver is expected to return the same constant.

        RHS_fn = fc.Constant(3.0)
        u_ref = RHS_fn

        error_L2, error_LInf = self.solve_obtain_error(RHS_fn, u_ref)
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

        error_L2, error_LInf = self.solve_obtain_error(RHS_fn, u_ref)
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
        n_list = np.asarray(np.ceil(np.logspace(3,9,10, base=2)),
                                dtype= np.int32)
        error_L2_list,error_LInf = np.empty((2,10))
        for ind,n_grid in enumerate(n_list):
            self.setup_function_mesh(n_grid)
            self.solve_obtain_error(RHS_fn,u_ref)
            error_L2_list[ind], error_LInf[ind] = self.solve_obtain_error(RHS_fn,u_ref)

        OOC = -(np.log(error_L2_list[1:])-np.log(error_L2_list[:-1])) / \
                (np.log(n_list[1:])-np.log(n_list[:-1]))
        #check that we are within 0.03 of the desired order of convergence
        np.testing.assert_allclose(OOC,2,atol=.03)


class TestPDEWrap(unittest.TestCase):
    n = 50
    mesh, fn_space = pde_utils.setup_function_space(50)

    def test_fenics_function(self):

        affine_fc = fc.Expression('1 + 3*x[0] + 4*x[1]',
            element=self.fn_space.ufl_element())
        affine_np = pde_utils.fenics_unit_square_function_wrap(self.mesh,self.n,affine_fc)
        X,Y = np.meshgrid(np.linspace(0,1,40),np.linspace(0,1,40),indexing='ij')
        eval_ref = 1 + 3*X + 4*Y
        eval_wrap = affine_np(np.stack((X,Y),axis=-1))
        np.testing.assert_almost_equal(eval_ref,eval_wrap)
