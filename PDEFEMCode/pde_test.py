from unittest import TestCase
import PDEFEMCode.pde_utils as pde_utils
import fenics as fc
import numpy as np

class TestPDE(TestCase):
    mesh,fn_space = pde_utils.setup_function_space(50)
    LHS = staticmethod(pde_utils.elliptic_LHS)
    RHS = staticmethod(pde_utils.elliptic_RHS)

    u_trial = fc.TrialFunction(fn_space)
    v_test = fc.TestFunction(fn_space)

    def test_PDE_solve_constant(self):
        # Constant function is supplied.
        # The solver is expected to return the same constant.

        RHS_fn = fc.Constant(3.0)
        u_ref = RHS_fn

        LHS_int, RHS_int = pde_utils.variational_formulation(
            self.u_trial,self.v_test,
            self.LHS,
            self.RHS, RHS_fn
        )
        u_sol = pde_utils.solve_pde(self.fn_space,LHS_int,RHS_int)

        error_L2 = pde_utils.error_L2(u_ref, u_sol)
        error_LInf = pde_utils.error_LInf_piece_lin(u_ref,u_sol,self.mesh)

        print('Constant function error_L2  =', error_L2)
        print('Constant function error_Linf =', error_LInf)
        np.testing.assert_almost_equal(error_L2,decimal=10)
        np.testing.assert_almost_equal(error_LInf, decimal=10)

    def test_PDE_solve_sines(self):
        # Constant function is supplied.
        # The solver is expected to return the same constant.

        RHS_fn = fc.Expression('(2*pi*pi + 1)*sin(pi*x[0] + pi/2)*sin(pi*x[1]+pi/2)',
            element = self.fn_space.ufl_element())
        u_ref = fc.Expression('sin(pi*x[0] + pi/2)*sin(pi*x[1]+pi/2)',
                element = self.fn_space.ufl_element())

        LHS_int, RHS_int = pde_utils.variational_formulation(
            self.u_trial, self.v_test,
            self.LHS,
            self.RHS, RHS_fn
        )
        u_sol = pde_utils.solve_pde(self.fn_space, LHS_int, RHS_int)

        error_L2 = pde_utils.error_L2(u_ref, u_sol)
        error_LInf = pde_utils.error_LInf_piece_lin(u_ref, u_sol, self.mesh)

        print('product sines function error_L2  =', error_L2)
        print('product sines function error_Linf =', error_LInf)
        np.testing.assert_almost_equal(error_L2, decimal=3)
        np.testing.assert_almost_equal(error_LInf, decimal=3)
        # 5e-4 error reported in test MATLAB code.

