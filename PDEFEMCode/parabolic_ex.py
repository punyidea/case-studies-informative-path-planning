'''
This file is a working example in the discretization of the heat equation with homogeneous Neumann condition, and zero
initial condition.
As structured, it reads in parameters from a .yaml file, (see elliptic_params.yaml for an example).
The parameters determine a series of variational equations which FEniCS will solve.

In the context of the project, this file was used to generate the
    parabolic PDE test cases with a single and multiple Gaussian PDE
    source terms on the RHS. See parameter files with more detail.

On a high level, the code:
    (0) reads in parameters from a .yaml file and parses them into an EllipticRunParams struct.
    (1) solves the heat equation at each time step using the implicit Euler method, generating
            equation LHS(t) = RHS(t).
    (2) generates custom numpy interpolators of the solution function and its gradient.
    (3) saves the results out to a .pickle file.

The code is run as follows:
    parabolic_ex -y param_file.yaml

    where param_file.yaml is structured like, e.g. par_moving_bump.yaml test case which generated.
    A result pickle file will be saved where the structure's io params determine.
'''


import PDEFEMCode.fenics_utils as pde_utils
import PDEFEMCode.interface as pde_interface
import fenics as fc
import numpy as np


def solve_parabolic_problem(in_params):
    '''
    Solves a single PDE with variational forms given in the parameter struct.
    The mesh structure is assumed to be a rectangular Friedrics-Keller triangulation,
        the same as the interpolator structure assumes.
    Input/Output:
    :param in_params: type EllipticRunParams
        A Dataclass struct containing all parameters necessary to run a single solution.
    :return: tuple:
        param_save: a dictionary containing:
            f: function interpolator class FenicsRectangleLinearInterpolator
            grad_f: gradient evaluator class    FenicsRectangleVecInterpolator
            param: the parameter struct used to run this script.
    '''

    # For convenience, separate master struct into separate variables.
    var_form_p = in_params.var_form
    var_form_fn_handles = pde_utils.VarFormFnHandles(var_form_p)
    mesh_p = in_params.rect_mesh
    time_disc_p = in_params.time_disc
    io_p = in_params.io

    # Space discretization
    mesh, fn_space = pde_utils.setup_rectangular_function_space(mesh_p)

    # Time discretization
    dt, times = pde_utils.setup_time_discretization(time_disc_p)

    # Setup variational formulation, tying the LHS form with the trial function
    # and the RHS form with the test functions and the RHS function.
    u_initial = fc.interpolate(fc.Constant(0), fn_space)  # the initial solution is zero according to our problem formulation
    u_trial = fc.TrialFunction(fn_space)
    v_test = fc.TestFunction(fn_space)

    if not var_form_p.rhs_expression_str:
        var_form_p.rhs_expression_str = var_form_fn_handles.rhs_expression(**var_form_p.rhs_exp_params)
    RHS_fn = fc.Expression(
        var_form_p.rhs_expression_str, degree=2, t=0)

    u_previous = u_initial
    LHS_int, RHS_int = pde_utils.variational_formulation(
        u_trial, v_test,
        var_form_fn_handles.LHS,
        var_form_fn_handles.RHS, RHS_fn,
        {'dt': dt},
        {'dt': dt, 'u_previous': u_previous}
    )

    t = times[0]  # initial time
    fenics_list = []  # list of Fenics functions to be passed to the interpolators

    # Start main solving block.
    fc.set_log_active(False)  # disable messages of Fenics
    for n in range(time_disc_p.Nt):
        # Time update
        t = times[n + 1]
        RHS_fn.t = t  # NB. This change is also reflected inside LHS_int, RHS_int

        # Solution at this time, and update previous solution
        u_current = pde_utils.solve_vp(fn_space, LHS_int, RHS_int)
        u_previous.assign(u_current)
        fenics_list.append(u_current)

        # Notify that something has happened
        print("Solving PDE, " + str(np.floor(100 * t/time_disc_p.T_fin)) + " % done.")

    # Prepending initial condition
    fenics_list.insert(0, fc.interpolate(fc.Constant(0), fn_space))

    # Obtaining the interpolator for u
    u = pde_interface.FenicsRectangleLinearInterpolator(mesh_p, fenics_list, T_fin=time_disc_p.T_fin, Nt=time_disc_p.Nt, time_dependent=True, verbose=True, time_as_indices=True)
    grad_u_list = [pde_utils.fenics_grad(mesh,u_fenics) for u_fenics in fenics_list]
    grad_u = pde_interface.FenicsRectangleVecInterpolator(mesh_p, grad_u_list, T_fin=time_disc_p.T_fin, Nt=time_disc_p.Nt, time_dependent=True, time_as_indices=True)

    # Organize and save final results.
    param_save = {'f':u,'grad_f':grad_u,'params':in_params}
    return param_save,fenics_list,grad_u_list

if __name__ == '__main__':
    params_yaml,args = pde_interface.parse_args_cli()
    # Organize the semi-structured parameters in the yaml file into Python dataclass objects used in our code.
    in_params = pde_utils.yaml_parse_parabolic(params_yaml, args.yaml_fname)

    param_save, _, _ = solve_parabolic_problem(in_params)
    pde_interface.pickle_save(in_params.io.out_folder, in_params.io.out_file_prefix, param_save)