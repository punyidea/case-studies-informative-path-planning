'''
This file is a  working example of an (time independent) PDE solver.
As structured, it reads in parameters from a .yaml file, (see elliptic_params.yaml for an example).
The parameters determine a single variational form which FEniCS will solve.

In the context of the project, this file was used to generate the
    elliptic PDE test cases with a single and multiple Gaussian PDE
    source terms on the RHS. See parameter files with more detail.

On a high level, the code
    (0) reads in parameters from a .yaml file and parses them into an EllipticRunParams struct.
    (1) solves a single variational form of form LHS = RHS.
    (2) generates custom numpy interpolators of the solution function and its gradient
    (3) saves the results out to a .pickle file.

The code is run as follows:
    python elliptic_ex.py -y param_file.yaml

    where param_file.yaml is structured like the example elliptic_params.yaml test case which generated.
    A result pickle file will be saved where the structure's io params determine.
'''

import PDEFEMCode.fenics_utils as pde_utils
import PDEFEMCode.interface as pde_interface
import fenics as fc


def solve_single_variation_eq(in_params):
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
    mesh_p = in_params.rect_mesh
    io_p = in_params.io
    # Organize all function handles specified by var_form_p in a separate struct.
    var_form_fn_handles = pde_utils.VarFormFnHandles(var_form_p)

    # Set up function mesh and FEniCS trial and test function spaces.
    mesh, fn_space = pde_utils.setup_rectangular_function_space(mesh_p)
    u_trial = fc.TrialFunction(fn_space)
    v_test = fc.TestFunction(fn_space)

    # Setup variational formulation, tying the LHS form with the trial function
    # and the RHS form with the test functions and the RHS function.
    if not var_form_p.rhs_expression_str:
        var_form_p.rhs_expression_str = var_form_fn_handles.rhs_expression(**var_form_p.rhs_exp_params)
    RHS_fn = fc.Expression(var_form_p.rhs_expression_str, element = fn_space.ufl_element())
    LHS_int, RHS_int = pde_utils.variational_formulation(
        u_trial, v_test,
        var_form_fn_handles.LHS,
        var_form_fn_handles.RHS, RHS_fn
    )

    # Solve VP and obtain solution functions, such that they work with numpy.
    u_sol = pde_utils.solve_vp(fn_space,LHS_int,RHS_int)
    f = pde_interface.FenicsRectangleLinearInterpolator(mesh_p, u_sol)
    u_grad = pde_utils.fenics_grad(mesh,u_sol)
    grad_f = pde_interface.FenicsRectangleVecInterpolator(mesh_p, u_grad)

    # Organize results, and return
    param_save = {'f':f,'grad_f':grad_f,'params':in_params}
    return param_save, u_sol,u_grad

if __name__ == '__main__':
    params_yaml,args = pde_interface.parse_args_cli()
    # Organize the semi-structured parameters in the yaml file into Python dataclass objects used in our code.
    in_params = pde_utils.yaml_parse_elliptic(params_yaml, args.yaml_fname)

    param_save, _, _ = solve_single_variation_eq(in_params)
    pde_interface.pickle_save(in_params.io.out_folder, in_params.io.out_file_prefix, param_save)

