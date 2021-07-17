'''
Contains a quick plot of a basis function in Friedrichs-Keller triangulation used in the Presentation.
'''
import numpy as np
import fenics as fc
from PDEFEMCode import interface as pde_IO
import matplotlib.pyplot as plt


def plot_FK_basis_func():
    mesh = fc.UnitSquareMesh(2, 2, diagonal='right')
    fn_space = fc.FunctionSpace(mesh, 'Lagrange', 1)

    hat_fc = fc.Expression('1 - 2*max(abs(x[0]-.5),abs(x[1]-.5))',
                                   degree=1)
    hat_fc_fenics = fc.interpolate(hat_fc, fn_space)

    x =y = np.linspace(0,1,31)
    X,Y = np.meshgrid(x,y)
    coords = np.stack((X,Y),axis =-1)
    hat_eval = pde_IO.native_fenics_eval_scalar(hat_fc_fenics,coords)

    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, hat_eval, cmap='viridis', edgecolor='none')
    #ax.set_title('Surface plot')
    ax.axis('off')
    plt.show()
