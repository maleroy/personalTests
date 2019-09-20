"""solid_angle_test.py
Does an interactive 3D visualization of a point moving in spherical coordinates
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import axes3d

TOW_X = 0.
TOW_Y = 0.
TOW_Z = 0.
TOW_H = 43.6  # 60
JIB_L = 61.07  # 68

N_SECT_2D = 9
I_SECT_2D = 360./N_SECT_2D
N_SECT_3D = 4
I_SECT_3D = 90./N_SECT_3D

N_WIREFRAME = 3

N_SECT_TOT = N_SECT_2D*N_SECT_3D
SECT_PASSED = np.zeros((N_SECT_2D, N_SECT_3D), dtype=bool)


def sph2car(r_s, phi, theta):
    """Converts spherical coordinates into cartesian coordinates

    Args:
        r_s (float): radius
        phi (float): azimuthal angle in [deg]
        theta (float): polar angle in [deg]

    Returns:
        3 floats: the 3 cartesian coordinates
    """
    x_c = r_s*np.sin(np.radians(theta))*np.cos(np.radians(phi))
    y_c = r_s*np.sin(np.radians(theta))*np.sin(np.radians(phi))
    z_c = r_s*np.cos(np.radians(theta))
    return x_c, y_c, z_c


def get_interval(lst, val):
    """Determines the interval a value is in

    Args:
        lst (np.array): list of intervals
        val (float): value whose interval must be found

    Returns:
        2 floats and 1 int: the interval boundaries and the left index
    """
    for i in range(len(lst)-1):
        if lst[i] <= val <= lst[i+1]:
            print("{} is between {} and {}".format(val, lst[i], lst[i+1]))
            return lst[i], lst[i+1], i
    print("Error")
    sys.exit()


def plot_all(my_ax, msh_x, msh_y, msh_z, p_x, p_y, p_z):
    """Plots a 3D surface as well as the line from center to current point

    Args:
        my_ax (plt figure 3D subplot): where results are plottes
        msh_x (np.array): current mesh x coordinates
        msh_y (np.array): current mesh y coordinates
        msh_z (np.array): current mesh z coordinates
        p_x (float): current point x coordinate
        p_y (float): current point y coordinate
        p_z (float): current point z coordinate
    """
    my_ax.plot_wireframe(msh_x, msh_y, msh_z, colors="green")
    my_ax.plot(
        [TOW_X, TOW_X, p_x], [TOW_Y, TOW_Y, p_y], [TOW_Z, TOW_Z+TOW_H, p_z],
        c="green")
    my_ax.scatter(p_x, p_y, p_z, s=50, c="green")

    my_ax.set_xlabel('X axis')
    my_ax.set_ylabel('Y axis')
    my_ax.set_zlabel('Z axis')
    b_b = 1.05*JIB_L
    my_ax.set_xlim3d([TOW_X-b_b, TOW_X+b_b])
    my_ax.set_ylim3d([TOW_Y-b_b, TOW_Y+b_b])
    my_ax.set_zlim3d([TOW_Z, TOW_Z+TOW_H+b_b])
    my_ax.set_aspect('equal')
    set_axes_equal(my_ax)


def set_axes_equal(my_ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    my_ax.set_aspect('equal') and my_ax.axis('equal') not working for 3D.

    Input
      my_ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = my_ax.get_xlim3d()
    y_limits = my_ax.get_ylim3d()
    z_limits = my_ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    my_ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    my_ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    my_ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def main():
    """Does a 3D visualization of a point moving in spherical coordinates
    """
    phi_l = np.linspace(0., 360., N_SECT_2D+1) - 360./(2*N_SECT_2D)
    theta_l = np.linspace(0., 90., N_SECT_3D+1)
    # print(phi_l)
    # print(theta_l)

    r_s = JIB_L
    phi_init = 60
    theta_init = 33.75

    fig = plt.figure(figsize=(6, 6))
    my_ax = fig.add_subplot(111, projection='3d')

    axphi = plt.axes([0.1, 0.15, 0.2, 0.01])
    sphi = Slider(axphi, 'Azimuthal angle (phi)', phi_l[0], phi_l[-1],
                  valinit=phi_init, valstep=1, color="blue")

    axtheta = plt.axes([0.1, 0.1, 0.2, 0.01])
    stheta = Slider(axtheta, 'Polar angle (theta)', theta_l[0], theta_l[-1],
                    valinit=theta_init, valstep=1, color="blue")

    p_a, p_b, p_i = get_interval(phi_l, phi_init)
    phi = np.linspace(p_a, p_b, N_WIREFRAME+1)

    t_a, t_b, t_i = get_interval(theta_l, theta_init)
    theta = np.linspace(t_a, t_b, N_WIREFRAME+1)

    SECT_PASSED[p_i, t_i] = True

    msh_phi, msh_theta = np.meshgrid(phi, theta)
    msh_x, msh_y, msh_z = sph2car(r_s, msh_phi, msh_theta)
    p_x, p_y, p_z = sph2car(r_s, phi_init, theta_init)

    msh_x += TOW_X
    msh_y += TOW_Y
    msh_z += TOW_Z + TOW_H

    p_x += TOW_X
    p_y += TOW_Y
    p_z += TOW_Z + TOW_H

    plot_all(my_ax, msh_x, msh_y, msh_z, p_x, p_y, p_z)

    def update(val):
        my_ax.clear()

        new_phi = sphi.val
        p_l, p_r, p_i = get_interval(phi_l, new_phi)
        phi = np.linspace(p_l, p_r, N_WIREFRAME+1)

        new_theta = stheta.val
        t_l, t_r, t_i = get_interval(theta_l, new_theta)
        theta = np.linspace(t_l, t_r, N_WIREFRAME+1)

        SECT_PASSED[p_i, t_i] = True
        for i in range(SECT_PASSED.shape[0]):
            for j in range(SECT_PASSED.shape[1]):
                if SECT_PASSED[i, j]:
                    old_phi = np.linspace(
                        i*I_SECT_2D - 360./(2*N_SECT_2D),
                        (i+1)*I_SECT_2D - 360./(2*N_SECT_2D),
                        N_WIREFRAME+1)
                    old_theta = np.linspace(
                        j*I_SECT_3D,
                        (j+1)*I_SECT_3D,
                        N_WIREFRAME+1)
                    old_msh_phi, old_msh_theta = np.meshgrid(
                        old_phi, old_theta)
                    msh_x, msh_y, msh_z = sph2car(
                        r_s, old_msh_phi, old_msh_theta)

                    msh_x += TOW_X
                    msh_y += TOW_Y
                    msh_z += TOW_Z + TOW_H

                    my_ax.plot_surface(
                        msh_x, msh_y, msh_z, color="blue", alpha=0.1)

        msh_phi, msh_theta = np.meshgrid(phi, theta)
        msh_x, msh_y, msh_z = sph2car(r_s, msh_phi, msh_theta)
        p_x, p_y, p_z = sph2car(r_s, new_phi, new_theta)

        msh_x += TOW_X
        msh_y += TOW_Y
        msh_z += TOW_Z + TOW_H

        p_x += TOW_X
        p_y += TOW_Y
        p_z += TOW_Z + TOW_H

        plot_all(my_ax, msh_x, msh_y, msh_z, p_x, p_y, p_z)

        fig.canvas.draw_idle()

    # Initial update now that everything is set up
    update(None)

    stheta.on_changed(update)
    sphi.on_changed(update)

    my_ax.view_init(45, 0)
    set_axes_equal(my_ax)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


if __name__ == '__main__':
    main()
