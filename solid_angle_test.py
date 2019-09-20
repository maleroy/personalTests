"""solid_angle_test.py
Does an interactive 3D visualization of a point moving in spherical coordinates
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import axes3d


class Crane(object):
    """Crane class which contains all parameters

    Args:
        object (Crane): Physical dimensions and visualization parameters
    """
    def __init__(self):
        self.tow_x = 0.
        self.tow_y = 0.
        self.tow_z = 0.

        self.tow_h = 43.6  # 60
        self.jib_l = 61.07  # 68

        self.n_sect_2d = 9
        self.i_sect_2d = 360./self.n_sect_2d
        self.n_sect_3d = 4
        self.i_sect_3d = 90./self.n_sect_3d

        self.n_wireframe = 3

        self.n_sect_tot = self.n_sect_2d*self.n_sect_3d
        self.sect_passed = np.zeros(
            (self.n_sect_2d, self.n_sect_3d), dtype=bool)

    def clear_sect_passed(self):
        """Clears history of which sectors have already been passed through
        """
        self.sect_passed = np.zeros(
            (self.n_sect_2d, self.n_sect_3d), dtype=bool)


def main():
    """Does a 3D visualization of a point moving in spherical coordinates
    """
    myc = Crane()

    phi_l = np.linspace(0., 360., myc.n_sect_2d+1) - 360./(2*myc.n_sect_2d)
    theta_l = np.linspace(0., 90., myc.n_sect_3d+1)

    r_s = myc.jib_l
    phi_init = 0  # 60
    theta_init = 90  # 33.75

    fig = plt.figure(figsize=(6, 6))
    my_ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')

    axphi = plt.axes([0.1, 0.15, 0.2, 0.01])
    sphi = Slider(axphi, 'Azimuthal angle (phi)', phi_l[0], phi_l[-1],
                  valinit=phi_init, valstep=1, color="blue")

    axtheta = plt.axes([0.1, 0.1, 0.2, 0.01])
    sthet = Slider(axtheta, 'Polar angle (theta)', theta_l[0], theta_l[-1],
                   valinit=theta_init, valstep=1, color="blue")

    p_a, p_b, p_i = get_interval(phi_l, phi_init)
    phi = np.linspace(p_a, p_b, myc.n_wireframe+1)

    t_a, t_b, t_i = get_interval(theta_l, theta_init)
    theta = np.linspace(t_a, t_b, myc.n_wireframe+1)

    myc.sect_passed[p_i, t_i] = True

    msh_phi, msh_theta = np.meshgrid(phi, theta)
    msh_x, msh_y, msh_z = sph2car(r_s, msh_phi, msh_theta)
    p_x, p_y, p_z = sph2car(r_s, phi_init, theta_init)

    msh_x += myc.tow_x
    msh_y += myc.tow_y
    msh_z += myc.tow_z + myc.tow_h

    p_x += myc.tow_x
    p_y += myc.tow_y
    p_z += myc.tow_z + myc.tow_h

    plot_all(my_ax, myc, msh_x, msh_y, msh_z, p_x, p_y, p_z, phi_init)

    def update(val):
        my_ax.clear()

        new_phi = sphi.val
        p_l, p_r, p_i = get_interval(phi_l, new_phi)
        phi = np.linspace(p_l, p_r, myc.n_wireframe+1)

        new_theta = sthet.val
        t_l, t_r, t_i = get_interval(theta_l, new_theta)
        theta = np.linspace(t_l, t_r, myc.n_wireframe+1)

        myc.sect_passed[p_i, t_i] = True
        plot_sect_hist(my_ax, myc)

        msh_phi, msh_theta = np.meshgrid(phi, theta)
        msh_x, msh_y, msh_z = sph2car(r_s, msh_phi, msh_theta)
        p_x, p_y, p_z = sph2car(r_s, new_phi, new_theta)

        msh_x += myc.tow_x
        msh_y += myc.tow_y
        msh_z += myc.tow_z + myc.tow_h

        p_x += myc.tow_x
        p_y += myc.tow_y
        p_z += myc.tow_z + myc.tow_h

        plot_all(my_ax, myc, msh_x, msh_y, msh_z, p_x, p_y, p_z, new_phi)

        fig.canvas.draw_idle()

    # Initial update now that everything is set up
    update(None)

    sthet.on_changed(update)
    sphi.on_changed(update)

    #my_ax.view_init(45, 0)
    my_ax.view_init(0, 90)
    set_axes_equal(my_ax)

    def press(event):
        if event.key == 'right':
            n_v = (
                sphi.valmax if sphi.val+5 > sphi.valmax else sphi.val+5)
            sphi.set_val(n_v)
        elif event.key == 'left':
            n_v = (
                sphi.valmin if sphi.val-5 < sphi.valmin else sphi.val-5)
            sphi.set_val(n_v)
        elif event.key == 'up':
            n_v = (
                sthet.valmin if sthet.val-5 < sthet.valmin else sthet.val-5)
            sthet.set_val(n_v)
        elif event.key == 'down':
            n_v = (
                sthet.valmax if sthet.val+5 > sthet.valmax else sthet.val+5)
            sthet.set_val(n_v)
        elif event.key == 'c':
            myc.clear_sect_passed()
            update(None)
        else:
            pass

    fig.canvas.mpl_connect('key_press_event', press)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


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
            # print("{} is between {} and {}".format(val, lst[i], lst[i+1]))
            return lst[i], lst[i+1], i
    print("Error")
    sys.exit()


def plot_all(my_ax, myc, msh_x, msh_y, msh_z, p_x, p_y, p_z, cur_phi):
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
        [myc.tow_x, myc.tow_x, p_x], [myc.tow_y, myc.tow_y, p_y],
        [myc.tow_z, myc.tow_z+myc.tow_h, p_z], c="green")
    my_ax.scatter([p_x, p_x], [p_y, p_y], [0, p_z], s=50, c="green")

    delta_h = p_z - 0
    delta_r = delta_h*np.tan(np.radians(0.5*45.4))
    delta_t = delta_h*np.tan(np.radians(0.5*64.2))
    delta_m = np.sqrt(delta_r**2+delta_t**2)
    hfov_d = np.degrees(np.arctan2(delta_m, delta_h))

    p_r = np.sqrt(p_x**2+p_y**2)

    pxs = [(p_r - delta_r)*np.cos(np.radians(cur_phi)),
           (p_r + delta_r)*np.cos(np.radians(cur_phi)),
           p_x - delta_t*np.sin(np.radians(cur_phi)),
           p_x + delta_t*np.sin(np.radians(cur_phi))]
    pys = [(p_r - delta_r)*np.sin(np.radians(cur_phi)),
           (p_r + delta_r)*np.sin(np.radians(cur_phi)),
           p_y + delta_t*np.cos(np.radians(cur_phi)),
           p_y - delta_t*np.cos(np.radians(cur_phi))]
    pzs = [0, 0, 0, 0]
    my_ax.scatter(pxs, pys, pzs, s=50, c="green")

    pxs = [p_r*np.cos(np.radians(cur_phi))-delta_m*np.sin(np.radians(hfov_d+cur_phi)),
           p_r*np.cos(np.radians(cur_phi))+delta_m*np.sin(np.radians(hfov_d-cur_phi)),
           p_r*np.cos(np.radians(cur_phi))-delta_m*np.sin(np.radians(hfov_d-cur_phi)),
           p_r*np.cos(np.radians(cur_phi))+delta_m*np.sin(np.radians(hfov_d+cur_phi))]
    pys = [p_r*np.sin(np.radians(cur_phi))+delta_m*np.cos(np.radians(hfov_d+cur_phi)),
           p_r*np.sin(np.radians(cur_phi))+delta_m*np.cos(np.radians(hfov_d-cur_phi)),
           p_r*np.sin(np.radians(cur_phi))-delta_m*np.cos(np.radians(hfov_d-cur_phi)),
           p_r*np.sin(np.radians(cur_phi))-delta_m*np.cos(np.radians(hfov_d+cur_phi))]
    pzs = [0, 0, 0, 0]
    my_ax.scatter(pxs, pys, pzs, s=50, c="blue")

    my_ax.set_xlabel('X axis')
    my_ax.set_ylabel('Y axis')
    my_ax.set_zlabel('Z axis')
    b_b = 1.05*myc.jib_l
    my_ax.set_xlim3d([myc.tow_x-b_b, myc.tow_x+b_b])
    my_ax.set_ylim3d([myc.tow_y-b_b, myc.tow_y+b_b])
    my_ax.set_zlim3d([myc.tow_z, myc.tow_z+myc.tow_h+b_b])
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


def plot_sect_hist(my_ax, myc):
    """Plots all sectors that have already been passed

    Args:
        my_ax (plt figure subplot): Where sectors should be plotted
    """
    for i in range(myc.sect_passed.shape[0]):
        for j in range(myc.sect_passed.shape[1]):
            if myc.sect_passed[i, j]:
                old_phi = np.linspace(
                    i*myc.i_sect_2d - 360./(2*myc.n_sect_2d),
                    (i+1)*myc.i_sect_2d - 360./(2*myc.n_sect_2d),
                    myc.n_wireframe+1)
                old_theta = np.linspace(
                    j*myc.i_sect_3d,
                    (j+1)*myc.i_sect_3d,
                    myc.n_wireframe+1)
                old_msh_phi, old_msh_theta = np.meshgrid(
                    old_phi, old_theta)
                msh_x, msh_y, msh_z = sph2car(
                    myc.jib_l, old_msh_phi, old_msh_theta)

                msh_x += myc.tow_x
                msh_y += myc.tow_y
                msh_z += myc.tow_z + myc.tow_h

                my_ax.plot_surface(
                    msh_x, msh_y, msh_z, color="blue", alpha=0.1)


if __name__ == '__main__':
    main()
