"""solid_angle_test.py
Does an interactive 3D visualization of a point moving in spherical coordinates
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import axes3d

PLOT_CORNERS_RADIAL = False
PLOT_CORNERS_EDGE = False
USE_CASE = True

np.set_printoptions(sign='+', precision=2, suppress=True)


class Crane(object):
    """Crane class which contains all parameters

    Args:
        object (Crane): Physical dimensions and visualization parameters
    """
    def __init__(self):
        self.tow_x = 0.
        self.tow_y = 0.
        self.tow_z = 0.

        self.tow_h = 100  # 43.6  # 60
        self.jib_l = 100  # 61.07  # 68

        self.hfov_v = 0.5*45.4
        self.hfov_h = 0.5*64.2

        self.n_sect_2d = 9
        self.i_sect_2d = 360./self.n_sect_2d
        self.n_sect_3d = 4
        self.i_sect_3d = 90./self.n_sect_3d

        self.n_wireframe = 3

        self.n_sect_tot = self.n_sect_2d*self.n_sect_3d
        self.sect_passed = np.zeros(
            (self.n_sect_2d, self.n_sect_3d), dtype=bool)

        self.sect_passed_2d = np.zeros((self.n_sect_2d, 4))
        self.prev_2d_sect = -1

        self.plot_cur_footprint = True
        self.plot_footprint_hist = False

        self.fix_ang = 30
        self.fix_ang_rad = np.radians(self.fix_ang)

    def clear_sect_passed(self):
        """Clears history of which sectors have already been passed through
        """
        self.sect_passed_2d = np.zeros((self.n_sect_2d, 4))
        self.sect_passed = np.zeros(
            (self.n_sect_2d, self.n_sect_3d), dtype=bool)
        self.prev_2d_sect = -1


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
    my_ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

    axphi = plt.axes([0.1, 0.15, 0.2, 0.01])
    sphi = Slider(axphi, 'Azimuthal angle (phi)', phi_l[0], phi_l[-1],
                  valinit=phi_init, valstep=1, color="blue")

    axtheta = plt.axes([0.1, 0.1, 0.2, 0.01])
    sthet = Slider(axtheta, 'Polar angle (theta)', theta_l[0], theta_l[-1],
                   valinit=theta_init, valstep=1, color="blue")

    axfix = plt.axes([0.1, 0.05, 0.2, 0.01])
    sfix = Slider(axfix, 'Cam fixed angle', 0, 80,
                  valinit=myc.fix_ang, valstep=5, color="blue")

    p_a, p_b, p_i = get_interval(phi_l, phi_init)
    phi = np.linspace(p_a, p_b, myc.n_wireframe+1)

    t_a, t_b, t_i = get_interval(theta_l, theta_init)
    theta = np.linspace(t_a, t_b, myc.n_wireframe+1)

    myc.sect_passed[p_i, t_i] = True

    msh_phi, msh_theta = np.meshgrid(phi, theta)
    msh_x, msh_y, msh_z = sph2car(r_s, msh_phi, msh_theta)
    p_x, p_y, p_z = sph2car(r_s, phi_init, theta_init)

    if not p_i == myc.prev_2d_sect:
        myc.sect_passed_2d[p_i] = [p_x, p_y, p_z, phi_init]
        myc.prev_2d_sect = p_i

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

        myc.fix_ang = sfix.val
        myc.fix_ang_rad = np.radians(sfix.val)

        myc.sect_passed[p_i, t_i] = True
        plot_sect_hist(my_ax, myc)

        msh_phi, msh_theta = np.meshgrid(phi, theta)
        msh_x, msh_y, msh_z = sph2car(r_s, msh_phi, msh_theta)
        p_x, p_y, p_z = sph2car(r_s, new_phi, new_theta)

        if not p_i == myc.prev_2d_sect:
            myc.sect_passed_2d[p_i] = [p_x, p_y, p_z, new_phi]
            myc.prev_2d_sect = p_i

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

    my_ax.view_init(45, 0)
    set_axes_equal(my_ax, myc)

    def press(event):
        if event.key == 'right':
            n_v = (
                sphi.valmax if sphi.val+sphi.valstep*5 > sphi.valmax else (
                    sphi.val+sphi.valstep*5))
            sphi.set_val(n_v)
        elif event.key == 'left':
            n_v = (
                sphi.valmin if sphi.val-sphi.valstep*5 < sphi.valmin else (
                    sphi.val-sphi.valstep*5))
            sphi.set_val(n_v)
        elif event.key == 'up':
            n_v = (
                sthet.valmin if sthet.val-sthet.valstep*5 < sthet.valmin else (
                    sthet.val-sthet.valstep*5))
            sthet.set_val(n_v)
        elif event.key == 'down':
            n_v = (
                sthet.valmax if sthet.val+sthet.valstep*5 > sthet.valmax else (
                    sthet.val+sthet.valstep*5))
            sthet.set_val(n_v)
        elif event.key == 'y':
            n_v = (
                sfix.valmin if sfix.val-sfix.valstep < sfix.valmin else (
                    sfix.val-sfix.valstep))
            sfix.set_val(n_v)
        elif event.key == 'x':
            n_v = (
                sfix.valmax if sfix.val+sfix.valstep > sfix.valmax else (
                    sfix.val+sfix.valstep))
            sfix.set_val(n_v)
        elif event.key == 'c':
            myc.clear_sect_passed()
        elif event.key == 'h':
            myc.plot_cur_footprint = not myc.plot_cur_footprint
        elif event.key == 'j':
            myc.plot_footprint_hist = not myc.plot_footprint_hist
        elif event.key == 't':
            my_ax.view_init(90, 0)
        elif event.key == 'f':
            my_ax.view_init(0, 0)
        elif event.key == 'o':
            my_ax.view_init(45, 0)
        else:
            pass
        update(None)

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
        my_ax (plt figure 3D subplot): where results are plotted
        myc (Crane): instance containing all physical parameters and history
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
    my_ax.scatter(p_x, p_y, p_z, s=50, c="green")

    if myc.plot_cur_footprint:
        plot_footprint(my_ax, myc, p_x, p_y, p_z, cur_phi, "green", 0.75)

    if myc.plot_footprint_hist:
        plot_footprint_hist(my_ax, myc)

    set_axes_equal(my_ax, myc)


def plot_footprint(my_ax, myc, p_x, p_y, p_z, cur_phi, colr="black", alp=0.1):
    """Plots a single camera footprint on the ground

    Args:
        my_ax (plt figure 3D subplot): where results are plotted
        myc (Crane): instance containing all physical parameters and history
        p_x (float): current point x coordinate
        p_y (float): current point y coordinate
        p_z (float): current point z coordinate
        cur_phi ([type]): [description]
        colr (str, optional): color of surface. Defaults to "black".
        alp (float, optional): alpha value of surface. Defaults to 0.1.
    """
    my_ax.scatter(p_x, p_y, 0, s=50, c="green")

    delta_h = p_z - 0
    delta_r = delta_h*np.tan(np.radians(myc.hfov_v))
    delta_t = delta_h*np.tan(np.radians(myc.hfov_h))
    delta_m = np.sqrt(delta_r**2+delta_t**2)

    if USE_CASE:
        hfov_d = np.degrees(np.arctan2(delta_h, delta_m))  # why not dm, dh??
    else:
        hfov_d = np.degrees(np.arctan2(delta_m, delta_h))

    if not myc.fix_ang < 0.0:
        plot_skewed_footprint(my_ax, myc, p_x, p_y, p_z, cur_phi, delta_h,
                              delta_r, delta_t, delta_m, hfov_d)

    p_r = np.sqrt(p_x**2+p_y**2)
    p_r_in = p_r - delta_r
    p_r_out = p_r + delta_r

    cur_phi_rad = np.radians(cur_phi)
    p_r_in_x = p_r_in*np.cos(cur_phi_rad)
    p_r_in_y = p_r_in*np.sin(cur_phi_rad)
    p_r_out_x = p_r_out*np.cos(cur_phi_rad)
    p_r_out_y = p_r_out*np.sin(cur_phi_rad)

    p_r_in_x_ll = p_r_in_x - delta_t*np.sin(cur_phi_rad)
    p_r_in_y_ll = p_r_in_y + delta_t*np.cos(cur_phi_rad)

    p_r_in_x_ul = p_r_out_x - delta_t*np.sin(cur_phi_rad)
    p_r_in_y_ul = p_r_out_y + delta_t*np.cos(cur_phi_rad)

    p_r_in_x_lr = p_r_in_x + delta_t*np.sin(cur_phi_rad)
    p_r_in_y_lr = p_r_in_y - delta_t*np.cos(cur_phi_rad)

    p_r_in_x_ur = p_r_out_x + delta_t*np.sin(cur_phi_rad)
    p_r_in_y_ur = p_r_out_y - delta_t*np.cos(cur_phi_rad)

    diag_phi_left = np.radians(cur_phi-hfov_d)
    diag_phi_right = np.radians(cur_phi+hfov_d)

    if USE_CASE:
        pxs = [p_r_in_x_ll, p_r_in_x_ul, p_r_in_x_ur, p_r_in_x_lr]

        pys = [p_r_in_y_ll, p_r_in_y_ul, p_r_in_y_ur, p_r_in_y_lr]
        
        """
        pxs = [p_x + delta_m*np.cos(diag_phi_right),
               p_x - delta_m*np.cos(diag_phi_left),
               p_x - delta_m*np.cos(diag_phi_right),
               p_x + delta_m*np.cos(diag_phi_left)]

        pys = [p_y + delta_m*np.sin(diag_phi_right),
               p_y - delta_m*np.sin(diag_phi_left),
               p_y - delta_m*np.sin(diag_phi_right),
               p_y + delta_m*np.sin(diag_phi_left)]
        """
    else:
        pxs = [p_x - delta_m*np.sin(diag_phi_left),
               p_x - delta_m*np.sin(diag_phi_right),
               p_x + delta_m*np.sin(diag_phi_left),
               p_x + delta_m*np.sin(diag_phi_right)]

        pys = [p_y + delta_m*np.cos(diag_phi_left),
               p_y + delta_m*np.cos(diag_phi_right),
               p_y - delta_m*np.cos(diag_phi_left),
               p_y - delta_m*np.cos(diag_phi_right)]

    pzs = [0, 0, 0, 0]

    try:
        my_ax.plot_trisurf(pxs, pys, pzs, color=colr, alpha=alp)
    except ValueError:
        print("No distinct points, skipping")

    if PLOT_CORNERS_EDGE:
        my_ax.scatter(pxs, pys, pzs, s=50, c="cyan")

    if PLOT_CORNERS_RADIAL:
        delta_rx = delta_r*np.cos(cur_phi_rad)
        delta_ry = delta_r*np.sin(cur_phi_rad)
        delta_tx = delta_t*np.sin(cur_phi_rad)
        delta_ty = delta_t*np.cos(cur_phi_rad)

        pxs = [p_x-delta_rx,
               p_x+delta_rx,
               p_x-delta_tx,
               p_x+delta_tx]

        pys = [p_y-delta_ry,
               p_y+delta_ry,
               p_y+delta_ty,
               p_y-delta_ty]

        pzs = [0, 0, 0, 0]

        my_ax.scatter(pxs, pys, pzs, s=50, c="cyan")


def plot_skewed_footprint(my_ax, myc, p_x, p_y, p_z, cur_phi, delta_h, delta_r,
                          delta_t, delta_m, hfov_d, colr="black", alp=0.25):
    """[summary]

    Args:
        my_ax ([type]): [description]
        myc ([type]): [description]
        p_x ([type]): [description]
        p_y ([type]): [description]
        p_z ([type]): [description]
        cur_phi ([type]): [description]
        delta_h ([type]): [description]
        delta_r ([type]): [description]
        delta_t ([type]): [description]
        delta_m ([type]): [description]
        hfov_d ([type]): [description]
        colr (str, optional): [description]. Defaults to "black".
        alp (float, optional): [description]. Defaults to 0.25.
    """
    if myc.hfov_v+myc.fix_ang < 90.0:
        delta_r_in = delta_h*np.tan(np.radians(myc.hfov_v-myc.fix_ang))
        delta_r_out = delta_h*np.tan(np.radians(myc.hfov_v+myc.fix_ang))

        print(np.array([delta_r, delta_r_in, delta_r_out,
                        np.sqrt(p_x**2+p_y**2), p_z, delta_m, hfov_d]))

    p_r_new = np.sqrt(p_x**2+p_y**2) - delta_h*np.tan(myc.fix_ang_rad)
    p_r_new_in = np.sqrt(p_x**2+p_y**2) - delta_h*np.tan(
        myc.fix_ang_rad+np.radians(myc.hfov_v))
    p_r_new_out = np.sqrt(p_x**2+p_y**2) - delta_h*np.tan(
        myc.fix_ang_rad-np.radians(myc.hfov_v))

    cur_phi_rad = np.radians(cur_phi)

    p_r_new_x = p_r_new*np.cos(cur_phi_rad)
    p_r_new_y = p_r_new*np.sin(cur_phi_rad)
    p_r_new_in_x = p_r_new_in*np.cos(cur_phi_rad)
    p_r_new_in_y = p_r_new_in*np.sin(cur_phi_rad)
    p_r_new_out_x = p_r_new_out*np.cos(cur_phi_rad)
    p_r_new_out_y = p_r_new_out*np.sin(cur_phi_rad)

    p_t_x_ll = p_r_new_in_x - delta_t*np.sin(cur_phi_rad)
    p_t_y_ll = p_r_new_in_y + delta_t*np.cos(cur_phi_rad)

    p_t_x_ul = p_r_new_out_x - delta_t*np.sin(cur_phi_rad)
    p_t_y_ul = p_r_new_out_y + delta_t*np.cos(cur_phi_rad)

    p_t_x_lr = p_r_new_in_x + delta_t*np.sin(cur_phi_rad)
    p_t_y_lr = p_r_new_in_y - delta_t*np.cos(cur_phi_rad)

    p_t_x_ur = p_r_new_out_x + delta_t*np.sin(cur_phi_rad)
    p_t_y_ur = p_r_new_out_y - delta_t*np.cos(cur_phi_rad)

    pxs = [p_r_new_x, p_r_new_in_x, p_r_new_out_x, p_t_x_ll, p_t_x_ul, p_t_x_lr, p_t_x_ur]
    pys = [p_r_new_y, p_r_new_in_y, p_r_new_out_y, p_t_y_ll, p_t_y_ul, p_t_y_lr, p_t_y_ur]
    pzs = [0, 0, 0, 0, 0, 0, 0]

    my_ax.scatter(pxs, pys, pzs, s=50, c=colr, alpha=alp)
    my_ax.plot_trisurf(pxs, pys, pzs, color=colr, alpha=alp)


def set_axes_equal(my_ax, myc):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    my_ax.set_aspect('equal') and my_ax.axis('equal') not working for 3D.

    Args
        my_ax (plt figure subplot): where sectors should be plotted
        myc (Crane): instance containing all physical parameters and history
    """
    my_ax.set_xlabel('X axis')
    my_ax.set_ylabel('Y axis')
    my_ax.set_zlabel('Z axis')
    b_b = 1.05*myc.jib_l
    my_ax.set_xlim3d([myc.tow_x-b_b, myc.tow_x+b_b])
    my_ax.set_ylim3d([myc.tow_y-b_b, myc.tow_y+b_b])
    my_ax.set_zlim3d([myc.tow_z, myc.tow_z+myc.tow_h+b_b])
    my_ax.set_aspect('equal')

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
        my_ax (plt figure subplot): where sectors should be plotted
        myc (Crane): instance containing all physical parameters and history
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


def plot_footprint_hist(my_ax, myc):
    """Plots the cam footprints on sectors that have already been passed over

    Args:
        my_ax (plt figure subplot): where sectors should be plotted
        myc (Crane): instance containing all physical parameters and history
    """
    for i in range(myc.sect_passed_2d.shape[0]):
        if myc.sect_passed_2d[i].any():
            my_ax.scatter(myc.sect_passed_2d[i, 0] + myc.tow_x,
                          myc.sect_passed_2d[i, 1] + myc.tow_y,
                          0, c="black")

            plot_footprint(my_ax,
                           myc,
                           myc.sect_passed_2d[i, 0] + myc.tow_x,
                           myc.sect_passed_2d[i, 1] + myc.tow_y,
                           myc.sect_passed_2d[i, 2] + myc.tow_z + myc.tow_h,
                           myc.sect_passed_2d[i, 3])


if __name__ == '__main__':
    main()
