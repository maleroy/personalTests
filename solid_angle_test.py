"""solid_angle_test.py
Does an interactive 3D visualization of a point moving in spherical coordinates
"""
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d import axes3d
import mpl_toolkits.mplot3d.art3d as art3d

np.set_printoptions(sign='+', precision=2, suppress=True)


class Crane(object):
    """Crane class which contains all parameters

    Args:
        object (Crane): Physical dimensions and visualization parameters
    """
    def __init__(self, d_conf=None):
        if d_conf == {}:
            # Characteristics from crane itself
            self.tow_x = 0.
            self.tow_y = 0.
            self.tow_z = 0.
            self.tow_h = 43.6
            self.jib_l = 61.07
            self.luf_ang = 85

            # Camera characteristics
            self.n_cams = 4
            self.k_cams = np.linspace(0.0, 1.0, self.n_cams+2)[1:-1]
            self.fix_ang = 30*np.ones(self.n_cams)
            self.cam_center_max_r = 2*self.jib_l//10*10

            # Capture characteristics
            self.n_sect_2d = 10
            self.n_sect_3d = 4
            self.hist_2d_only = False
            self.units = 'm'
            self.pics_per_slice = 3

            # Building characteristics
            self.bldg_h = 40
            self.bldg_d = 0
            self.bldg_w = 55
            self.bldg_x = -34
            self.bldg_y = 34

            # Booleans for plot
            self.plot_bldg = True
            self.plot_bldg_as_wedge = True
            self.plot_cur_footprint = True
            self.plot_footprint_hist = False
            self.plot_sect_hist = False

        else:
            self.tow_x = d_conf.get('crane').get('tow_x')
            self.tow_y = d_conf.get('crane').get('tow_y')
            self.tow_z = d_conf.get('crane').get('tow_z')
            self.tow_h = d_conf.get('crane').get('tow_h')
            self.jib_l = d_conf.get('crane').get('jib_l')
            self.luf_ang = d_conf.get('crane').get('init_luf_ang')

            self.n_cams = d_conf.get('cams').get('n_cams')
            self.k_cams = np.array(d_conf.get('cams').get('k_cams'))/self.jib_l
            self.fix_ang = np.array(d_conf.get('cams').get('fix_ang'))
            self.cam_center_max_r = d_conf.get('cams').get('cam_center_max_r')

            self.n_sect_2d = d_conf.get('capture').get('n_sect_2d')
            self.n_sect_3d = d_conf.get('capture').get('n_sect_3d')
            self.hist_2d_only = d_conf.get('capture').get('hist_2d_only')
            self.units = d_conf.get('capture').get('units')
            self.pics_per_slice = d_conf.get('capture').get('pics_per_slice')

            self.bldg_h = d_conf.get('building').get('bldg_h')
            self.bldg_d = d_conf.get('building').get('bldg_d')
            self.bldg_w = d_conf.get('building').get('bldg_w')
            self.bldg_x = d_conf.get('building').get('bldg_x')
            self.bldg_y = d_conf.get('building').get('bldg_y')

            # Booleans for plot
            self.plot_bldg = d_conf.get('plot_bools').get('bldg')
            self.plot_bldg_as_wedge = d_conf.get('plot_bools').get(
                'bldg_as_wedge')
            self.plot_cur_footprint = d_conf.get('plot_bools').get(
                'cur_footprint')
            self.plot_footprint_hist = d_conf.get('plot_bools').get(
                'footprint_hist')
            self.plot_sect_hist = d_conf.get('plot_bools').get('sect_hist')

        self.luf_ang_rad = np.radians(self.luf_ang)
        self.fix_ang_rad = np.radians(self.fix_ang)

        # Definining list of sectors
        self.phi_l = (np.linspace(0., 360., self.n_sect_2d+1)
                      - 360./(2*self.n_sect_2d))
        self.theta_l = np.linspace(0., 90., self.n_sect_3d+1)

        # FOV characteristics from camera
        self.hfov_v = 0.5*45.4
        self.hfov_h = 0.5*64.2

        self.cur_cam = 0

        # Slices' / Sectors' characteristics
        self.i_sect_2d = 360./self.n_sect_2d
        self.i_sect_3d = 90./self.n_sect_3d

        self.n_wireframe = 3

        self.n_sect_tot = self.n_sect_2d*self.n_sect_3d
        self.sect_passed = np.zeros(
            (self.n_sect_2d, self.n_sect_3d), dtype=bool)

        self.sect_passed_2d = np.zeros((self.n_cams, self.n_sect_2d, 5))
        self.sect_passed_3d = np.zeros(
            (self.n_cams, self.n_sect_2d, self.n_sect_3d, self.pics_per_slice,
             5))
        self.prev_2d_sect = -1*np.ones(self.n_cams)
        self.prev_3d_sect = -1*np.ones((self.n_cams, 2))

    def clear_sect_passed(self):
        """Clears history of which sectors have already been passed through
        """
        self.sect_passed = np.zeros(
            (self.n_sect_2d, self.n_sect_3d), dtype=bool)

        self.sect_passed_2d = np.zeros((self.n_cams, self.n_sect_2d, 5))
        self.sect_passed_3d = np.zeros(
            (self.n_cams, self.n_sect_2d, self.n_sect_3d, self.pics_per_slice,
             5))
        self.prev_2d_sect = -1*np.ones(self.n_cams)
        self.prev_3d_sect = -1*np.ones((self.n_cams, 2))


def main():
    """Does a 3D visualization of a point moving in spherical coordinates
    """
    d_conf = {}
    if len(sys.argv) > 1:
        with open(str(sys.argv[1]), 'r') as stream:
            d_conf = yaml.load(stream)
    myc = Crane(d_conf)

    # Initial config of camera
    r_s = myc.jib_l
    phi_init = (
        d_conf.get('crane').get('init_az_ang') if d_conf != {} else 135)
    theta_init = 90-myc.luf_ang

    s_w = 0.15
    s_h = 0.01
    s_l = 0.15
    s_u = 0.55
    s_uk = 0.05

    # Defining plot area and widgets
    fig = plt.figure(figsize=(6, 6))
    my_ax = fig.add_subplot(111, projection='3d', proj_type='ortho',
                            label='my ax')

    scol = "blue"
    salp = 0.2

    axradio = plt.axes([s_l, s_u, s_w, 10*s_h])
    rcams = RadioButtons(
        axradio, tuple(
            ['Cam '+str(x+1)+' selected' for x in range(myc.n_cams)]), 0,
        activecolor='green')

    s_u -= s_uk
    axphi = plt.axes([s_l, s_u, s_w, s_h])
    sphi = Slider(axphi, 'Azimuthal angle (phi)', myc.phi_l[0], myc.phi_l[-1],
                  valinit=phi_init, valstep=1, color=scol, alpha=salp)
    sphi.set_val(phi_init)  # Sometimes valinit does not work, forcing it

    s_u -= s_uk
    axtheta = plt.axes([s_l, s_u, s_w, s_h])
    sthet = Slider(axtheta, 'Polar angle (theta)', myc.theta_l[0],
                   myc.theta_l[-1], valinit=theta_init, valstep=1, color=scol,
                   alpha=salp)

    s_u -= s_uk
    axcamk = plt.axes([s_l, s_u, s_w, s_h])
    scamk = Slider(axcamk, 'Selected cam\'s relative distance', 0, 1,
                   valinit=myc.k_cams[0], valstep=0.5/myc.jib_l, color=scol,
                   alpha=salp)

    s_u -= s_uk
    axfix = plt.axes([s_l, s_u, s_w, s_h])
    sfix = Slider(axfix, 'Selected cam\'s bracket angle', 0, 80,
                  valinit=myc.fix_ang[0], valstep=10, color=scol, alpha=salp)

    s_u -= s_uk
    axmaxd = plt.axes([s_l, s_u, s_w, s_h])
    smaxd = Slider(axmaxd, 'Maximum distance from center', 0,
                   2*myc.jib_l//10*10, valinit=1.1*myc.jib_l//10*10, valstep=5,
                   color=scol, alpha=salp)

    s_u -= s_uk
    axbldgh = plt.axes([s_l, s_u, s_w, s_h])
    sbldh = Slider(axbldgh, 'Building height', 0, myc.tow_z+myc.tow_h,
                   valinit=myc.bldg_h, valstep=10, color=scol, alpha=salp)

    s_u -= s_uk
    axbldgw = plt.axes([s_l, s_u, s_w, s_h])
    sbldw = Slider(axbldgw, 'Building width', 0, 2*myc.jib_l,
                   valinit=myc.bldg_w, valstep=1, color=scol, alpha=salp)

    s_u -= s_uk
    axbldgd = plt.axes([s_l, s_u, s_w, s_h])
    sbldd = Slider(axbldgd, 'Building distance (radial)', 0, myc.jib_l,
                   valinit=myc.bldg_d, valstep=1, color=scol, alpha=salp)

    s_u -= s_uk
    axbldgx = plt.axes([s_l, s_u, s_w, s_h])
    sbldx = Slider(axbldgx, 'Building x-offset', np.floor(-myc.jib_l),
                   np.ceil(myc.jib_l), valinit=myc.bldg_x, valstep=1,
                   color=scol, alpha=salp)

    s_u -= s_uk
    axbldgy = plt.axes([s_l, s_u, s_w, s_h])
    sbldy = Slider(axbldgy, 'Building y-offset', np.floor(-myc.jib_l),
                   np.ceil(myc.jib_l), valinit=myc.bldg_y, valstep=1,
                   color=scol, alpha=salp)

    def update_cam_a(val=None):
        myc.cur_cam = int(rcams.value_selected.split()[1]) - 1
        scamk.set_val(myc.k_cams[myc.cur_cam])
        sfix.set_val(myc.fix_ang[myc.cur_cam])
        update()

    def update_cam_k(val=None):
        old_val = myc.k_cams[myc.cur_cam]
        myc.k_cams[myc.cur_cam] = scamk.val
        for i in range(myc.n_sect_2d):
            myc.sect_passed_2d[myc.cur_cam, i, :3] = (
                scamk.val * myc.sect_passed_2d[myc.cur_cam, i, :3]/old_val)
            for j in range(myc.n_sect_3d):
                for k in range(myc.pics_per_slice):
                    myc.sect_passed_3d[myc.cur_cam, i, j, k, :3] = (
                        scamk.val * (
                            myc.sect_passed_3d[
                                myc.cur_cam, i, j, k, :3]/old_val))
        update()

    def update_cam_f(val=None):
        myc.fix_ang[myc.cur_cam] = sfix.val
        myc.fix_ang_rad[myc.cur_cam] = np.radians(sfix.val)
        update()

    # For when a value is changed (either with widget or key presses)
    def update(val=None):
        """Update object instance and plot as per widget status

        Args:
            val (widget value): Attribute to access values
        """
        my_ax.clear()

        my_ax.text2D(
            0.1, 0.24, ('(' + str(round(myc.k_cams[myc.cur_cam]*myc.jib_l, 2))
                        + '[' + myc.units + '])'),
            ma="right", transform=my_ax.transAxes)

        myc.bldg_h = sbldh.val
        myc.bldg_d = sbldd.val
        myc.bldg_w = sbldw.val
        myc.bldg_x = sbldx.val
        myc.bldg_y = sbldy.val
        myc.cam_center_max_r = smaxd.val

        # Re-determine current slice / sector and its coordinates
        new_phi = sphi.val
        p_l, p_r, p_i = get_interval(myc.phi_l, new_phi)
        phi = np.linspace(p_l, p_r, myc.n_wireframe+1)

        new_theta = sthet.val
        t_l, t_r, t_i = get_interval(myc.theta_l, new_theta)
        theta = np.linspace(t_l, t_r, myc.n_wireframe+1)

        myc.luf_ang = 90 - new_theta
        myc.luf_ang_rad = np.radians(90 - new_theta)

        myc.sect_passed[p_i, t_i] = True
        if myc.plot_sect_hist:
            plot_sect_hist(my_ax, myc)

        msh_phi, msh_theta = np.meshgrid(phi, theta)
        msh_x, msh_y, msh_z = sph2car(r_s, msh_phi, msh_theta)
        jt_x, jt_y, jt_z = sph2car(r_s, new_phi, new_theta)

        msh_x += myc.tow_x
        msh_y += myc.tow_y
        msh_z += myc.tow_z + myc.tow_h

        jt_x += myc.tow_x
        jt_y += myc.tow_y
        jt_z += myc.tow_z + myc.tow_h

        msh_p = [msh_x, msh_y, msh_z]
        jt_p = [jt_x, jt_y, jt_z]

        cam_p = []
        for i in range(myc.n_cams):
            p_x, p_y, p_z = sph2car(myc.k_cams[i]*r_s, new_phi, new_theta)

            if (((p_x**2+p_y**2) < myc.cam_center_max_r**2
                 and not p_i == myc.prev_2d_sect[i])):
                myc.sect_passed_2d[i, p_i] = [p_x, p_y, p_z, new_phi,
                                              myc.luf_ang_rad]
                myc.prev_2d_sect[i] = p_i

            if (((p_x**2+p_y**2) < myc.cam_center_max_r**2
                 and not set([p_i, t_i]) == set(myc.prev_3d_sect[i]))):
                myc.prev_3d_sect[i] = [p_i, t_i]
                ctr = -1
                for j in range(myc.pics_per_slice):
                    if not myc.sect_passed_3d[i, p_i, t_i, j].any():
                        ctr = j
                        break
                if ctr != -1:
                    myc.sect_passed_3d[i, p_i, t_i, ctr] = [p_x, p_y, p_z,
                                                            new_phi,
                                                            myc.luf_ang_rad]

            p_x += myc.tow_x
            p_y += myc.tow_y
            p_z += myc.tow_z + myc.tow_h

            cam_p.append([p_x, p_y, p_z])

        # Redraw everything
        plot_all(my_ax, myc, msh_p, cam_p, jt_p, new_phi)
        set_axes_equal(my_ax, myc)
        fig.canvas.draw_idle()

    # Initial update now that everything is set up
    update()

    rcams.on_clicked(update_cam_a)
    sphi.on_changed(update)
    sthet.on_changed(update)
    scamk.on_changed(update_cam_k)
    sfix.on_changed(update_cam_f)
    sbldh.on_changed(update)
    sbldd.on_changed(update)
    sbldw.on_changed(update)
    sbldx.on_changed(update)
    sbldy.on_changed(update)
    smaxd.on_changed(update)

    my_ax.view_init(45, 0)
    set_axes_equal(my_ax, myc)

    # Handles keyboard events
    def press(event):
        """Handles keyboard events

        Args:
            event (key_press_event): Key that was pressed
        """
        if event.key == 'right':  # Azimuth increase
            k_mul = 5
            if sphi.val + sphi.valstep*k_mul > sphi.valmax:
                sphi.set_val(
                    sphi.valmin + sphi.val + sphi.valstep*k_mul - sphi.valmax)
            else:
                check_slider_min_max(sphi, mul=k_mul)
        elif event.key == 'left':  # Azimuth decrease
            k_mul = 5
            if sphi.val - sphi.valstep*k_mul < sphi.valmin:
                sphi.set_val(
                    sphi.valmax + sphi.val - sphi.valstep*k_mul - sphi.valmin)
            else:
                check_slider_min_max(sphi, '-', k_mul)

        elif event.key == 'up':  # Polar angle decrease / Luffing increase
            check_slider_min_max(sthet, '-', 5)
        elif event.key == 'down':  # Polar angle increase / Luffing decrease
            check_slider_min_max(sthet, mul=5)

        elif event.key == 'Y':  # Camera relative position decrease
            check_slider_min_max(scamk, '-')
        elif event.key == 'X':  # Camera relative position increase
            check_slider_min_max(scamk)

        elif event.key == 'y':  # Fixed camera bracket angle decrease
            check_slider_min_max(sfix, '-')
        elif event.key == 'x':  # Fixed camera bracket angle increase
            check_slider_min_max(sfix)

        elif event.key == 'v':  # Max distance tower-cam footprint decrease
            check_slider_min_max(smaxd, '-')
        elif event.key == 'b':  # Max distance tower-cam footprint increase
            check_slider_min_max(smaxd)

        elif event.key == 'n':  # Building height decrease
            check_slider_min_max(sbldh, '-')
        elif event.key == 'm':  # Building height increase
            check_slider_min_max(sbldh)

        elif event.key == 'ctrl+n':  # Building distance decrease
            check_slider_min_max(sbldd, '-')
        elif event.key == 'ctrl+m':  # Building distance increase
            check_slider_min_max(sbldd)

        elif event.key == 'alt+n':  # Building width decrease
            check_slider_min_max(sbldw, '-')
        elif event.key == 'alt+m':  # Building width increase
            check_slider_min_max(sbldw)

        elif event.key == 'alt+up':  # Building x-offset decrease
            check_slider_min_max(sbldx, '-')
        elif event.key == 'alt+down':  # Building x-offset increase
            check_slider_min_max(sbldx)

        elif event.key == 'alt+left':  # Building y-offset decrease
            check_slider_min_max(sbldy, '-')
        elif event.key == 'alt+right':  # Building y-offset increase
            check_slider_min_max(sbldy)

        elif event.key == 'c':  # Clears all sectors' history
            myc.clear_sect_passed()
            update()

        elif event.key == 'h':  # Hides / Shows current footprint
            hide_show_plot(myc, "myc.plot_cur_footprint")
        elif event.key == 'H':  # Hides / Shows sectors' history
            hide_show_plot(myc, "myc.plot_sect_hist")
        elif event.key == 'j':  # Hides / Shows footprints' history
            hide_show_plot(myc, "myc.plot_footprint_hist")
        elif event.key == 'ctrl+h':  # Hides / Shows building
            hide_show_plot(myc, "myc.plot_bldg")
        elif event.key == 'alt+h':  # Toggles between wedge and polygon bldg
            hide_show_plot(myc, "myc.plot_bldg_as_wedge")

        elif event.key == 'T':  # Go to top view
            my_ax.view_init(90, 0)
            update()
        elif event.key == 'F':  # Go to front view
            my_ax.view_init(0, 0)
            update()
        elif event.key == 'O':  # Go to orthogonal view
            my_ax.view_init(45, 45)
            update()

        # Change current cam (for sliders) with keyboard numbers 1 to 9
        elif len(event.key) == 1:
            if ord(event.key) > 48 and ord(event.key) < 58:
                myc.cur_cam = myc.n_cams-1 if int(
                    event.key) % myc.n_cams == 0 else(
                        int(event.key) % myc.n_cams - 1)

                rcams.set_active(myc.cur_cam)
                update_cam_f()
                update_cam_k()
        else:
            if ((not event.key == 'ctrl'
                 and not event.key == 'alt'
                 and not event.key == 'shift')):
                print(event.key)

    def check_slider_min_max(sli, sign='+', mul=1):
        """Updates slider values while checking if min / max has been reached

        Args:
            sli (matplotlib widget): Slider to be modified
            sign (str, optional): Increase / decrease logic. Defaults to '+'.
            mul (int, optional): Valstep multiplication factor. Defaults to 1.
        """
        if sign == '+':
            n_v = (sli.valmax if sli.val+sli.valstep*mul > sli.valmax else (
                sli.val+sli.valstep*mul))
        else:
            n_v = (sli.valmin if sli.val-sli.valstep*mul < sli.valmin else (
                sli.val-sli.valstep*mul))
        sli.set_val(n_v)

    def hide_show_plot(myc, plot_bool):
        """Executes a boolean change in a parameter of a class object

        Args:
            myc (class object): Object whose argument must be changed
            plot_bool (string): Argument
        """
        exec(plot_bool + " = not " + plot_bool)
        update()

    fig.canvas.mpl_connect('key_press_event', press)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


def sph2car(r_s, phi, theta):
    """Converts spherical coordinates into cartesian coordinates

    Args:
        r_s (float): Radius
        phi (float): Azimuthal angle in [deg]
        theta (float): Polar angle in [deg]

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
        lst (np.array): List of intervals
        val (float): Value whose interval must be found

    Returns:
        2 floats and 1 int: The interval boundaries and the left index
    """
    for i in range(len(lst)-1):
        if lst[i] <= val <= lst[i+1]:
            # print("{} is between {} and {}".format(val, lst[i], lst[i+1]))
            return lst[i], lst[i+1], i
    print("Error")
    sys.exit()


def plot_all(my_ax, myc, msh_p, cam_p, jt_p, cur_phi):
    """Plots a 3D surface as well as the line from center to current point

    Args:
        my_ax (plt figure 3D subplot): Where results are plotted
        myc (Crane): Instance containing all physical parameters and history
        msh_p (np.array): Current mesh coordinates
        cam_p (float): Current point's coordinates
        jt_p (float): Jib tip point's coordinates
        cur_phi (float): Current azimuthal angle [deg]
    """
    print("Re-plotting with latest values")
    # Plots current sector as a wireframe
    if myc.plot_sect_hist:
        my_ax.plot_wireframe(*msh_p, colors="green")

    ext = 5
    my_ax.plot(
        [myc.tow_x+ext, myc.tow_x-ext, myc.tow_x, myc.tow_x-ext, myc.tow_x+ext,
         myc.tow_x, myc.tow_x, jt_p[0]],
        [myc.tow_y-ext, myc.tow_y+ext, myc.tow_y, myc.tow_y-ext, myc.tow_y+ext,
         myc.tow_y, myc.tow_y, jt_p[1]],
        [myc.tow_z, myc.tow_z, myc.tow_z, myc.tow_z, myc.tow_z, myc.tow_z,
         myc.tow_z+myc.tow_h, jt_p[2]],
        c="orange")

    if myc.plot_bldg and not myc.bldg_h < 1 and not myc.bldg_w < 1:
        plot_bldg(my_ax, myc)

    ftpt_lim = Circle((myc.tow_x, myc.tow_y), myc.cam_center_max_r, fill=False,
                      alpha=0.1)
    my_ax.add_patch(ftpt_lim)
    art3d.pathpatch_2d_to_3d(ftpt_lim, z=0, zdir="z")

    # Plot cameras
    tot_area = 0
    for i, item in enumerate(cam_p):
        # Plot current footprint
        sca = False
        if myc.plot_cur_footprint:
            sca, area = plot_footprint(my_ax, myc, item, cur_phi,
                                       myc.fix_ang_rad[i], i, False, colr="g",
                                       alp=0.3, sc_size=20)
            tot_area += area

        my_ax.scatter(*item, s=50, c="green" if sca else "red")
    if myc.plot_cur_footprint:
        print("Total current footprint area is {:.1f}[{}^2]".format(
            tot_area, myc.units))
        print()

    # Plot footprint history
    if myc.plot_footprint_hist:
        plot_footprint_hist(my_ax, myc)

    set_axes_equal(my_ax, myc)
    print()


def plot_footprint(my_ax, myc, cam_p, cur_phi, fix_ang_rad, cam_num, bool_hist,
                   luf_ang_rad=None, draw_trace=True, colr="black", alp=0.05,
                   sc_size=10):
    """Plots the footprint of a single camera

    Args:
        my_ax (plt figure 3D subplot): Where results are plotted
        myc (Crane): Instance containing all physical parameters and history
        cam_p (float): Current point's coordinates
        cur_phi (float): Current azimuthal angle [deg]
        fix_ang_rad (float): Bracket angle [rad]
        luf_ang_rad (float, optional): Luffing angle [rad]. Defaults to None.
        draw_trace (bool, optional): Draws pyramid edges. Defaults to True.
        colr (str, optional): Footprint color. Defaults to "black".
        alp (float, optional): Footprint alpha value. Defaults to 0.1.
        sc_size (int, optional): Scatter point size. Defaults to 10.

    Returns:
        bool: Bool to say if camera is active (i.e. should take a pic or not)
    """
    if luf_ang_rad is None:
        luf_ang_rad = myc.luf_ang_rad

    delta_h = cam_p[2] - myc.bldg_h  # Difference between camera z and gnd
    # delta_r = delta_h*np.tan(np.radians(myc.hfov_v))  # Radial difference
    delta_t = delta_h*np.tan(np.radians(myc.hfov_h))  # Tangential difference

    p_r_old = np.sqrt(cam_p[0]**2+cam_p[1]**2)  # Radial position

    # Radial positions at ground considering luffing angle and bracket angle
    p_r_new = p_r_old - delta_h*np.tan(fix_ang_rad-luf_ang_rad)

    area = 0
    sca = np.abs(p_r_new) < myc.cam_center_max_r
    if sca:
        p_r_new_in = p_r_old - delta_h*np.tan(
            fix_ang_rad+np.radians(myc.hfov_v)-luf_ang_rad)
        p_r_new_out = p_r_old - delta_h*np.tan(
            fix_ang_rad-np.radians(myc.hfov_v)-luf_ang_rad)

        # Converting them to cartesian coordinates
        cur_phi_rad = np.radians(cur_phi)
        p_r_new_x = p_r_new*np.cos(cur_phi_rad)
        p_r_new_y = p_r_new*np.sin(cur_phi_rad)
        p_r_new_in_x = p_r_new_in*np.cos(cur_phi_rad)
        p_r_new_in_y = p_r_new_in*np.sin(cur_phi_rad)
        p_r_new_out_x = p_r_new_out*np.cos(cur_phi_rad)
        p_r_new_out_y = p_r_new_out*np.sin(cur_phi_rad)

        # Deducing edge corners' positions
        p_t_x_ll = p_r_new_in_x - delta_t*np.sin(cur_phi_rad)
        p_t_y_ll = p_r_new_in_y + delta_t*np.cos(cur_phi_rad)

        p_t_x_ul = p_r_new_out_x - delta_t*np.sin(cur_phi_rad)
        p_t_y_ul = p_r_new_out_y + delta_t*np.cos(cur_phi_rad)

        p_t_x_lr = p_r_new_in_x + delta_t*np.sin(cur_phi_rad)
        p_t_y_lr = p_r_new_in_y - delta_t*np.cos(cur_phi_rad)

        p_t_x_ur = p_r_new_out_x + delta_t*np.sin(cur_phi_rad)
        p_t_y_ur = p_r_new_out_y - delta_t*np.cos(cur_phi_rad)

        pxs = [p_r_new_x, p_t_x_ll, p_t_x_ul, p_t_x_ur, p_t_x_lr]
        pys = [p_r_new_y, p_t_y_ll, p_t_y_ul, p_t_y_ur, p_t_y_lr]
        pzs = 5*[myc.bldg_h]

        # Draw FOV's pyramid-like shape from camera to footprint
        if draw_trace:
            my_ax.plot([cam_p[0], p_t_x_ll],
                       [cam_p[1], p_t_y_ll],
                       [cam_p[2], myc.bldg_h],
                       color=colr, alpha=alp)
            my_ax.plot([cam_p[0], p_t_x_lr],
                       [cam_p[1], p_t_y_lr],
                       [cam_p[2], myc.bldg_h],
                       color=colr, alpha=alp)
            my_ax.plot([cam_p[0], p_t_x_ul],
                       [cam_p[1], p_t_y_ul],
                       [cam_p[2], myc.bldg_h],
                       color=colr, alpha=alp)
            my_ax.plot([cam_p[0], p_t_x_ur],
                       [cam_p[1], p_t_y_ur],
                       [cam_p[2], myc.bldg_h],
                       color=colr, alpha=alp)

        # Draw edges + center point then cover it with a patch
        my_ax.scatter(pxs, pys, pzs, s=sc_size, c=colr, alpha=alp)
        my_ax.plot_trisurf(pxs, pys, pzs, color=colr, alpha=alp, shade=False)

        area = get_polygon_area(np.array(pxs[1:]), np.array(pys[1:]), 4)

        if bool_hist:
            print("Area of historical footprint of cam {} at [{:+6.1f}, "
                  "{:+6.1f}][deg], i.e. sector [{:2d}, {:2d}]: "
                  "{:+9.1f}[{}^2]".format(
                      cam_num, cur_phi, np.degrees(luf_ang_rad),
                      get_interval(myc.phi_l, cur_phi)[2],
                      get_interval(myc.theta_l, 90-np.degrees(luf_ang_rad))[2],
                      area, myc.units))
        else:
            print("Area of current footprint for cam {}: {:.1f}[{}^2]".format(
                cam_num, area, myc.units))

    return sca, area


def get_polygon_area(x_s, y_s, n_points):
    """Computes the area of a polygon made of n_points

    Args:
        x_s (list): Array of polygon's points' x coordinates
        y_s (list): Array of polygon's points' y coordinates
        n_points (int): Number of points the polygon has

    Returns:
        float: Polygon are
    """
    area = 0.0
    idx = n_points-1

    for i in range(n_points):
        area += (x_s[idx]+x_s[i])*(y_s[idx]-y_s[i])
        idx = i

    return np.abs(0.5*area)


def plot_bldg(my_ax, myc):
    """Plots the building next to the crane

    Args:
        my_ax (plt figure subplot): Where sectors should be plotted
        myc (Crane): Instance containing all physical parameters and history
    """
    if myc.plot_bldg_as_wedge:
        rad = 0.5*np.linspace(myc.bldg_d, myc.bldg_d + myc.bldg_w, 2)
        phi = np.linspace(0, 2*np.pi, 50)
        alt = np.linspace(0, myc.bldg_h, 2)

        r_msh, p_msh = np.meshgrid(rad, phi)
        alt_zero = np.zeros(r_msh.shape)
        alt_high = myc.bldg_h*np.ones(r_msh.shape)

        p_wall, z_wall = np.meshgrid(phi, alt)
        rad_in = 0.5*myc.bldg_d*np.ones(p_wall.shape)
        rad_out = 0.5*(myc.bldg_d+myc.bldg_w)*np.ones(p_wall.shape)

        x_flat = myc.tow_x + r_msh*np.cos(p_msh) + myc.bldg_x
        y_flat = myc.tow_y + r_msh*np.sin(p_msh) + myc.bldg_y

        x_in = myc.tow_x + rad_in*np.cos(p_wall) + myc.bldg_x
        y_in = myc.tow_y + rad_in*np.sin(p_wall) + myc.bldg_y

        x_out = myc.tow_x + rad_out*np.cos(p_wall) + myc.bldg_x
        y_out = myc.tow_y + rad_out*np.sin(p_wall) + myc.bldg_y

        my_ax.plot_surface(x_flat, y_flat, alt_zero, color="b", alpha=0.1,
                           rcount=1, ccount=1, lw=0, antialiased=False)
        my_ax.plot_surface(x_flat, y_flat, alt_high, color="b", alpha=0.1,
                           rcount=1, ccount=1, lw=0, antialiased=False)
        my_ax.plot_surface(x_in, y_in, z_wall, color="b", alpha=0.1,
                           rcount=1, ccount=10, lw=0, antialiased=False)
        my_ax.plot_surface(x_out, y_out, z_wall, color="b", alpha=0.1,
                           rcount=1, ccount=10, lw=0, antialiased=False)

    else:
        x_1 = myc.tow_x - 0.5*myc.bldg_w + myc.bldg_x
        x_2 = myc.tow_x + 0.5*myc.bldg_w + myc.bldg_x
        y_1 = myc.tow_y - 0.5*myc.bldg_w + myc.bldg_y
        y_2 = myc.tow_y + 0.5*myc.bldg_w + myc.bldg_y

        z_1 = 0
        z_2 = myc.bldg_h

        x_a = [x_1, x_2]
        y_a = [y_1, y_2]
        z_a = [z_1, z_2]

        x_1, y_1 = np.meshgrid(x_a, y_a)
        x_2, z_2 = np.meshgrid(x_a, z_a)
        y_3, z_3 = np.meshgrid(y_a, z_a)

        s_1 = [x_1, y_1, np.ones(x_1.shape)*z_a[0]]
        s_2 = [x_1, y_1, np.ones(x_1.shape)*z_a[-1]]
        s_3 = [x_2, np.ones(x_2.shape)*y_a[0], z_2]
        s_4 = [x_2, np.ones(x_2.shape)*y_a[-1], z_2]
        s_5 = [np.ones(y_3.shape)*x_a[0], y_3, z_3]
        s_6 = [np.ones(y_3.shape)*x_a[-1], y_3, z_3]
        s_s = [s_1, s_2, s_3, s_4, s_5, s_6]

        # Plot the surface.
        for item in s_s:
            my_ax.plot_surface(*item, color="b", alpha=0.1, rcount=1, ccount=1,
                               lw=0, antialiased=False)


def plot_sect_hist(my_ax, myc):
    """Plots all sectors that have already been passed

    Args:
        my_ax (plt figure subplot): Where sectors should be plotted
        myc (Crane): Instance containing all physical parameters and history
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
        my_ax (plt figure subplot): Where sectors should be plotted
        myc (Crane): Instance containing all physical parameters and history
    """
    tot_area = 0
    if myc.hist_2d_only:
        for i in range(myc.n_cams):
            for j in range(myc.sect_passed_2d.shape[1]):
                if myc.sect_passed_2d[i, j].any():
                    cam_pft = [myc.sect_passed_2d[i, j, 0] + myc.tow_x,
                               myc.sect_passed_2d[i, j, 1] + myc.tow_y,
                               (myc.sect_passed_2d[i, j, 2] + myc.tow_z
                                + myc.tow_h)]

                    plot_footprint(my_ax,
                                   myc,
                                   cam_pft,
                                   myc.sect_passed_2d[i, j, 3],
                                   myc.fix_ang_rad[i],
                                   i,
                                   True,
                                   myc.sect_passed_2d[i, j, 4],
                                   False)
    else:
        for i in range(myc.n_cams):
            for j in range(myc.n_sect_2d):
                for k in range(myc.n_sect_3d):
                    for l in range(myc.pics_per_slice):
                        if myc.sect_passed_3d[i, j, k, l].any():
                            cam_pft = [
                                myc.sect_passed_3d[i, j, k, l, 0] + myc.tow_x,
                                myc.sect_passed_3d[i, j, k, l, 1] + myc.tow_y,
                                (myc.sect_passed_3d[i, j, k, l, 2] + myc.tow_z
                                 + myc.tow_h)]

                            area = plot_footprint(
                                my_ax,
                                myc,
                                cam_pft,
                                myc.sect_passed_3d[i, j, k, l, 3],
                                myc.fix_ang_rad[i],
                                i,
                                True,
                                myc.sect_passed_3d[i, j, k, l, 4],
                                False)[1]
                            tot_area += area
            print()
        print("Total historical footprint area is {:+.1f}[{}^2]".format(
            tot_area, myc.units))
    print()


def set_axes_equal(my_ax, myc):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    my_ax.set_aspect('equal') and my_ax.axis('equal') not working for 3D.

    Args
        my_ax (plt figure subplot): Where sectors should be plotted
        myc (Crane): Instance containing all physical parameters and history
    """
    my_ax.set_xlabel('X axis')
    my_ax.set_ylabel('Y axis')
    my_ax.set_zlabel('Z axis')
    b_b = 1.5*myc.jib_l
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

    my_ax.text2D(1.32, 0.65, "Keyboard commands", fontweight="bold",
                 ma="right", transform=my_ax.transAxes)
    my_ax.text2D(1, 0.10,
                 """left/right: modify azimuthal angle
                 up/down: modify polar angle
                 Y/X: modify selected cam\'s relative position
                 y/x: modify selected cam\'s fixed bracket angle
                 v/b: modify max tower-cam footprint distance
                 n/m: modify building height
                 ctrl+n/ctrl+m: modify building\'s distance from tower
                 alt+n/alt+m: modify building\'s width
                 h: toggle current footprint
                 H: toggle sector history
                 j: toggle footprint history
                 ctrl+h: toggle building
                 alt+h: toggle building shape
                 ctrl+alt+h: toggle building center location""",
                 linespacing=3,
                 ma="right",
                 transform=my_ax.transAxes)


if __name__ == '__main__':
    main()
