"""fixed_luffing.py: Simple 2D cov. analysis tool for a single luffing crane
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon, Rectangle, Wedge
from matplotlib.widgets import Slider, RadioButtons

NCAMS = 4
TOW_X = 0
TOW_Y = 0
TOW_H = 60
JIB_L = 68
CAM_HFOV = np.radians(0.5*45.4)
USING_BRACKET = False
BRACKET_ANGLES = [-61.5, -59.0, -56.5, -53.5, -49.5, -45.0, -40.0, -34.0,
                  -27.0, -18.5, -9.5, 0.0, 9.5, 18.5, 27.0, 34.0, 40.0, 45.0,
                  49.5, 53.5, 56.5, 59.0, 61.5]


def plot_scene_fv(cur_ax, luf_ang, cams_kr, cams_ang, bldg_h, bldg_d, bldg_w,
                  ax_title):
    """Plots the desired scene in front view

    Args:
        cur_ax ([type]): Subplot to be plotted
        luf_ang ([type]): Crane luffing angle [rad]
        cams_kr ([type]): Array of cameras' locs (rel. to crane length) [-]
        cams_ang ([type]): Array of cameras' angles (when jib is hor.) [rad]
        bldg_h ([type]): Bldg height [m]
        bldg_d ([type]): Bldg distance from crane tower [m]
        bldg_w ([type]): Bldg width [m]
        ax_title ([type]): Title of the subplot
    """
    cur_ax.plot([0, 0, 0 + JIB_L*np.cos(luf_ang)],
                [0, TOW_H, TOW_H + JIB_L*np.sin(luf_ang)],
                color='orange')

    for i in range(NCAMS):
        plot_cam_fv(cur_ax, cams_kr[i], cams_ang[i], luf_ang)

    cur_ax.add_patch(Rectangle((bldg_d, 0.0), bldg_w, bldg_h, fc='black',
                               ec=None, alpha=0.6, zorder=-1))

    cur_ax.title.set_text(ax_title)
    cur_ax.set_xlim([-200, 200])
    cur_ax.set_ylim([-10, 200])
    cur_ax.grid(True)
    print("\n")


def plot_cam_fv(cur_ax, cam_kr, cam_ang, luf_ang):
    """Plots a single camera in front view

    Args:
        cur_ax ([type]): Subplot to be plotted
        cam_kr ([type]): Camera loc (rel. to crane length) [-]
        cam_ang ([type]): Camera angle (when jib is hor.) [rad]
        luf_ang ([type]): Crane luffing angle [rad]
    """
    cur_ax.scatter([0 + cam_kr*JIB_L*np.cos(luf_ang),
                    0 + cam_kr*JIB_L*np.cos(luf_ang),
                    0 + cam_kr*JIB_L*np.cos(luf_ang)],
                   [TOW_H + cam_kr*JIB_L*np.sin(luf_ang),
                    TOW_H + cam_kr*JIB_L*np.sin(luf_ang),
                    TOW_H + cam_kr*JIB_L*np.sin(luf_ang)],
                   color='blue')

    fov_left = luf_ang+cam_ang-CAM_HFOV
    fov_mid = luf_ang+cam_ang
    fov_right = luf_ang+cam_ang+CAM_HFOV

    fov_left_deg = np.degrees(fov_left)
    fov_mid_deg = np.degrees(fov_mid)
    fov_right_deg = np.degrees(fov_right)

    luf_ang_deg = np.degrees(luf_ang)

    print("Cam @ {:4.2f} with luf_ang {:5.1f}: left FOV is {:5.1f}[°], mid "
          "FOV is {:5.1f} and right FOV angle is {:5.1f}[°]".format(
              cam_kr,
              luf_ang_deg,
              fov_left_deg,
              fov_mid_deg,
              fov_right_deg))

    fov_limit = np.radians(90.0)
    dist_far_near = (TOW_H + cam_kr*JIB_L*np.sin(luf_ang))*(
        np.tan(fov_right)-np.tan(fov_left))

    if fov_left >= fov_limit:
        print("\tCam @ {:4.2f} with luf_ang {:5.1f}: FOV not pointing to the"
              " ground, {:7.2f}!".format(cam_kr, luf_ang_deg, dist_far_near))

    elif fov_mid >= fov_limit:
        print("\tCam @ {:4.2f} with luf_ang {:5.1f}: FOV mid-line pointing "
              "above the horizon, {:7.2f}!".format(
                  cam_kr, luf_ang_deg, dist_far_near))

    elif fov_right >= fov_limit:
        print("\tCam @ {:4.2f} with luf_ang {:5.1f}: FOV is surely pretty "
              "far from the crane, {:7.2f}!".format(
                  cam_kr, luf_ang_deg, dist_far_near))

    else:
        x_limit = 250.0

        cam_r_rel = cam_kr*JIB_L

        cam_x_rel = cam_r_rel*np.cos(luf_ang)
        cam_x_abs = TOW_X + cam_x_rel

        cam_y_rel = TOW_H + cam_r_rel*np.sin(luf_ang)
        cam_y_abs = TOW_Y + cam_y_rel

        cam_x_left_rel = cam_y_rel*np.tan(fov_left)
        cam_x_right_rel = cam_y_rel*np.tan(fov_right)

        cam_x_left_abs = cam_x_abs + cam_x_left_rel
        cam_x_right_abs = cam_x_abs + cam_x_right_rel

        if cam_x_left_abs >= x_limit:
            print("\t\tCam @ {:4.2f} with luf_ang {:5.1f}: Left FOV on the "
                  "ground too far from the crane, {:7.2f}!".format(
                      cam_kr, luf_ang_deg, cam_x_left_abs))
        elif cam_x_right_abs >= x_limit:
            print("\t\tCam @ {:4.2f} with luf_ang {:5.1f}: Right FOV on the "
                  "ground too far from the crane, {:7.2f}!".format(
                      cam_kr, luf_ang_deg, cam_x_right_abs))
        else:
            cur_ax.add_patch(Polygon(
                np.array([[cam_x_abs, cam_y_abs],
                          [cam_x_left_abs, 0],
                          [cam_x_right_abs, 0]]),
                fc='blue', ec=None, alpha=0.3))


def plot_scene_tv(cur_ax, luf_ang, cams_kr, cams_ang, bldg_h, bldg_d, bldg_w,
                  ax_title):
    """Plots the desired scene in front view

    Args:
        cur_ax ([type]): Subplot to be plotted
        luf_ang ([type]): Crane luffing angle [rad]
        cams_kr ([type]): Array of cameras' locs (rel. to crane length) [-]
        cams_ang ([type]): Array of cameras' angles (when jib is hor.) [rad]
        bldg_h ([type]): Bldg height [m]
        bldg_d ([type]): Bldg distance from crane tower [m]
        bldg_w ([type]): Bldg width [m]
        ax_title ([type]): Title of the subplot
    """
    cur_ax.plot([TOW_X, TOW_X + JIB_L*np.cos(luf_ang)],
                [TOW_Y, TOW_Y],
                color='orange')

    for i in range(NCAMS):
        plot_cam_tv(cur_ax, cams_kr[i], cams_ang[i], luf_ang)

    cur_ax.add_patch(
        Wedge((TOW_X, TOW_Y), TOW_X+bldg_d+bldg_w, 0, 360, width=bldg_w,
              fc='black', ec=None, alpha=0.6, zorder=-1))

    cur_ax.title.set_text(ax_title)
    cur_ax.set_xlim([-200, 200])
    cur_ax.set_ylim([-200, 200])
    cur_ax.grid(True)
    print("\n")


def plot_cam_tv(cur_ax, cam_kr, cam_ang, luf_ang):
    """Plots a single camera in front view

    Args:
        cur_ax ([type]): Subplot to be plotted
        cam_kr ([type]): Camera loc (rel. to crane length) [-]
        cam_ang ([type]): Camera angle (when jib is hor.) [rad]
        luf_ang ([type]): Crane luffing angle [rad]
    """
    cur_ax.scatter([TOW_X + cam_kr*JIB_L*np.cos(luf_ang),
                    TOW_X + cam_kr*JIB_L*np.cos(luf_ang),
                    TOW_X + cam_kr*JIB_L*np.cos(luf_ang)],
                   [0,
                    0,
                    0],
                   color='blue')

    fov_left = luf_ang+cam_ang-CAM_HFOV
    fov_mid = luf_ang+cam_ang
    fov_right = luf_ang+cam_ang+CAM_HFOV

    fov_left_deg = np.degrees(fov_left)
    fov_mid_deg = np.degrees(fov_mid)
    fov_right_deg = np.degrees(fov_right)

    luf_ang_deg = np.degrees(luf_ang)

    print("Cam @ {:4.2f} with luf_ang {:5.1f}: left FOV is {:5.1f}[°], mid "
          "FOV is {:5.1f} and right FOV angle is {:5.1f}[°]".format(
              cam_kr,
              luf_ang_deg,
              fov_left_deg,
              fov_mid_deg,
              fov_right_deg))

    fov_limit = np.radians(90.0)
    dist_far_near = (TOW_H + cam_kr*JIB_L*np.sin(luf_ang))*(
        np.tan(fov_right)-np.tan(fov_left))

    if fov_left >= fov_limit:
        print("\tCam @ {:4.2f} with luf_ang {:5.1f}: FOV not pointing to the"
              " ground, {:7.2f}!".format(cam_kr, luf_ang_deg, dist_far_near))

    elif fov_mid >= fov_limit:
        print("\tCam @ {:4.2f} with luf_ang {:5.1f}: FOV mid-line pointing "
              "above the horizon, {:7.2f}!".format(
                  cam_kr, luf_ang_deg, dist_far_near))

    elif fov_right >= fov_limit:
        print("\tCam @ {:4.2f} with luf_ang {:5.1f}: FOV is surely pretty "
              "far from the crane, {:7.2f}!".format(
                  cam_kr, luf_ang_deg, dist_far_near))

    else:
        x_limit = 250.0

        cam_r_rel = cam_kr*JIB_L

        cam_x_rel = cam_r_rel*np.cos(luf_ang)
        cam_x_abs = TOW_X + cam_x_rel

        cam_y_rel = TOW_H + cam_r_rel*np.sin(luf_ang)
        cam_y_abs = TOW_Y + cam_y_rel

        cam_x_left_rel = cam_y_rel*np.tan(fov_left)
        cam_x_right_rel = cam_y_rel*np.tan(fov_right)

        cam_x_left_abs = cam_x_abs + cam_x_left_rel
        cam_x_right_abs = cam_x_abs + cam_x_right_rel

        if cam_x_left_abs >= x_limit:
            print("\t\tCam @ {:4.2f} with luf_ang {:5.1f}: Left FOV on the "
                  "ground too far from the crane, {:7.2f}!".format(
                      cam_kr, luf_ang_deg, cam_x_left_abs))
        elif cam_x_right_abs >= x_limit:
            print("\t\tCam @ {:4.2f} with luf_ang {:5.1f}: Right FOV on the "
                  "ground too far from the crane, {:7.2f}!".format(
                      cam_kr, luf_ang_deg, cam_x_right_abs))
        else:
            cur_ax.add_patch(
                Wedge((TOW_X, TOW_Y), cam_x_right_abs, 0, 360,
                      width=dist_far_near, fc='blue', ec=None, alpha=0.3))


def main():
    """Main function that calls all the rest
    """
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    plt.setp(axs.flat, aspect=1.0)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

    axs[1, 1].remove()

    luf_min = np.radians(35)
    luf_max = np.radians(85)

    vert_pos = 0.45
    vert_low = 0.05
    hori_pos = 0.60
    axluf = plt.axes([hori_pos, vert_pos, 0.2, 0.01])
    axck = []
    axca = []
    jump = (vert_pos-vert_low)/(3*NCAMS+6)
    vert_pos -= jump
    for i in range(NCAMS):
        vert_pos -= jump
        axck.append(plt.axes([hori_pos, vert_pos, 0.2, 0.01]))
        vert_pos -= jump
        axca.append(plt.axes([hori_pos, vert_pos, 0.2, 0.01]))
        vert_pos -= jump

    vert_pos -= jump
    axbldgh = plt.axes([hori_pos, vert_pos, 0.2, 0.01])
    vert_pos -= jump
    axbldgd = plt.axes([hori_pos, vert_pos, 0.2, 0.01])
    vert_pos -= jump
    axbldgw = plt.axes([hori_pos, vert_pos, 0.2, 0.01])

    vert_pos -= 4*jump
    axradfvtv = plt.axes([hori_pos, vert_pos, 0.2, 0.06])

    ck_min = 0.0
    ck_max = 1.0

    cams_kr = [x/(NCAMS+1) for x in range(1, NCAMS+1)]

    if USING_BRACKET:
        ca_min = 0
        ca_max = 22
    else:
        ca_min = -80.0
        ca_max = 50.0

    pa_min = -45
    pa_max = 0
    cams_ang = np.radians(
        [pa_max-x/(NCAMS-1) for x in (pa_max-pa_min)*np.array(range(NCAMS))])

    axs_titles = ['At minimum luffing angle',
                  'At maximum luffing angle',
                  'At luffing angle from slider']

    sluf = Slider(axluf, 'Luffing angle ', 35, 85,
                  valinit=35, valstep=1, color="blue")
    sck = []
    sca = []
    for i in range(NCAMS):
        str_pos = "Cam" + str(i+1) + " rel. loc."
        sck.append(Slider(
            axck[i], str_pos, ck_min, ck_max,
            valinit=cams_kr[i], valstep=0.05, color="blue"))
        str_ang = "Cam" + str(i+1) + " angle   "
        if USING_BRACKET:
            sca.append(Slider(
                axca[i], str_ang, ca_min, ca_max,
                valinit=11, valstep=1, color="blue"))
        else:
            sca.append(Slider(
                axca[i], str_ang, ca_min, ca_max,
                valinit=np.degrees(cams_ang[i]), valstep=10.0, color="blue"))

    sbldgh = Slider(axbldgh, "Bldg height   ", 0, TOW_H-10,
                    valinit=0.0, valstep=1.0, color='blue')
    sbldgd = Slider(axbldgd, "Bldg distance ", 0, JIB_L+10,
                    valinit=10.0, valstep=1.0, color='blue')
    sbldgw = Slider(axbldgw, "Bldg width    ", 0, JIB_L+10,
                    valinit=50.0, valstep=1.0, color='blue')

    radfvtv = RadioButtons(axradfvtv, ("Front view", "Top view"))

    plot_scene_fv(axs[0, 0], luf_min, cams_kr, cams_ang, 0.0, 10.0, 50.0,
                  axs_titles[0])
    plot_scene_fv(axs[0, 1], luf_max, cams_kr, cams_ang, 0.0, 10.0, 50.0,
                  axs_titles[1])
    plot_scene_fv(axs[1, 0], luf_min, cams_kr, cams_ang, 0.0, 10.0, 50.0,
                  axs_titles[2])

    def update(val):
        """Updates the whole figure by clearing all plots and redrawing with
           new parameters coming from widgets
        """
        print("UPDATE")
        axs[0, 0].clear()
        axs[0, 1].clear()
        axs[1, 0].clear()

        if USING_BRACKET:
            for i in range(NCAMS):
                cur_sca = sca[i].val
                sca[i].valtext.set_text(BRACKET_ANGLES[int(cur_sca)])
            cams_ang = np.radians([BRACKET_ANGLES[int(x.val)] for x in sca])
        else:
            cams_ang = np.radians([x.val for x in sca])

        luf_ang = np.radians(sluf.val)
        cams_kr = [x.val for x in sck]
        bldgh = sbldgh.val
        bldgd = sbldgd.val
        bldgw = sbldgw.val

        if radfvtv.value_selected == "Front view":
            plot_scene_fv(axs[0, 0], luf_min, cams_kr, cams_ang, bldgh, bldgd,
                          bldgw, axs_titles[0])
            plot_scene_fv(axs[0, 1], luf_max, cams_kr, cams_ang, bldgh, bldgd,
                          bldgw, axs_titles[1])
            plot_scene_fv(axs[1, 0], luf_ang, cams_kr, cams_ang, bldgh, bldgd,
                          bldgw, axs_titles[2])

        else:
            plot_scene_tv(axs[0, 0], luf_min, cams_kr, cams_ang, bldgh, bldgd,
                          bldgw, axs_titles[0])
            plot_scene_tv(axs[0, 1], luf_max, cams_kr, cams_ang, bldgh, bldgd,
                          bldgw, axs_titles[1])
            plot_scene_tv(axs[1, 0], luf_ang, cams_kr, cams_ang, bldgh, bldgd,
                          bldgw, axs_titles[2])

        fig.canvas.draw_idle()

    # Initial update now that everything is set up
    update(None)

    # Make all widgets interactive
    sluf.on_changed(update)
    for i in range(NCAMS):
        sck[i].on_changed(update)
        sca[i].on_changed(update)
    sbldgh.on_changed(update)
    sbldgd.on_changed(update)
    sbldgw.on_changed(update)
    radfvtv.on_clicked(update)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


if __name__ == '__main__':
    main()
