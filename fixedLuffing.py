import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from matplotlib.widgets import Slider

NCAMS = 3
TOW_H = 60
JIB_L = 68
CAM_HFOV = np.radians(0.5*45.4)


def plot_scene(cur_ax, luf_ang, cams_kr, cams_ang, ax_title):
    cur_ax.plot([0, 0, 0 + JIB_L*np.cos(luf_ang)],
                [0, TOW_H, TOW_H + JIB_L*np.sin(luf_ang)],
                color='orange')

    for i in range(NCAMS):
        plot_cam(cur_ax, cams_kr[i], cams_ang[i], luf_ang)

    cur_ax.title.set_text(ax_title)
    cur_ax.set_xlim([-200, 200])
    cur_ax.set_ylim([-10, 200])
    cur_ax.grid(True)
    print("\n")


def plot_cam(cur_ax, cam_kr, cam_ang, luf_ang):
    fov_limit = np.radians(60.0)
    cur_ax.scatter([0 + cam_kr*JIB_L*np.cos(luf_ang),
                    0 + cam_kr*JIB_L*np.cos(luf_ang),
                    0 + cam_kr*JIB_L*np.cos(luf_ang)],
                   [TOW_H + cam_kr*JIB_L*np.sin(luf_ang),
                    TOW_H + cam_kr*JIB_L*np.sin(luf_ang),
                    TOW_H + cam_kr*JIB_L*np.sin(luf_ang)],
                   color='blue')

    print("Cam @ {:4.2f} with luf_ang {:5.1f}: left border is {:5.1f} and "
          "right border is {:5.1f}".format(
              cam_kr, np.degrees(luf_ang),
              np.degrees(luf_ang+cam_ang-CAM_HFOV),
              np.degrees(luf_ang+cam_ang+CAM_HFOV)))

    if (luf_ang+cam_ang-CAM_HFOV) >= fov_limit:
        print("Cam @ {:4.2f} with luf_ang {:5.1f}: FOV not pointing to the "
              "ground!\n".format(cam_kr, np.degrees(luf_ang)))

    elif (luf_ang+cam_ang) >= fov_limit:
        print("Cam @ {:4.2f} with luf_ang {:5.1f}: FOV mid-line pointing "
              "above the horizon!\n".format(cam_kr, np.degrees(luf_ang)))

    elif (luf_ang+cam_ang+CAM_HFOV) >= fov_limit:
        print("Cam @ {:4.2f} with luf_ang {:5.1f}: FOV is surely pretty far "
              "from the crane!\n".format(cam_kr, np.degrees(luf_ang)))

    else:
        cur_ax.add_patch(
            Polygon(np.array([[0 + cam_kr*JIB_L*np.cos(luf_ang),
                               TOW_H + cam_kr*JIB_L*np.sin(luf_ang)],
                              [0 + cam_kr*JIB_L*np.cos(luf_ang) + (
                                  TOW_H + cam_kr*JIB_L*np.sin(luf_ang))*np.tan(
                                      luf_ang+cam_ang-CAM_HFOV), 0],
                              [0 + cam_kr*JIB_L*np.cos(luf_ang) + (
                                  TOW_H + cam_kr*JIB_L*np.sin(luf_ang))*np.tan(
                                      luf_ang+cam_ang+CAM_HFOV), 0]]),
                    fc='blue', ec=None, alpha=0.3))


def main():
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
    jump = (vert_pos-vert_low)/(3*NCAMS+2)
    vert_pos -= jump
    for i in range(NCAMS):
        vert_pos -= jump
        axck.append(plt.axes([hori_pos, vert_pos, 0.2, 0.01]))
        vert_pos -= jump
        axca.append(plt.axes([hori_pos, vert_pos, 0.2, 0.01]))
        vert_pos -= jump

    ck_min = 0.0
    ck_max = 1.0
    ca_min = -85.0
    ca_max = 45.0

    cams_kr = [x/(NCAMS+1) for x in range(1, NCAMS+1)]
    pa_min = -45
    pa_max = 0
    cams_ang = np.radians(
        [pa_max-x/(NCAMS-1) for x in (pa_max-pa_min)*np.array(range(NCAMS))])
    axs_titles = ['At minimum luffing angle',
                  'At maximum luffing angle',
                  'At luffing angle from slider']

    sluf = Slider(axluf, 'Luffing angle ', 35, 85, valinit=35, valstep=1)
    sck = []
    sca = []
    for i in range(NCAMS):
        str_pos = "Cam" + str(i+1) + " rel. loc."
        sck.append(Slider(
            axck[i], str_pos, ck_min, ck_max,
            valinit=cams_kr[i], valstep=0.05))
        str_ang = "Cam" + str(i+1) + " angle   "
        sca.append(Slider(
            axca[i], str_ang, ca_min, ca_max,
            valinit=np.degrees(cams_ang[i]), valstep=5.0))

    plot_scene(axs[0, 0], luf_min, cams_kr, cams_ang, axs_titles[0])
    plot_scene(axs[0, 1], luf_max, cams_kr, cams_ang, axs_titles[1])
    plot_scene(axs[1, 0], luf_min, cams_kr, cams_ang, axs_titles[2])

    def update(val):
        luf_ang = np.radians(sluf.val)
        cams_ang = np.radians([x.val for x in sca])
        cams_kr = [x.val for x in sck]
        axs[0, 0].clear()
        axs[0, 1].clear()
        axs[1, 0].clear()
        plot_scene(axs[0, 0], luf_min, cams_kr, cams_ang, axs_titles[0])
        plot_scene(axs[0, 1], luf_max, cams_kr, cams_ang, axs_titles[1])
        plot_scene(axs[1, 0], luf_ang, cams_kr, cams_ang, axs_titles[2])
        fig.canvas.draw_idle()

    sluf.on_changed(update)
    for i in range(NCAMS):
        sck[i].on_changed(update)
        sca[i].on_changed(update)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


if __name__ == '__main__':
    main()
