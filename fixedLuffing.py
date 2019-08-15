import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Polygon
from matplotlib.widgets import Slider

n_cams = 3
tow_h = 60
jib_l = 68
cam_hfov = np.radians(0.5*45.4)

def plotScene(ax, luf_ang, cams_kr, cams_ang, ax_title):
    ax.plot([0,     0,     0 + jib_l*np.cos(luf_ang)],
            [0, tow_h, tow_h + jib_l*np.sin(luf_ang)],
            color='orange')

    for i in range(n_cams):
        plotCam(ax, cams_kr[i], cams_ang[i], luf_ang)

    ax.title.set_text(ax_title)
    ax.grid(True)
    print("\n")

def plotCam(ax, cam_kr, cam_ang, luf_ang):
    fov_limit = np.radians(60.0)
    ax.scatter([    0 + cam_kr*jib_l*np.cos(luf_ang),     0 + cam_kr*jib_l*np.cos(luf_ang),     0 + cam_kr*jib_l*np.cos(luf_ang)],
               [tow_h + cam_kr*jib_l*np.sin(luf_ang), tow_h + cam_kr*jib_l*np.sin(luf_ang), tow_h + cam_kr*jib_l*np.sin(luf_ang)],
               color='blue')

    print("Cam @ {:4.2f} with luf_ang {:5.1f}: left border is {:5.1f} and right border is {:5.1f}".format(cam_kr, np.degrees(luf_ang), np.degrees(luf_ang+cam_ang-cam_hfov), np.degrees(luf_ang+cam_ang+cam_hfov)))

    if (luf_ang+cam_ang-cam_hfov)>=fov_limit:
        print("Cam @ {:4.2f} with luf_ang {:5.1f}: FOV not pointing to the ground!\n".format(cam_kr, np.degrees(luf_ang)))

    elif (luf_ang+cam_ang)>=fov_limit:
        print("Cam @ {:4.2f} with luf_ang {:5.1f}: FOV mid-line pointing above the horizon!\n".format(cam_kr, np.degrees(luf_ang)))

    elif (luf_ang+cam_ang+cam_hfov)>=fov_limit:
        print("Cam @ {:4.2f} with luf_ang {:5.1f}: FOV is surely pretty far from the crane!\n".format(cam_kr, np.degrees(luf_ang)))

    else:
        ax.add_patch(Polygon(np.array([[0 + cam_kr*jib_l*np.cos(luf_ang),   tow_h + cam_kr*jib_l*np.sin(luf_ang)],
                                       [0 + cam_kr*jib_l*np.cos(luf_ang) + (tow_h + cam_kr*jib_l*np.sin(luf_ang))*np.tan(luf_ang+cam_ang-cam_hfov), 0],
                                       [0 + cam_kr*jib_l*np.cos(luf_ang) + (tow_h + cam_kr*jib_l*np.sin(luf_ang))*np.tan(luf_ang+cam_ang+cam_hfov), 0]]),
                                 fc='blue', ec=None, alpha=0.3))


def main():
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    plt.setp(axs.flat, aspect=1.0)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    
    axs[1, 1].remove()

    luf_min = np.radians(35)
    luf_max = np.radians(85)

    cams_kr = [0.25, 0.50, 0.75]

    axluf = plt.axes([0.7, 0.45, 0.2, 0.01])
    axca1 = plt.axes([0.7, 0.42, 0.2, 0.01])
    axca2 = plt.axes([0.7, 0.39, 0.2, 0.01])
    axca3 = plt.axes([0.7, 0.36, 0.2, 0.01])

    ca_min = -85.0
    ca_max =  45.0

    cams_ang = np.radians([0.0, -40.0, -45.0]) 
    axs_titles = ['At minimum luffing angle','At maximum luffing angle','At luffing angle from slider']

    sluf = Slider(axluf, 'Luffing angle', 35.0, 85.0, valinit=35.0, valstep=1.0)
    sca1 = Slider(axca1, 'Camera 1 angle', ca_min, ca_max, valinit=np.degrees(cams_ang[0]), valstep=5.0)
    sca2 = Slider(axca2, 'Camera 2 angle', ca_min, ca_max, valinit=np.degrees(cams_ang[1]), valstep=5.0)
    sca3 = Slider(axca3, 'Camera 3 angle', ca_min, ca_max, valinit=np.degrees(cams_ang[2]), valstep=5.0)

    plotScene(axs[0, 0], luf_min, cams_kr, cams_ang, axs_titles[0])
    plotScene(axs[0, 1], luf_max, cams_kr, cams_ang, axs_titles[1])
    plotScene(axs[1, 0], luf_min, cams_kr, cams_ang, axs_titles[2])


    def update(val):
        axs[0, 0].clear()
        axs[0, 1].clear()
        axs[1, 0].clear()
        luf_ang = np.radians(sluf.val)
        cams_ang = np.radians([sca1.val, sca2.val, sca3.val])
        plotScene(axs[0, 0], luf_min, cams_kr, cams_ang, axs_titles[0])
        plotScene(axs[0, 1], luf_max, cams_kr, cams_ang, axs_titles[1])
        plotScene(axs[1, 0], luf_ang, cams_kr, cams_ang, axs_titles[2])
        fig.canvas.draw_idle()

    sluf.on_changed(update)
    sca1.on_changed(update)
    sca2.on_changed(update)
    sca3.on_changed(update)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

if __name__=='__main__':
    main()
