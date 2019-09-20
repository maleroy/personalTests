import sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import axes3d


N_SECT_2D = 12
I_SECT_2D = 360./N_SECT_2D
N_SECT_3D = 8
I_SECT_3D = 180./N_SECT_3D
N_WIREFRAME = 3
N_SECT_TOT = N_SECT_2D*N_SECT_3D
SECT_PASSED = np.zeros((N_SECT_2D, N_SECT_3D), dtype=bool)

def sph2car(r, phi, theta):
    x = r*np.sin(np.radians(theta))*np.cos(np.radians(phi))
    y = r*np.sin(np.radians(theta))*np.sin(np.radians(phi))
    z = r*np.cos(np.radians(theta))
    return x, y, z


def plot_all(ax, X, Y, Z, px, py, pz):
    # Plot a 3D surface
    ax.plot_wireframe(X, Y, Z, colors="green")
    ax.plot([0, px], [0, py], [0, pz], c="green")
    ax.scatter(px, py, pz, s=50, c="green")

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_aspect('equal')
    bb = 105
    ax.set_xlim3d([-bb, bb])
    ax.set_ylim3d([-bb, bb])
    ax.set_zlim3d([-bb, bb])


def get_interval(l, v):
    for i in range(len(l)-1):
        if l[i] <= v <= l[i+1]:
            print("{} is between {} and {}".format(v, l[i], l[i+1]))
            return l[i], l[i+1], i
    print("Error")
    sys.exit()


def main():
    phi_l = np.linspace(0., 360., N_SECT_2D+1) - 360./(2*N_SECT_2D)
    theta_l = np.linspace(0., 180., N_SECT_3D+1)
    print(phi_l)
    print(theta_l)

    r = 100
    phi_init = 60
    theta_init = 33.75

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    axphi = plt.axes([0.1, 0.15, 0.2, 0.01])
    sphi = Slider(axphi, 'Azimuthal angle (phi)', -15, 345,
                  valinit=phi_init, valstep=1, color="blue")

    axtheta = plt.axes([0.1, 0.1, 0.2, 0.01])
    stheta = Slider(axtheta, 'Polar angle (theta)', 0, 180,
                    valinit=theta_init, valstep=1, color="blue")

    pa, pb, p_i = get_interval(phi_l, phi_init)
    phi = np.linspace(pa, pb, N_WIREFRAME+1)

    ta, tb, t_i = get_interval(theta_l, theta_init)
    theta = np.linspace(ta, tb, N_WIREFRAME+1)

    SECT_PASSED[p_i, t_i] = True

    Phi, Theta = np.meshgrid(phi, theta)
    X, Y, Z = sph2car(r, Phi, Theta)
    px, py, pz = sph2car(r, phi_init, theta_init)
    plot_all(ax, X, Y, Z, px, py, pz)

    def update(val):
        ax.clear()

        new_phi = sphi.val
        pl, pr, p_i = get_interval(phi_l, new_phi)
        phi = np.linspace(pl, pr, N_WIREFRAME+1)

        new_theta = stheta.val
        tl, tr, t_i = get_interval(theta_l, new_theta)
        theta = np.linspace(tl, tr, N_WIREFRAME+1)

        SECT_PASSED[p_i, t_i] = True
        # print(SECT_PASSED)
        for i in range(SECT_PASSED.shape[0]):
            for j in range(SECT_PASSED.shape[1]):
                if SECT_PASSED[i, j]:
                    old_phi = np.linspace(i*I_SECT_2D - 360./(2*N_SECT_2D), (i+1)*I_SECT_2D - 360./(2*N_SECT_2D), N_WIREFRAME+1)
                    old_theta = np.linspace(j*I_SECT_3D, (j+1)*I_SECT_3D, N_WIREFRAME+1)
                    old_Phi, old_Theta = np.meshgrid(old_phi, old_theta)
                    X, Y, Z = sph2car(r, old_Phi, old_Theta)
                    ax.plot_wireframe(X, Y, Z, colors="blue", alpha=0.1)

        Phi, Theta = np.meshgrid(phi, theta)
        X, Y, Z = sph2car(r, Phi, Theta)

        px, py, pz = sph2car(r, new_phi, new_theta)
        plot_all(ax, X, Y, Z, px, py, pz)

        fig.canvas.draw_idle()

    # Initial update now that everything is set up
    update(None)

    stheta.on_changed(update)
    sphi.on_changed(update)

    ax.view_init(45, 0)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


if __name__ == '__main__':
    main()
