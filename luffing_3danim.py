from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import mpl_toolkits.mplot3d.art3d as art3d


fig = plt.figure()
ax = p3.Axes3D(fig)
circ_arr = []


def gen(n):
    r = 95
    inc = 55
    az = 0

    full = np.zeros((3, n))
    full[:, 0] = comp_cart_from_sph(r, inc, az)

    len_step0 = 50  # just a bit stopped
    len_step1 = 60  # 60 deg from dir 6 to 8
    len_step2 = 50  # luff 35 to 85
    len_step3 = 195 # 195 deg from dir 8 to 2.5
    len_step4 = 40  # luff 85 to 45
    len_step5 = 40  # luff 45 to 85
    len_step6 = 255 # 255 from dir 2.5 to 6 (CCW)
    len_step7 = 50  # luff from 85 to 35
    lens = np.cumsum([len_step0, len_step1, len_step2, len_step3,
                      len_step4, len_step5, len_step6, len_step7])
    for i in range(1, n):
        if i < lens[0]:
            inc = inc
        elif i < lens[1]:
            az -= 1
        elif i < lens[2]:
            inc -= 1
        elif i < lens[3]:
            az -= 1
        elif i < lens[4]:
            inc += 1
        elif i < lens[5]:
            inc -= 1
        elif i < lens[6]:
            az += 1
        elif i < lens[7]:
            inc += 1
        full[:, i] = comp_cart_from_sph(r, inc, az)

    return full


def comp_cart_from_sph(r, inc, az):
    return np.array([r*np.sin(np.radians(inc))*np.cos(np.radians(az)),
                     r*np.sin(np.radians(inc))*np.sin(np.radians(az)),
                     r*np.cos(np.radians(inc))])


def plot_3d_cylinder(ax, radius, height, elevation=0, resolution=3, color='orange', alpha=0.5, x_center = 0, y_center = 0):

    x = np.linspace(x_center-radius, x_center+radius, resolution)
    z = np.linspace(elevation, elevation+height, resolution)
    X, Z = np.meshgrid(x, z)

    Y = np.sqrt(radius**2 - (X - x_center)**2) + y_center # Pythagorean theorem

    ax.plot_surface(X, Y, Z, linewidth=0, color=color, alpha=alpha)
    ax.plot_surface(X, (2*y_center-Y), Z, linewidth=0, color=color, alpha=alpha, zorder=100)

    floor = Circle((x_center, y_center), radius, color=color, alpha=alpha, zorder=80)
    ax.add_patch(floor)
    art3d.pathpatch_2d_to_3d(floor, z=elevation, zdir="z")

    ceiling = Circle((x_center, y_center), radius, color=color, alpha=alpha)
    ax.add_patch(ceiling)
    art3d.pathpatch_2d_to_3d(ceiling, z=elevation+height, zdir="z")


def update(num, data, line, X, Y, Z):
    global surf
    global circ_arr
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])

    surf.remove()

    r_max = data[2, num]
    h_max = r_max*np.tan(np.radians(18))
    r = np.linspace(0, r_max, 10)
    p = np.linspace(0, 2*np.pi, 10)
    R, P = np.meshgrid(r, p)

    if num%10 == 0:
        p = Circle((data[0, num], data[1, num]), h_max,
                   alpha=0.1, zorder=-1)

        if True:
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
        else:
            circ_arr.append(p)
            circ_coll = PatchCollection(circ_arr)
            ax.add_collection3d(circ_coll)
            print(circ_coll._paths)
            print()
        
    Z = data[2, num] + R-r_max
    X = data[0, num] + (h_max-h_max*R/r_max)*np.cos(P)
    Y = data[1, num] + (h_max-h_max*R/r_max)*np.sin(P)
    surf = ax.plot_surface(X, Y, Z, color='g', zorder=1000)

N = 1000
data = gen(N)
#rint(data)
line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

# Setting the axes properties
lim = 100.0

r_max = 190
h_max = 50
r = np.linspace(0, r_max, 10)
p = np.linspace(0, 2*np.pi, 10)
R, P = np.meshgrid(r, p)

Z = R-r_max
X = (h_max-h_max*R/r_max)*np.cos(P)
Y = (h_max-h_max*R/r_max)*np.sin(P)
surf = ax.plot_surface(X, Y, Z, color='g')

plot_3d_cylinder(ax, 5, 50)

ax.set_xlim3d([-lim, lim])
ax.set_xlabel('X')

ax.set_ylim3d([-lim, lim])
ax.set_ylabel('Y')

ax.set_zlim3d([0, lim])
ax.set_zlabel('Z')

ax.view_init(elev=60, azim=0)

ani = animation.FuncAnimation(fig, update, N, fargs=(data, line, X, Y, Z),
                              interval=10000/N, blit=False)
#ani.save('matplot003.gif', writer='imagemagick')
plt.show()
