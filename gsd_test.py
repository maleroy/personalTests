from math import radians, sin, cos, tan, sqrt


BRACKET_ANGLE_DEG = 0
CAM_RADIUS = 0.5  # 19.63
JIB_ALT = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]# 57.6
REFERENCE_LEVEL = 0

YAW_IMU = 0
LUF_ANG = [0, 10, 20, 30, 40, 50, 60, 70, 80]  #8.4375 

IDXGSD = 0
GSD_LIM = 3

SODA_HFOV_V = 0.5 * 45.4   # [deg]
SODA_HFOV_H = 0.5 * 64.2   # [deg] 
SODA_FOCAL_LENGTH = 10.6   # [mm]
SODA_SENSOR_WIDTH = 12.75  # [mm]
SODA_SENSOR_HEIGHT = 8.5   # [mm]
SODA_IMAGE_WIDTH = 5472    # [pxl]
SODA_IMAGE_HEIGHT = 3648   # [pxl]

def sph2car(r_s, phi, theta, center=None):
    """Converts spherical coordinates into cartesian ones (and offsets origin)

    Args:
        r_s (float): radius of sphere [-]
        phi (float): azimuthal angle in [deg]
        theta (float): polar angle in [deg]
        center (list, optional): [description]. Defaults to None.

    Returns:
        list: cartesian coordinates of point
    """
    if center is None:
        center = [0, 0, 0]

    x_c = r_s*sin(radians(theta))*cos(radians(phi))
    y_c = r_s*sin(radians(theta))*sin(radians(phi))
    z_c = r_s*cos(radians(theta))

    return [i+j for i, j in zip([x_c, y_c, z_c], center)]


def get_footprint_radial_coords(p_cam, delta_h, yaw_deg, luf_ang):
    """Computes radius x&y footprint coordinates of image considering
       camera specs

    Args:
        p_cam (list of 3 floats): Camera [x, y, z] position.
        delta_h (float): vertical distance to reference level.
        yaw_deg (float): yaw value in [deg].
        luf_ang (float): luffing angle in [deg].

    Returns:
        list of 6 floats: radius x&y coordinates of footprint
    """
    p_r_old = sqrt(p_cam[0]**2+p_cam[1]**2)

    p_r_new = p_r_old + delta_h*tan(radians(BRACKET_ANGLE_DEG + luf_ang))

    p_r_new_in = p_r_old + delta_h*tan(radians(BRACKET_ANGLE_DEG - SODA_HFOV_V + luf_ang))

    p_r_new_out = p_r_old + delta_h*tan(radians(BRACKET_ANGLE_DEG + SODA_HFOV_V + luf_ang))

    yaw_rad = radians(yaw_deg)
    p_r_new_x = p_r_new*cos(yaw_rad)
    p_r_new_y = p_r_new*sin(yaw_rad)
    p_r_new_in_x = p_r_new_in*cos(yaw_rad)
    p_r_new_in_y = p_r_new_in*sin(yaw_rad)
    p_r_new_out_x = p_r_new_out*cos(yaw_rad)
    p_r_new_out_y = p_r_new_out*sin(yaw_rad)

    return [p_r_new_x, p_r_new_y,
            p_r_new_in_x, p_r_new_in_y,
            p_r_new_out_x, p_r_new_out_y]

def get_sorted_gsds(p_cam, delta_h, p_rs):
    """Returns a sorted list of GSDs for a tilted camera

    Args:
        p_cam (list of 3 floats): Camera [x, y, z] position.
        delta_h (float): vertical distance to reference level.
        p_rs (list of 6 floats): footprint radius coordinates.

    Returns:
        list of 3 floats: sorted list of GSDs.
    """
    pxlhi = sqrt(
        (p_cam[0]-p_rs[2])**2
        + (p_cam[1] - p_rs[3])**2
        + (delta_h)**2)

    pxlhm = sqrt(
        (p_cam[0]-p_rs[0])**2
        + (p_cam[1] - p_rs[1])**2
        + (delta_h)**2)

    pxlho = sqrt(
        (p_cam[0]-p_rs[4])**2
        + (p_cam[1] - p_rs[5])**2
        + (delta_h)**2)

    gsdi = get_gsd(pxlhi)[IDXGSD]
    gsdm = get_gsd(pxlhm)[IDXGSD]
    gsdo = get_gsd(pxlho)[IDXGSD]
    
    gsds = [gsdi, gsdm, gsdo]

    return [gsds, sorted(range(len(gsds)), key=lambda k: gsds[k]), [pxlhi, pxlhm, pxlho]]


def get_gsd(pxlh, units='m'):
    """Computes a single (double) GSD value based on a height value

    Args:
        pxlh (float): distance to a reference level.

    Returns:
        list of 2 floats: GSD values (one for each camera axis)
    """
    if units == 'm':
        k_unit = 100
    elif units == 'cm':
        k_unit = 1
    elif units == 'in':
        k_unit = 2.54
    else:
        return [-1, -1]

    return [(SODA_SENSOR_WIDTH*pxlh*k_unit)/(
        SODA_FOCAL_LENGTH*SODA_IMAGE_WIDTH),
            (SODA_SENSOR_HEIGHT*pxlh*k_unit)/(
                SODA_FOCAL_LENGTH*SODA_IMAGE_HEIGHT)]


def main():
    dl = GSD_LIM*SODA_IMAGE_WIDTH*SODA_FOCAL_LENGTH/SODA_SENSOR_WIDTH/100
    for jib_alt in JIB_ALT:
        print("\n\nStarting computation with \033[0;32mCAM_RADIUS={}[m]\033[m "
              "and \033[0;32mjib_alt={}[m]\033[m, thus \033[0;32mdl={:.2f}[m]\033[m\n".format(CAM_RADIUS, jib_alt, dl)) 
        for luf_ang in LUF_ANG:
            if luf_ang + BRACKET_ANGLE_DEG + SODA_HFOV_V < 90.0 and luf_ang + BRACKET_ANGLE_DEG - SODA_HFOV_V > -90:
                p_cam = sph2car(CAM_RADIUS, YAW_IMU, 90-luf_ang, [0, 0, jib_alt])
                delta_h = p_cam[2] - REFERENCE_LEVEL

                p_rs = get_footprint_radial_coords(p_cam, delta_h, YAW_IMU, luf_ang)
                gsds = get_sorted_gsds(p_cam, delta_h, p_rs)
                vals_g = [i>GSD_LIM for i in gsds[0]]
                
                colors_g = [0]*3
                for idx, myb in enumerate(vals_g):
                    colors_g[idx] = '\033[0;91m' if myb else '\033[0;96m'
                
                hl = sqrt(dl**2/((tan(radians(BRACKET_ANGLE_DEG + luf_ang)))**2+1))
                hli = sqrt(dl**2/((tan(radians(BRACKET_ANGLE_DEG + luf_ang - SODA_HFOV_V)))**2+1))
                hlo = sqrt(dl**2/((tan(radians(BRACKET_ANGLE_DEG + luf_ang + SODA_HFOV_V)))**2+1))

                hls = []
                for i in range(3):
                    if gsds[1][i] == 0:
                        hls.append(hli)
                    elif gsds[1][i] == 1:
                        hls.append(hl)
                    elif gsds[1][i] == 2:
                        hls.append(hlo)
                    else:
                        print("\nERROR")
                        exit(0)

                vals_h = [i>jib_alt for i in [hli, hl, hlo]]
                colors_h = [0]*3
                for idx, myb in enumerate(vals_h):
                    colors_h[idx] = '\033[0;96m' if myb else '\033[0;91m'

                vals_p = [i>dl for i in gsds[2]]
                colors_p = [0]*3
                for idx, myb in enumerate(vals_p):
                    colors_p[idx] = '\033[0;91m' if myb else '\033[0;96m'

                print("luf_ang={:2d}[deg] & h_cam={:.2f}[m] --> gsds=[{}{:6.3f}\033[m, {}{:6.3f}\033[m, {}{:6.3f}\033[m][cm/pxl] ("
                      #"hls=[{}{:6.2f}\033[m, {}{:6.2f}\033[m, {}{:6.2f}\033[m][m], "
                      "pxlhs=[{}{:6.2f}\033[m, {}{:6.2f}\033[m, {}{:6.2f}\033[m][m]"
                      ")".format(
                    luf_ang, delta_h,
                    colors_g[0], gsds[0][0],
                    colors_g[1], gsds[0][1],
                    colors_g[2], gsds[0][2],
                    #colors_h[0], hls[0],
                    #colors_h[1], hls[1],
                    #colors_h[2], hls[2],
                    colors_p[0], gsds[2][0],
                    colors_p[1], gsds[2][1],
                    colors_p[2], gsds[2][2]))

            else:
                print("Skipping computation with luf_ang={}[deg] as angle sum is {}".format(luf_ang, luf_ang + BRACKET_ANGLE_DEG + SODA_HFOV_V))

    print('\n\n')


if __name__ == '__main__':
    main()
