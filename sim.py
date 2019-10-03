import sys
import time

from math import radians, sin, cos, tan, sqrt
from Adafruit_BNO055 import BNO055


class Capture(object):
    def __init__(self):
        self.nos2d = 4
        self.dps2d = int(360.0/self.nos2d)

        self.nos3d = 3
        self.dps3d = int(90.0/self.nos3d)

        self.pps = 2
        self.imtot = int(self.nos2d*self.nos3d*self.pps)
        self.cap_slices = []
        self.pr_slices = [[0]*self.nos2d for i in range(self.nos3d)]

        self.bracket_ang = -10

        self.r_cam = 50
        self.p_cam = [0, 0, 0]

        self.bldg_h = 0

        self.hfov_h = 0.5*64.2
        self.hfov_v = 0.5*45.4

        self.cam_r_limit = 75
        self.cam_active = True

        self.tow_x = 10
        self.tow_y = 20
        self.tow_z = 30
        self.tow_h = 40

    def get_act_status(self):
        return self.cam_active

    def set_act_status(self, p_r):
        self.cam_active = p_r < self.cam_r_limit

    def get_hfov_h(self):
        return self.hfov_h

    def get_hfov_v(self):
        return self.hfov_v

    def set_bldg_h(self, new_h):
        self.bldg_h = new_h

    def get_bldg_h(self):
        return self.bldg_h

    def set_p_cam(self, new_p):
        self.p_cam = new_p

    def get_p_cam(self):
        return self.p_cam

    def get_r_cam(self):
        return self.r_cam

    def get_nos2d(self):
        return self.nos2d

    def get_dps2d(self):
        return self.dps2d

    def get_nos3d(self):
        return self.nos3d

    def get_dps3d(self):
        return self.dps3d

    def get_pps(self):
        return self.pps

    def get_imtot(self):
        return self.imtot

    def get_cap_slices(self):
        return self.cap_slices

    def get_pr_slices(self):
        return self.pr_slices

    def get_bracket_ang(self):
        return self.bracket_ang

    def add_to_cap_slices(self, val):
        self.cap_slices.append(val)
        self.pr_slices[val[1]][val[0]] += 1


def trigger_logic(head, roll, cap_slices, pps, imtot):
    slice_complete = bool(
        not cap_slices.count([head.get('curr'), roll.get('curr')]) < pps)

    head_moved = bool(
        (head.get('curr') != head.get('prev'))
        and (head.get('shft') != head.get('shpr')))
    roll_moved = bool(
        (roll.get('curr') != roll.get('prev'))
        and (roll.get('shft') != roll.get('shpr')))

    crane_moved = head_moved or roll_moved

    last_slice = bool(len(cap_slices) + pps > imtot)

    if bool(not slice_complete and (crane_moved or last_slice)):
        return [True, head_moved, roll_moved]

    else:
        return [False, False, False]


def sph2car(r_s, phi, theta):
    x_c = r_s*sin(radians(theta))*cos(radians(phi))
    y_c = r_s*sin(radians(theta))*sin(radians(phi))
    z_c = r_s*cos(radians(theta))
    return [x_c, y_c, z_c]


def main():
    cap = Capture()

    head = {
        'curr': 0,
        'prev': -1,
        'shft': 0,
        'shpr': -1,
    }

    roll = {
        'curr': 0,
        'prev': -1,
        'shft': 0,
        'shpr': -1,
    }

    imgs_taken = 0

    bno = BNO055.BNO055(rst='P9_12')

    P1 = {'x': 0x00, 'y': 0x01, 'z': 0x02,
          'x_sign': 0x00, 'y_sign': 0x00, 'z_sign': 0x00}
    PC = {'x': 0x01, 'y': 0x02, 'z': 0x00,
          'x_sign': 0x01, 'y_sign': 0x01, 'z_sign': 0x00}

    MODE_NDOF = 0x0C
    MODE_IMU = 0x08

    if not bno.begin(mode=MODE_IMU):
        raise RuntimeError('FAILED TO INITIALIZE')

    bno.set_axis_remap(**PC)

    stat, stest, err = bno.get_system_status()
    print('System status: {} --- self-test result (should be 0x0F): 0x{:02X}'
          ' --- error: {}'.format(stat, stest, err))
    print()
    time.sleep(1)  # so that the IMU starts spitting data

    nos2d = cap.get_nos2d()
    dps2d = cap.get_dps2d()
    nos3d = cap.get_nos3d()
    dps3d = cap.get_dps3d()
    imtot = cap.get_imtot()
    pps = cap.get_pps()
    br_ang = cap.get_bracket_ang()

    cursor_up = (nos3d+1)*"\033[F"

    while True:
        y_imu, p_imu, r_imu = bno.read_euler()
        head['curr'] = int(((y_imu + 0.5*dps2d) % 360)/dps2d)
        head['shft'] = int((y_imu % 360)/dps2d)

        luf_ang = r_imu - br_ang
        roll['curr'] = int((luf_ang % 90)/dps3d)
        roll['shft'] = int(((luf_ang + 0.5*dps3d) % 90)/dps3d)

        cam_sph = sph2car(cap.get_r_cam(), y_imu, 90-luf_ang)
        cap.set_p_cam([x+y for x, y in zip(cam_sph, [
            cap.tow_x, cap.tow_y, cap.tow_z])])
        p_cam = cap.get_p_cam()

        luf_ang_rad = radians(luf_ang)

        delta_h = p_cam[2] - cap.get_bldg_h()
        # delta_t = delta_h*tan(radians(cap.get_hfov_h()))

        p_r_old = sqrt(p_cam[0]**2 + p_cam[1]**2)
        p_r_new = p_r_old + delta_h*tan(radians(br_ang)+luf_ang_rad)

        cap.set_act_status(p_r_new)

        pre_cap_slices = cap.get_cap_slices()
        cap_slices = cap.get_cap_slices()
        res = trigger_logic(head, roll, cap_slices, pps, imtot)

        if res[0]:
            if res[1] and res[2]:
                head['prev'] = head.get('curr')
                head['shpr'] = head.get('shft')
                roll['prev'] = roll.get('curr')
                roll['shpr'] = roll.get('shft')

            elif res[1]:
                head['prev'] = head.get('curr')
                head['shpr'] = head.get('shft')

            elif res[2]:
                roll['prev'] = roll.get('curr')
                roll['shpr'] = roll.get('shft')

            imgs_taken += 1
            cap.add_to_cap_slices([head.get('curr'), roll.get('curr')])

        sys.stdout.write("y={:+7.2f} -- p={:+7.2f}: -- r={:+7.2f}"
                         " -- cur_pos: [{}, {}] -- luf_ang={:+7.2f} as br_ang="
                         "{} -- p_cam=[{:+7.2f}, {:+7.2f}, {:+7.2f}] -- p_r_ol"
                         "d={:+7.2f} and p_r_new={:+7.2f} -- Cam act? -> {}\n"
                         .format(y_imu, p_imu, r_imu,
                                 head.get('curr'), roll.get('curr'), luf_ang,
                                 br_ang, *[i for i in cap.get_p_cam()],
                                 p_r_old, p_r_new, cap.get_act_status()))

        pr_slices = cap.get_pr_slices()
        for i in range(nos3d):
            sys.stdout.write(str(pr_slices[-i-1])+"\n")

        sys.stdout.write(cursor_up)
        sys.stdout.flush()

        if len(cap_slices) == imtot:
            print((nos3d+2)*"\n" + "FINAL PIC TAKEN, WILL THUS QUIT LOOP\n")
            break

        time.sleep(.05)


if __name__ == '__main__':
    main()
