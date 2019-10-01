import sys
import time

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

    def get_dps2d(self):
        return self.dps2d

    def get_dps3d(self):
        return self.dps3d

    def get_pps(self):
        return self.pps

    def get_imtot(self):
        return self.imtot

    def get_cap_slices(self):
        return self.cap_slices

    def add_to_cap_slices(self, val):
        self.cap_slices.append(val)


def trigger_logic(h_imu, head, r_imu, roll, cap_slices, pps, imtot):
    slice_complete = bool(not(cap_slices.count([head.get('curr'), roll.get('curr')]) < pps))

    head_moved = bool((head.get('curr') != head.get('prev')) and (head.get('shft') != head.get('shpr')))
    roll_moved = bool((roll.get('curr') != roll.get('prev')) and (roll.get('shft') != roll.get('shpr')))

    crane_moved = head_moved or roll_moved

    last_slice = bool(len(cap_slices) + pps > imtot)

    if bool(not slice_complete and (crane_moved or last_slice)):
        return [True, head_moved, roll_moved]

    else:
        return [False, False, False]


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

    P1 = {'x': 0x00, 'y': 0x01, 'z': 0x02, 'x_sign': 0x00, 'y_sign': 0x00, 'z_sign': 0x00}
    PC = {'x': 0x01, 'y': 0x02, 'z': 0x00, 'x_sign': 0x01, 'y_sign': 0x01, 'z_sign': 0x00}

    MODE_NDOF = 0x0C
    MODE_IMU = 0x08

    if not bno.begin(mode=MODE_IMU):
        raise RuntimeError('FAILED TO INITIALIZE')

    bno.set_axis_remap(**PC)

    stat, stest, err = bno.get_system_status()
    print('System status: {} --- self-test result (should be 0x0F): 0x{:02X} --- error: {}'.format(stat, stest, err))
    print()

    dps2d = cap.get_dps2d()
    dps3d = cap.get_dps3d()
    imtot = cap.get_imtot()
    pps = cap.get_pps()

    while True:
        y_imu, p_imu, r_imu = bno.read_euler()
        head['curr'] = int(((y_imu + 0.5*dps2d)%360)/dps2d)
        head['shft'] = int((y_imu%360)/dps2d)

        roll['curr'] = int((r_imu%90)/dps3d)
        roll['shft'] = int(((r_imu + 0.5*dps3d)%90)/dps3d)

        res = trigger_logic(y_imu, head, r_imu, roll, cap.get_cap_slices(), pps, imtot)

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

        cap_slices = cap.get_cap_slices()
        sys.stdout.write('\ry={:+7.2f} -- p={:+7.2f}: -- r={:+7.2f} -- {}/{} images taken -- {}'.format(y_imu, p_imu, r_imu, len(cap_slices), imtot, cap_slices))
        sys.stdout.flush()

        if len(cap_slices) == imtot:
            print("\nFINAL CAPTURE TAKEN, WILL THUS QUIT LOOP")
            break

        time.sleep(.05)


if __name__ == '__main__':
    main()
