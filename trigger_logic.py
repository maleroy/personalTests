import random

random.seed(0)

TEST = False

NOS2D = 4
DPS2D = int(360.0/NOS2D)

NOS3D = 3
DPS3D = int(90.0/NOS3D)

PPS = 2
IMTOT = int(NOS2D*NOS3D*PPS)
CAP_SLICES = []


def trigger_logic(h_imu, head, r_imu, roll):
    head_moved = head.get('curr') != head.get('prev') and head.get('shft') != head.get('shpr')
    roll_moved = roll.get('curr') != roll.get('prev') and roll.get('shft') != roll.get('shpr')
    crane_moved = bool(head_moved or roll_moved)
 
    slice_complete = bool(
        not( CAP_SLICES.count([head.get('curr'), roll.get('curr')]) < PPS ))

    last_slice = bool(len(CAP_SLICES) + PPS > IMTOT)
    print(slice_complete, crane_moved, last_slice, bool(not slice_complete and (crane_moved or last_slice)))

    if bool(not slice_complete and (crane_moved or last_slice)):
        return [True, head_moved, roll_moved]

    else:
        return [False, False, False]


def man_rand(s, e, n):
    res = []

    for i in range(n):
        res.append(random.randint(s, e))

    return res


def main():
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

    n_samples = 50
    
    if TEST:
        h_imu = [123, 123, 12, 12, 123]
        h_imu = [248, 207, 155]
        r_imu = [40, 41, 40]
    else:
        #h_imu = random.sample(range(360), n_samples)
        #r_imu = random.sample(range(0, 90), n_samples)
        h_imu = man_rand(0, 359, n_samples)
        r_imu = man_rand(0, 89, n_samples)    

    for i in range(n_samples):
        print("{:2d}: [{:3d}, {:3d}]".format(i, h_imu[i], r_imu[i]))
    #print("h_imu: {}".format([x[1] for x in enumerate(h_imu)]))
    #print("r_imu: {}".format([x[1] for x in enumerate(r_imu)]))
    
    print()
    imgs_taken = 0

    for i, item in enumerate(h_imu):
        print("{:2d}/{:2d}".format(i,len(h_imu)))
        head['curr'] = int(((item + 0.5*DPS2D)%360)/DPS2D)
        head['shft'] = int((item%360)/DPS2D)

        roll['curr'] = int((r_imu[i]%90)/DPS3D)
        roll['shft'] = int(((r_imu[i] + 0.5*DPS3D)%90)/DPS3D)

        #print('{} -> {}'.format(item, head['curr']) )
        #rint('head: '.join(['{0}:{1} --- '.format(k, v) for k,v in sorted(head.items())]))
        #rint('roll: '.join(['{0}:{1} --- '.format(k, v) for k,v in sorted(roll.items())]))
        print("h_imu: {:3d} ------ head: prev:{}, curr:{} --- shpr:{}, shft:{}".format(h_imu[i], head.get('prev'), head.get('curr'), head.get('shpr'), head.get('shft')))
        print("r_imu: {:3d} ------ roll: prev:{}, curr:{} --- shpr:{}, shft:{}".format(r_imu[i], roll.get('prev'), roll.get('curr'), roll.get('shpr'), roll.get('shft')))
        
        r = trigger_logic(item, head, r_imu[i], roll)
        if r[0]:
            if r[1] and r[2]:
                print("Change due to both!")
                head['prev'] = head.get('curr')
                head['shpr'] = head.get('shft')
                roll['prev'] = roll.get('curr')
                roll['shpr'] = roll.get('shft')

            elif r[1]:
                print("Change due to head")
                head['prev'] = head.get('curr')
                head['shpr'] = head.get('shft')

            elif r[2]:
                print("Change due to roll")
                roll['prev'] = roll.get('curr')
                roll['shpr'] = roll.get('shft')

            else:
                print("WTF DID NOT EXPECT THIS TO HAPPEN; MUST INVESTIGATE")

            imgs_taken += 1
            CAP_SLICES.append([head.get('curr'), roll.get('curr')])

            print("n_pics: ", len(CAP_SLICES), " --- ", CAP_SLICES)
            if len(CAP_SLICES)==IMTOT:
                print("\nFINAL PICTURE TAKEN, WILL THUS QUIT LOOP EARLY")
                break

        else:
            print("Would have been {}".format([head.get('curr'), roll.get('curr')]))

        print()
    print()


if __name__ == '__main__':
    main()
