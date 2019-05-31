import numpy as np
import re
import matplotlib.pyplot as plt
import cv2 as cv

np.set_printoptions(suppress=True, precision=4, sign='+')


def main():
    logFile = open('myEKFlogFile.txt','r')

    obs       = []
    prior_est = []
    diff      = []
    gps_p_est = []
    gps_v_est = []
    gps_q_est = []    

    for line in logFile:
        #print('I\'m reading: ', line)
        if "Observation" in line:
            obs.append(re.findall(r"[-+]?\d*\.\d+|\d+",line))
        elif "Prior estimate" in line:
            prior_est.append(re.findall(r"[-+]?\d*\.\d+|\d+",line))
        elif "Difference" in line:
            diff.append(re.findall(r"[-+]?\d*\.\d+|\d+",line))
        elif "Exit" in line:
            myStr = re.findall(r"[-+]?\d*\.\d+|\d+",line)
            gps_p_est.append(myStr[5:8])
            gps_v_est.append(myStr[8:11])
            gps_q_est.append(myStr[11:])
        else:
            pass


    for i in range(len(obs)):
        if([]==obs[i]):
            obs[i] = ['nan','nan','nan']
    for i in range(len(prior_est)):
        if([]==prior_est[i]):
            prior_est[i] = ['nan','nan','nan']
    for i in range(len(diff)):
        if([]==diff[i]):
            diff[i] = ['nan','nan','nan','nan','nan','nan','nan','nan','nan']
    for i in range(len(gps_p_est)):
        if([]==gps_p_est[i]):
            gps_p_est[i] = ['nan','nan','nan']
    for i in range(len(gps_v_est)):
        if([]==gps_v_est[i]):
            gps_v_est[i] = ['nan','nan','nan']
    for i in range(len(gps_q_est)):
        if([]==gps_q_est[i]):
            gps_q_est[i] = ['nan','nan','nan','nan']


    obs       = np.asarray(obs,       dtype=np.float)
    prior_est = np.asarray(prior_est, dtype=np.float)
    diff      = np.asarray(diff,      dtype=np.float)
    gps_p_est = np.asarray(gps_p_est, dtype=np.float)
    gps_v_est = np.asarray(gps_v_est, dtype=np.float)
    gps_q_est = np.asarray(gps_q_est, dtype=np.float)

    fig, axs = plt.subplots(nrows=2,ncols=6, constrained_layout=True)
    gs = axs[1, 2].get_gridspec()
    for i in range(axs.shape[1]):
        for ax in axs[0:,-i-1]:
            ax.remove()
    fig.suptitle('myEKFlogFile.txt read')

    ax_x = fig.add_subplot(gs[0,0:2])
    ax_x.plot(      obs[:,0],'r',label='obs_x')
    ax_x.plot(prior_est[:,0],'g',label='prior_est_x')
    ax_x.plot(gps_p_est[:,0],'b',label='gps_p_est_x')
    ax_x.set_title('x values [m]')
    ax_x.legend(loc='lower right')

    ax_y = fig.add_subplot(gs[0,2:4])
    ax_y.plot(      obs[:,1],'r',label='obs_y')
    ax_y.plot(prior_est[:,1],'g',label='prior_est_y')
    ax_y.plot(gps_p_est[:,1],'b',label='gps_p_est_y')
    ax_y.set_title('y values [m]')
    ax_y.legend(loc='lower right')

    ax_z = fig.add_subplot(gs[0,4:])
    ax_z.plot(      obs[:,2],'r',label='obs_z')
    ax_z.plot(prior_est[:,2],'g',label='prior_est_z')
    ax_z.plot(gps_p_est[:,2],'b',label='gps_p_est_z')
    ax_z.set_title('z values [m]')
    ax_z.set_ylim(-40, 80)
    ax_z.legend(loc='lower right')

    ax_v = fig.add_subplot(gs[1,0:3])
    ax_v.plot(gps_v_est[:,0],'r',label='gps_v_est_x')
    ax_v.plot(gps_v_est[:,1],'g',label='gps_v_est_y')
    ax_v.plot(gps_v_est[:,2],'b',label='gps_v_est_z')
    ax_v.set_title('GPS V estimates [m/s]')
    ax_v.legend(loc='lower right')

    ax_q = fig.add_subplot(gs[1,3:])
    ax_q.plot(gps_q_est[:,0],'k',label='gps_q_est_w')
    ax_q.plot(gps_q_est[:,1],'r',label='gps_q_est_x')
    ax_q.plot(gps_q_est[:,2],'g',label='gps_q_est_y')
    ax_q.plot(gps_q_est[:,3],'b',label='gps_q_est_z')
    ax_q.set_title('Q estimates [-]')
    ax_q.legend(loc='lower right')

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

    pic, axs_pic = plt.subplots(1,1,constrained_layout=True)
    pic.suptitle('Going to the post office')

    img1 = cv.imread('stsulpice.png')
    img2 = cv.imread('stsulpice.png')

    origin = (277,169)
    pxls = 2
    k = 81./50.

    img1[origin[0]-3*pxls:origin[0]+3*pxls, origin[1]-3*pxls:origin[1]+3*pxls] = [0,0,0]
    img2[origin[0]-3*pxls:origin[0]+3*pxls, origin[1]-3*pxls:origin[1]+3*pxls] = [0,0,0]

    for i in range(700):
        if not(np.isnan(gps_p_est[i,0]) or np.isnan(gps_p_est[i,1])):
            img1[np.int(origin[0]-k*gps_p_est[i,0]-pxls):np.int(origin[0]-k*gps_p_est[i,0]+pxls),
                 np.int(origin[1]+k*gps_p_est[i,1]-pxls):np.int(origin[1]+k*gps_p_est[i,1]+pxls)] = [0,0,0]

    plt.imshow(img1)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

    pic, axs_pic = plt.subplots(1,1,constrained_layout=True)
    pic.suptitle('Returning from the post office')

    for i in range(700,gps_p_est[:,0].shape[0]):
        if not(np.isnan(gps_p_est[i,0]) or np.isnan(gps_p_est[i,1])):
            img2[np.int(origin[0]-k*gps_p_est[i,0]-pxls):np.int(origin[0]-k*gps_p_est[i,0]+pxls),
                 np.int(origin[1]+k*gps_p_est[i,1]-pxls):np.int(origin[1]+k*gps_p_est[i,1]+pxls)] = [0,0,0]

    plt.imshow(img2)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

if __name__=='__main__':
    main()
