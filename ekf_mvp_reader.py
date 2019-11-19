import yaml
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import datetime
import numpy as np
import time


def plot_plot(arr_x, arr_y, titles):
    nsubs = arr_y.shape[-1]
    npp = arr_y.shape[0]
    
    if npp==1:
        colors = ['k']
    elif npp==2:
        colors = ['g', 'b']
        leg = ('sensor', 'es-ekf')
    elif npp==3:
        colors = ['c', 'y--', 'y--']
        leg = ('error', 'cov')

    fig, ax = plt.subplots(nsubs)
    for i in range(nsubs):
        for j in range(npp):
            ax[i].plot(arr_x, arr_y[j, :, i], colors[j])
        ax[i].set_ylabel(titles[i])
        ax[i].grid(True)
        if not npp==1:
            plt.legend(leg, loc='upper right')
    
    if not npp==3:
        fig.tight_layout()
    
    fig.autofmt_xdate()


def plot_plots(fig, subgrids, arr_x, arr_y, titles):
    nsubs = arr_y.shape[-1]
    npp = arr_y.shape[0]
    
    if npp==1:
        colors = ['k']
    elif npp==2:
        colors = ['g', 'b']
        leg = ('sensor', 'es-ekf')
    elif npp==3:
        colors = ['c', 'y--', 'y--']
        leg = ('error', 'cov')

    for i in range(nsubs):
        cur_ax = fig.add_subplot(subgrids[i])
        for j in range(npp):
            cur_ax.plot(arr_x, arr_y[j, :, i], colors[j])
        cur_ax.set_ylabel(titles[i])
        cur_ax.grid(True)
        if not npp==1:
            cur_ax.legend(leg, loc='upper left')
    
        fig.autofmt_xdate()


def main():
    with open('ekf_log.log', 'r') as f:
        ekf = yaml.load(f)

    timeout = 0.1
    s_d = 3

    a_times = []
    a_gps_lat = []
    a_gps_lon = []
    a_gps_alt = []
    a_imu_quat = []
    a_imu_rpy = []
    a_ned_obs = []
    a_p_cov = []
    a_p_est = []
    a_q_est = []
    a_rpy_cov = []
    a_rpy_est = []

    for k in sorted(ekf.keys()):
        a_times.append(datetime.datetime.strptime(k, '%a %b %d %H:%M:%S %Y'))
        for ks in sorted(ekf[k].keys()):
            if ks == 'gps_obs':
                a_gps_lat.append(ekf[k][ks].get('lat'))
                a_gps_lon.append(ekf[k][ks].get('lon'))
                a_gps_alt.append(ekf[k][ks].get('alt'))
            elif ks == 'imu_quat':
                a_imu_quat.append(ekf[k][ks])
            elif ks == 'imu_rpy':
                a_imu_rpy.append(ekf[k][ks])
            elif ks == 'ned_obs':
                a_ned_obs.append(ekf[k][ks])
            elif ks == 'p_cov':
                a_p_cov.append(ekf[k][ks])
            elif ks == 'p_est':
                a_p_est.append(ekf[k][ks])
            elif ks == 'q_est':
                a_q_est.append(ekf[k][ks])
            elif ks == 'rpy_cov':
                a_rpy_cov.append(ekf[k][ks])
            elif ks == 'rpy_est':
                a_rpy_est.append(ekf[k][ks])
            else:
                print("KeyError")

    a_gps_lat = np.array(a_gps_lat).reshape(len(a_gps_lat), 1)
    a_gps_lon = np.array(a_gps_lon).reshape(len(a_gps_lon), 1)
    a_gps_alt = np.array(a_gps_alt).reshape(len(a_gps_alt), 1)
    a_gps = np.hstack((np.hstack((a_gps_lat, a_gps_lon)), a_gps_alt))
    a_imu_quat = np.array(a_imu_quat)
    a_imu_rpy = np.degrees(np.array(a_imu_rpy))
    a_ned_obs = np.array(a_ned_obs)
    a_p_cov = np.array(a_p_cov)
    a_p_est = np.array(a_p_est)
    a_q_est = np.array(a_q_est)
    a_rpy_cov = np.degrees(np.array(a_rpy_cov))
    a_rpy_est = np.degrees(np.array(a_rpy_est))

    str_cov_d = ' error and covariances [deg]'
    str_cov_m = ' error and covariances [m]'

    my_var = 1

    if my_var:
        fig = plt.figure()
        grid = gridspec.GridSpec(38, 2)
        plot_plots(fig, [grid[:4, 0], grid[4:8, 0], grid[8:12, 0]], a_times, np.array([a_gps]), ['Lat\n[deg]', 'Lon\n[deg]', 'Alt\n[m]'])
        plot_plots(fig, [grid[:3, 1], grid[3:6, 1], grid[6:9, 1], grid[9:12, 1]], a_times, np.array([a_imu_quat, a_q_est]), ['q_w', 'q_x', 'q_y', 'q_z'])
        plot_plots(fig, [grid[13:17, 0], grid[17:21, 0], grid[21:25, 0]], a_times, np.array([a_ned_obs, a_p_est]), ['N\n[m]', 'E\n[m]', 'D\n[m]'])
        plot_plots(fig, [grid[13:17, 1], grid[17:21, 1], grid[21:25, 1]], a_times, np.array([a_imu_rpy, a_rpy_est]), ['R\n[deg]', 'P\n[deg]', 'Y\n[deg]'])
        plot_plots(fig, [grid[26:30, 0], grid[30:34, 0], grid[34:, 0]], a_times, np.array([a_ned_obs - a_p_est, s_d*a_p_cov[:, :3], -s_d*a_p_cov[:, :3]]), ['N\n[m]', 'E\n[m]', 'D\n[m]'])
        plot_plots(fig, [grid[26:30, 1], grid[30:34, 1], grid[34:, 1]], a_times, np.array([a_imu_rpy - a_rpy_est, s_d*a_rpy_cov, -s_d*a_rpy_cov]), ['R\n[deg]', 'P\n[deg]', 'Y\n[deg]'])
    
        #rid.tight_layout(fig)

    else:
        plot_plot(a_times, np.array([a_gps]), ['Latitude [deg]', 'Longitude [deg]', 'Altitude [m]'])
        plot_plot(a_times, np.array([a_imu_quat, a_q_est]), ['q_w', 'q_x', 'q_y', 'q_z'])
        plot_plot(a_times, np.array([a_ned_obs, a_p_est]), ['North [m]', 'East [m]', 'Down [m]'])
        plot_plot(a_times, np.array([a_imu_rpy, a_rpy_est]), ['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]'])
        plot_plot(a_times, np.array([a_ned_obs - a_p_est, s_d*a_p_cov[:, :3], -s_d*a_p_cov[:, :3]]), ['North' + str_cov_m, 'East' + str_cov_m, 'Down' + str_cov_m])
        plot_plot(a_times, np.array([a_imu_rpy - a_rpy_est, s_d*a_rpy_cov, -s_d*a_rpy_cov]), ['Roll' + str_cov_d, 'Pitch' + str_cov_d, 'Yaw' + str_cov_d])

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


if __name__ == '__main__':
    main()
