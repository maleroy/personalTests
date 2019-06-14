#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" es_ekf_sixfab.py
Runs an ES-EKF for GNSS / IMU combination (BG96+LSM9DS1).
Run as is, but be aware this code is work in progress.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
__authors__ = ["Miguel Baquero",
               "Matthew Franklin",
               "Lazare Girardin",
               "Marc Leroy"]
__contact__ = "construction.tech@pix4d.com"
__copyright__ = "Copyright 2019, Pix4D"
__credits__ = ["Miguel Baquero",
               "Matthew Franklin",
               "Lazare Girardin",
               "Marc Leroy"]
__date__ = "2019/06/13"
__deprecated__ = False
__email__ = "construction.tech@pix4d.com"
__license__ = "GPLv3"
__maintainer__ = "Crane Camera R&D Team"
__status__ = "Production"
__version__ = "0.0.1"

import os
import os.path
import sys
import time

import RTIMU

import numpy as np

from cellulariot import cellulariot

np.set_printoptions(precision=4, suppress=True, sign='+', linewidth=204)


# GPS CONVERSION FUNCTIONS.


def geodetic2ecef(lat, lon, alt):
    """Converts Geodetic coordinates to ECEF on Earth.

    Args:
        lat (float): Latitude in [rad]
        lon (float): Longitude in [rad]
        alt (float): Altitude in [m]

    Returns:
        np.ndarray((3, 1), dtype='float64'): ECEF coordinates
    """
    x_i = 1./np.sqrt(1. - 0.00669437999014 * np.sin(lat) * np.sin(lat))

    ecef = np.zeros((3, 1), dtype=np.float)

    ecef[0, 0] = (6378137*x_i + alt) * np.cos(lat) * np.cos(lon)
    ecef[1, 0] = (6378137*x_i + alt) * np.cos(lat) * np.sin(lon)
    ecef[2, 0] = (6378137*x_i * (1. - 0.00669437999014) + alt) * np.sin(lat)

    return ecef


def ecef2ned(p_lat, p_lon, p_alt, init_ecef, ecef2ned_mat):
    """Converts ECEF coordinates to local NED coordinates.

    Args:
        p_lat (float): Latitude in [rad]
        p_lon (float): Longitude in [rad]
        p_alt (float): Altitude in [m]
        init_ecef ([type]): Initial frame coordinates in ECEF model
        ecef2ned_mat ([type]): Matrix that converts ECEF coords to NED

    Returns:
        np.ndarray((3,1), dtype='float64'): NED coordinates
    """
    xyz = geodetic2ecef(p_lat, p_lon, p_alt)

    vect = np.zeros((3, 1), dtype=np.float)
    vect[0, 0] = xyz[0, 0] - init_ecef[0, 0]
    vect[1, 0] = xyz[1, 0] - init_ecef[1, 0]
    vect[2, 0] = xyz[2, 0] - init_ecef[2, 0]

    ret = ecef2ned_mat @ vect
    ret[2, 0] = -ret[2, 0]

    return ret


def ned2geodetic(ned, init_ecef, ned2ecef_mat):
    """Converts NED coordinates to Geodetic.

    Args:
        ned ([type]): [description]
        init_ecef ([type]): [description]
        ned2ecef_mat ([type]): [description]

    Returns:
        [type]: [description]
    """
    vect = np.array([ned[0, 0], ned[1, 0], -ned[2, 0]]).reshape(3, 1)
    xyz = ned2ecef_mat @ vect
    xyz += init_ecef

    k_semimajor_axis = 6378137
    k_semiminor_axis = 6356752.3142
    first_ecc_sq = 0.00669437999014
    second_ecc_sq = 0.00673949674228

    k_r = np.sqrt(((xyz[0, 0])**2) + ((xyz[1, 0])**2))
    k_esq = k_semimajor_axis**2 - k_semiminor_axis**2
    k_f = 54. * k_semiminor_axis**2 * xyz[2, 0]**2
    k_g = (k_r**2
           + (1. - first_ecc_sq) * xyz[2, 0]**2
           - first_ecc_sq * k_esq)
    k_c = (first_ecc_sq**2 * k_f * k_r**2) / (k_g**3)
    k_s = (1 + k_c + np.sqrt(k_c**2 + 2.*k_c))**(1./3.)
    k_p = k_f / (3. * ((k_s + 1./k_s + 1.)**2) * k_g**2)
    k_q = np.sqrt(1. + 2.*first_ecc_sq**2 * k_p)
    r_0 = (- (k_p*k_r*first_ecc_sq) / (1.+k_q)
           + np.sqrt(0.5*(1.+1./k_q)*(k_semimajor_axis**2)
                     - k_p * (1.-first_ecc_sq) * xyz[2, 0]**2 / (k_q*(1.+k_q))
                     - 0.5*k_p*(k_r**2)))
    k_u = np.sqrt(((k_r-first_ecc_sq*r_0)**2) + xyz[2, 0]**2)
    k_v = np.sqrt(((k_r-first_ecc_sq*r_0)**2)
                  + (1.-first_ecc_sq) * xyz[2, 0]**2)
    z_0 = k_semiminor_axis**2 * xyz[2, 0] / (k_semimajor_axis * k_v)
    lat = np.degrees(np.arctan((xyz[2, 0] + second_ecc_sq*z_0)/k_r))
    lon = np.degrees(np.arctan2(xyz[1, 0], xyz[0, 0]))
    alt = k_u * (1. - (k_semiminor_axis**2)/(k_semimajor_axis*k_v))

    return np.array([[lat], [lon], [alt]])


def comp_ecef_to_ned_mat(lat, lon):
    """Computes the ECEF coordinates to local NED matrix.

    Args:
        lat (float): Latitude in [rad]
        lon (float): Longitude in [rad]

    Returns:
        np.ndarray((3, 3), dtype='float64'): ECEF to local NED matrix
    """
    s_lat = np.sin(lat)
    s_lon = np.sin(lon)
    c_lat = np.cos(lat)
    c_lon = np.cos(lon)

    ret = np.zeros((3, 3), dtype=np.float)
    ret[0, 0] = -s_lat * c_lon
    ret[0, 1] = -s_lat * s_lon
    ret[0, 2] = c_lat
    ret[1, 0] = -s_lon
    ret[1, 1] = c_lon
    ret[1, 2] = 0.
    ret[2, 0] = c_lat * c_lon
    ret[2, 1] = c_lat * s_lon
    ret[2, 2] = s_lat

    return ret


# GEOMETRIC FUNCTIONS (EULER, DCM, QUAT, ETC).


def comp_rot_mat_from_rpy(roll, pitch, yaw):
    """Computes rotation matrix from roll, pitch and yaw values

    Args:
        roll (float): roll in [rad]
        pitch (float): pitch in [rad]
        yaw (float): yaw in [rad]

    Returns:
        np.ndarray((3, 3), dtype='float64'): rotation matrix
    """
    return np.array([[np.cos(pitch)*np.cos(yaw),
                      (np.sin(roll)*np.sin(pitch)*np.cos(yaw)
                       - np.sin(yaw)*np.cos(roll)),
                      (np.sin(roll)*np.sin(yaw)
                       + np.sin(pitch)*np.cos(roll)*np.cos(yaw))],
                     [np.sin(yaw)*np.cos(pitch),
                      (np.sin(roll)*np.sin(pitch)*np.sin(yaw)
                       + np.cos(roll)*np.cos(yaw)),
                      (-np.sin(roll)*np.cos(yaw)
                       + np.sin(pitch)*np.sin(yaw)*np.cos(roll))],
                     [-np.sin(pitch),
                      np.sin(roll)*np.cos(pitch),
                      np.cos(roll)*np.cos(pitch)]])


def comp_quat_from_rot_mat(rot):
    """Computes quaternion from rotation matrix.

    Args:
        rot (np.ndarray((3, 3), dtype='float64')): rotation matrix

    Returns:
        np.ndarray((4, 1), dtype='float64'): quaternion
    """
    mat_trace = rot[0, 0]+rot[1, 1]+rot[2, 2]

    if mat_trace > 0:
        val = np.sqrt(mat_trace+1.)*2
        q_w = 0.25 * val
        q_x = (rot[2, 1] - rot[1, 2]) / val
        q_y = (rot[0, 2] - rot[2, 0]) / val
        q_z = (rot[1, 0] - rot[0, 1]) / val

    elif ((rot[0, 0] > rot[1, 1]) and (rot[0, 0] > rot[2, 2])):
        val = np.sqrt(1. + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        q_w = (rot[2, 1] - rot[1, 2]) / val
        q_x = 0.25 * val
        q_y = (rot[0, 1] + rot[1, 0]) / val
        q_z = (rot[0, 2] + rot[2, 0]) / val

    elif (rot[1, 1] > rot[2, 2]):
        val = np.sqrt(1. + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        q_w = (rot[0, 2] - rot[2, 0]) / val
        q_x = (rot[0, 1] + rot[1, 0]) / val
        q_y = 0.25 * val
        q_z = (rot[1, 2] + rot[2, 1])

    else:
        val = np.sqrt(1. + rot[2, 2] - rot[0, 0] - rot[1, 1])
        q_w = (rot[1, 0] - rot[0, 1]) / val
        q_x = (rot[0, 2] + rot[2, 0]) / val
        q_y = (rot[1, 2] + rot[2, 1]) / val
        q_z = 0.25 * val

    quat = np.array([[q_w], [q_x], [q_y], [q_z]])
    quat /= np.linalg.norm(quat)
    return quat


def comp_rot_mat_from_quat(quat):
    """Computes rotation matrix from quaternion (rotation matrix from body to
       inertial frames as it is using the fusionQPose from RTIMU).

    Args:
        quat (): quaternion

    Returns:
        np.ndarray((3, 3), dtype='float64'): rotation matrix
    """
    q00 = quat[0]**2
    q11 = quat[1]**2
    q22 = quat[2]**2
    q33 = quat[3]**2

    q01 = quat[0]*quat[1]
    q02 = quat[0]*quat[2]
    q03 = quat[0]*quat[3]

    q12 = quat[1]*quat[2]
    q13 = quat[1]*quat[3]

    q23 = quat[2]*quat[3]

    rot_bn = np.zeros((3, 3), dtype=np.float)

    rot_bn[0, 0] = q00 + q11 - q22 - q33
    rot_bn[0, 1] = 2 * (q12 - q03)
    rot_bn[0, 2] = 2 * (q13 + q02)

    rot_bn[1, 0] = 2 * (q12 + q03)
    rot_bn[1, 1] = q00 - q11 + q22 - q33
    rot_bn[1, 2] = 2 * (q23 - q01)

    rot_bn[2, 0] = 2 * (q13 - q02)
    rot_bn[2, 1] = 2 * (q23 + q01)
    rot_bn[2, 2] = q00 - q11 - q22 + q33

    return rot_bn


def comp_rpy_from_quat(quat):
    """Computes roll, pitch and yaw values in [rad] from quaternion.

    Args:
        quat (): quaternion

    Returns:
        TBD : TBD
    """
    return comp_rpy_from_rot_mat(comp_rot_mat_from_quat(quat))


def comp_rpy_diff(ea_after, ea_before):
    """Computes the roll, pitch yaw sequence to go from one set of Euler
       angles to another.

    Args:
        ea_after ([type]): [description]
        ea_before ([type]): [description]

    Returns:
        [type]: [description]
    """
    return comp_rpy_from_rot_mat(comp_rot_mat_from_rpy(ea_after[0, 0],
                                                       ea_after[1, 0],
                                                       ea_after[2, 0]).T
                                 @ comp_rot_mat_from_rpy(ea_before[0, 0],
                                                         ea_before[1, 0],
                                                         ea_before[2, 0]))


def comp_rpy_from_rot_mat(rot):
    """Computes roll, pitch and yaw values in [rad] from rotation matrix.

    Args:
        rot ([type]): rotation matrix

    Returns:
        np.ndarray((3, 1), dtype='float64'): roll, pitch and yaw angles [rad]
    """
    roll = np.arctan2(rot[1, 2], rot[2, 2])
    pitch = -np.arcsin(rot[0, 2])
    yaw = np.arctan2(rot[0, 1], rot[0, 0])
    return np.array([[roll], [pitch], [yaw]])


def comp_quat_from_rpy(ori):
    """Computes quaternion from roll, pitch and yaw values in [rad].

    Args:
        ori (dict): orientation dictionary with keys roll, pitch and yaw

    Returns:
        [type]: [description]
    """
    return comp_quat_from_rot_mat(comp_rot_mat_from_rpy(ori['roll'],
                                                        ori['pitch'],
                                                        ori['yaw']))


# 3D DYNAMICS FUNCTIONS.


def skew_symmetric(vect):
    """Returns the skew symmetric form of a three-dimensional vector

    Args:
        vect (np.ndarray((3, 1), dtype='float64')): vector of dimension 3

    Returns:
        np.ndarray((3, 3), dtype='float64'): skew symmetric matrix
    """
    return np.array([[0, -vect[2, 0], vect[1, 0]],
                     [vect[2, 0], 0, -vect[0, 0]],
                     [-vect[1, 0], vect[0, 0], 0]])


def get_omega_matrix(vect):
    """Computes 4x4 matrix form of gyroscope information.

    Args:
        vect ([type]): [description]

    Returns:
        [type]: [description]
    """
    # To compute matrix form of gyroscope info
    return np.array([[0, -vect[0, 0], -vect[1, 0], -vect[2, 0]],
                     [vect[0, 0], 0, vect[2, 0], -vect[1, 0]],
                     [vect[1, 0], -vect[2, 0], 0, vect[0, 0]],
                     [vect[2, 0], vect[1, 0], -vect[0, 0], 0]])


def comp_rpy_derivative_with_quaternion(quat):
    """Computes roll, pitch and yaw derivatives in [rad/s] from quaternion.

    Args:
        quat (np.ndarray((4, 1), dtype='float64')): quaternion

    Returns:
        np.ndarray((4, 3), dtype='float64'): matrix describing RPY derivatives
    """
    q_0 = quat[0, 0]
    q_1 = quat[1, 0]
    q_2 = quat[2, 0]
    q_3 = quat[3, 0]

    q00 = q_0*q_0
    q01 = q_0*q_1
    q02 = q_0*q_2
    q03 = q_0*q_3

    q11 = q_1*q_1
    q12 = q_1*q_2
    q13 = q_1*q_3

    q22 = q_2*q_2
    q23 = q_2*q_3

    q33 = q_3*q_3

    kr_a = q00 - q11 - q22 + q33
    kr_b = q01 + q23
    kr_c = 2./((kr_a**2) + 4*(kr_b**2))
    drdw = kr_c*(q_1*kr_a - 2*q_0*kr_b)
    drdx = kr_c*(q_0*kr_a + 2*q_1*kr_b)
    drdy = kr_c*(q_3*kr_a + 2*q_2*kr_b)
    drdz = kr_c*(q_2*kr_a - 2*q_3*kr_b)

    kp_a = 2*np.sqrt(1 - 4*((q13-q02)**2))
    dpdw = q_2*kp_a
    dpdx = -q_3*kp_a
    dpdy = q_0*kp_a
    dpdz = -q_1*kp_a

    ky_a = q00 + q11 - q22 - q33
    ky_b = q03 + q12
    ky_c = 2./((ky_a**2) + 4*(ky_b**2))
    dydw = ky_c*(q_3*ky_a - 2*q_0*ky_b)
    dydx = ky_c*(q_2*ky_a - 2*q_1*ky_b)
    dydy = ky_c*(q_1*ky_a + 2*q_2*ky_b)
    dydz = ky_c*(q_0*ky_a + 2*q_3*ky_b)

    return np.array([[drdw, drdx, drdy, drdz],
                     [dpdw, dpdx, dpdy, dpdz],
                     [dydw, dydx, dydy, dydz]])


# CLASSES.


class Quaternion():
    """Class for all things quaternion-related.

    Raises:
        AttributeError: if user tries to declare quaternion from multiple args
        ValueError: if user does not provide 4-dimensional column quaternion

    Returns:
        NoneType: returns nothing upon __init__ termination
    """

    def __init__(self, q_w=1., q_x=0., q_y=0., q_z=0., axis_angle=None,
                 euler=None):
        if axis_angle is None and euler is None:
            self.q_w = q_w
            self.q_x = q_x
            self.q_y = q_y
            self.q_z = q_z

        elif euler is not None and axis_angle is not None:
            raise AttributeError("Give either axis_angle or euler, not both!")

        elif axis_angle is not None:
            if ((not isinstance(axis_angle, (list, np.ndarray)))
                    or len(axis_angle) != 3):
                raise ValueError("axis_angle == (list or np.ndarray of len 3)")

            axis_angle = np.array(axis_angle)
            norm = np.linalg.norm(axis_angle)
            self.q_w = np.cos(0.5*norm)

            if norm < 1e-50:
                self.q_x = 0
                self.q_y = 0
                self.q_z = 0

            else:
                imag = axis_angle / norm * np.sin(0.5*norm)
                self.q_x = imag[0].item()
                self.q_y = imag[1].item()
                self.q_z = imag[2].item()

        else:
            roll = euler[0]
            pitch = euler[1]
            yaw = euler[2]

            c_r = np.cos(0.5*roll)
            s_r = np.sin(0.5*roll)
            c_p = np.cos(0.5*pitch)
            s_p = np.sin(0.5*pitch)
            c_y = np.cos(0.5*yaw)
            s_y = np.sin(0.5*yaw)

            self.q_w = c_r * c_p * c_y + s_r * s_p * s_y
            self.q_x = s_r * c_p * c_y - c_r * s_p * s_y
            self.q_y = c_r * s_p * c_y + s_r * c_p * s_y
            self.q_z = c_r * c_p * s_y - s_r * s_p * c_y

    def __repr__(self):
        return "Quaternion (wxyz): [{}, {}, {}, {}]".format(self.q_w, self.q_x,
                                                            self.q_y, self.q_z)

    def to_mat(self):
        """Converts quaternion to matrix representation

        Returns:
            [type]: [description]
        """
        vect = np.array([self.q_x, self.q_y, self.q_z]).reshape(3, 1)
        return ((self.q_w**2 - np.dot(vect.T, vect)) * np.eye(3)
                + 2*np.dot(vect, vect.T)
                + 2*self.q_w*skew_symmetric(vect))

    def to_euler(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        roll = np.arctan2(2 * (self.q_w * self.q_x + self.q_y * self.q_z),
                          1 - 2 * (self.q_x**2 + self.q_y**2))
        pitch = np.arcsin(2 * (self.q_w * self.q_y - self.q_z * self.q_x))
        yaw = np.arctan2(2 * (self.q_w * self.q_z + self.q_x * self.q_y),
                         1 - 2 * (self.q_y**2 + self.q_z**2))
        return np.array([roll, pitch, yaw])

    def to_numpy(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return np.array([self.q_w, self.q_x, self.q_y, self.q_z])

    def normalize(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        norm_inv = 1./np.linalg.norm([self.q_w, self.q_x, self.q_y, self.q_z])
        return Quaternion(self.q_w*norm_inv, self.q_x*norm_inv,
                          self.q_y*norm_inv, self.q_z*norm_inv)

    def quat_mult(self, quat, out='np'):
        """[summary]

        Args:
            quat ([type]): [description]
            out (str, optional): [description]. Defaults to 'np'.

        Returns:
            [type]: [description]
        """
        vect = np.array([self.q_x, self.q_y, self.q_z]).reshape(3, 1)
        sum_term = np.zeros([4, 4])
        sum_term[0, 1:] = -vect[:, 0]
        sum_term[1:, 0] = vect[:, 0]
        sum_term[1:, 1:] = -skew_symmetric(vect)
        sigma = self.q_w * np.eye(4) + sum_term

        if type(quat).__name__ == "Quaternion":
            quat_np = np.dot(sigma, quat.to_numpy())
        else:
            quat_np = np.dot(sigma, quat)

        if out == 'np':
            return quat_np
        elif out == 'Quaternion':
            quat_obj = Quaternion(quat_np[0], quat_np[1],
                                  quat_np[2], quat_np[3])
            return quat_obj


class GNSSaidedINSwithEKF():
    """Class for GNSS-aided INS.
    """
    def __init__(self, dt, imu_data):
        self.kf_dt = dt  # Default timestep length in [s]

        # Reference location parameters to be the origin of NED frame
        self.crane_center_lat = np.radians(46.51197)  # 46.513250)  # [rad]
        self.crane_center_lon = np.radians(6.624637)  # 6.546444)  # [rad]
        self.crane_center_alt = 404  # 435.  # [m]

        # Compute base world coordinate matrices
        self.init_ecef = geodetic2ecef(self.crane_center_lat,
                                       self.crane_center_lon,
                                       self.crane_center_alt)
        phi_p = np.arctan2(self.init_ecef[2],
                           np.sqrt(((self.init_ecef[0])**2)
                                   + ((self.init_ecef[1])**2)))
        self.ecef2ned_matrix = comp_ecef_to_ned_mat(phi_p,
                                                    self.crane_center_lon)
        self.ned2ecef_matrix = self.ecef2ned_matrix.T

        # vvv FOR VERIFICATION / DEBUGGING vvv
        # or with 46.513519, 6.545234, 30.
        self.ned_pos = ecef2ned(np.radians(46.51187), np.radians(6.624647),
                                30., self.init_ecef, self.ecef2ned_matrix)
        self.ned_vel = np.zeros((3, 1), dtype=np.float)
        self.lla_pos = ned2geodetic(self.ned_pos, self.init_ecef,
                                    self.ned2ecef_matrix)
        # ^^^ FOR VERIFICATION / DEBUGGING ^^^

        # Initial IMU values
        self.cur_imu_data = imu_data
        cur_quat = np.array(imu_data['fusionQPose']).reshape(4, 1)
        # delete here? acc_B = np.array(imu_data['accel']).reshape(3, 1)
        # delete here? gyr_B = np.array(imu_data['gyro']).reshape(3, 1)
        self.rotmat_bn = comp_rot_mat_from_quat(cur_quat)

        # IMU noise specifications (acc at 8[g]:.09[g],
        # gyr at 2000[dps]:30[dps], mag at 4[G]:1[G])
        self.sigma_acc = 0.01125
        self.sigma_gyr = 0.0003

        self.prev_q_est = Quaternion()

        self.p_est = np.array(self.ned_pos)
        self.v_est = np.array(self.ned_vel)
        self.q_est = Quaternion().to_numpy()
        self.p_cov = np.eye(9)

        self.gravity = np.array([[0.], [0.], [-9.81]])
        self.l_jac = np.zeros([9, 6])
        self.l_jac[3:, :] = np.eye(6)
        self.h_jac = np.zeros([3, 9])
        self.h_jac[:, :3] = np.eye(3)

    def measurement_update(self, sensor_var, y_k, kf_log_file):
        """[summary]

        Args:
            sensor_var ([type]): [description]
            y_k ([type]): [description]
            kf_log_file ([type]): [description]
        """
        mat_s = (self.h_jac @ self.p_cov @ self.h_jac.T
                 + np.diag([sensor_var, sensor_var, sensor_var]))
        mat_s_inv = np.linalg.inv(mat_s)
        mat_k = self.p_cov @ self.h_jac.T @ mat_s_inv

        kf_log_file.write('Observation:    {}\n'.format(y_k.T))
        kf_log_file.write('Prior estimate: {}\n'.format(self.p_est.T))
        delta_x = (mat_k @ (y_k-self.p_est).reshape((3, 1))).reshape(9, 1)
        kf_log_file.write('Difference:     {}\n'.format(delta_x.T))
        self.prev_q_est = np.copy(self.q_est)

        self.p_est += delta_x[0:3]
        self.v_est += delta_x[3:6]
        self.q_est = Quaternion(*self.q_est).quat_mult(
            Quaternion(euler=delta_x[6:]))

        self.p_cov = (np.eye(9) - mat_k@self.h_jac) @ self.p_cov

    def predict(self):
        """[summary]
        """
        # Update variable shortcuts (right before calling this function,
        # the cur_imu_data field is updated)
        acc_body = np.array(self.cur_imu_data['accel']).reshape(3, 1)*9.81
        gyr_body = np.array(self.cur_imu_data['gyro']).reshape(3, 1)

        if True in np.isnan(self.p_est):
            self.p_est = np.array(self.ned_pos)
        if True in np.isnan(self.v_est):
            self.v_est = np.array(self.ned_vel)
        if True in np.isnan(self.q_est):
            self.q_est = Quaternion().to_numpy()

        self.prev_q_est = np.copy(self.q_est)
        prev_acc = ((Quaternion(*self.q_est.reshape(4)).to_mat() @ acc_body)
                    - self.gravity)

        self.p_est += self.kf_dt * self.v_est + 0.5*(self.kf_dt**2)*prev_acc
        self.v_est += self.kf_dt * prev_acc
        self.q_est = Quaternion(*self.q_est).quat_mult(
            Quaternion(axis_angle=(gyr_body*self.kf_dt)))

        mat_f = np.eye(9)
        mat_f[0:3, 3:6] = self.kf_dt * np.eye(3)
        mat_f[3:6, 6:] = -self.kf_dt * skew_symmetric(
            (Quaternion(*self.prev_q_est).to_mat() @ acc_body))
        mat_q = (self.kf_dt**2) * np.diag([self.sigma_acc, self.sigma_acc,
                                           self.sigma_acc, self.sigma_gyr,
                                           self.sigma_gyr, self.sigma_gyr])
        self.p_cov = (mat_f @ self.p_cov @ mat_f.T
                      + self.l_jac @ mat_q @ self.l_jac.T)


# INITIALIZATION AND MAIN FUNCTION.


def init_sixfab_cellulariot():
    """[summary]

    Returns:
        [type]: [description]
    """
    node = cellulariot.CellularIoT()
    node.setupGPIO()

    print("Power up sequence - disabling first")
    node.disable()
    print("Disable done\n")
    time.sleep(1)
    print("Starting enable")
    node.enable()
    print("Enable done\n")
    time.sleep(1)
    print("Starting power up")
    node.powerUp()
    print("Power up done\n")

    time.sleep(0.5)
    node.sendATComm("ATE0", "OK")
    time.sleep(0.5)
    node.sendATComm("AT+CMEE=2", "OK")
    time.sleep(0.5)

    print("Turning GNSS on")
    node.turnOnGNSS()
    time.sleep(1)

    node.sendATComm("AT+QGPSCFG=\"galileonmeatype\",1", "OK")
    time.sleep(0.5)
    node.sendATComm("AT+QGPSCFG=\"nmeasrc\",1", "OK")
    time.sleep(0.5)

    return node


def init_imu_breakout():
    """[summary]

    Returns:
        [type]: [description]
    """
    sys.path.append('.')
    rtimu_settings_file = "myRTIMULib"
    print("Using settings file " + rtimu_settings_file + ".ini")
    if not os.path.exists("./myRTIMULib.ini"):
        print("Settings file does not exist, but will be created")

    imu_settings = RTIMU.Settings(rtimu_settings_file)
    my_imu = RTIMU.RTIMU(imu_settings)

    print("IMU Name: " + my_imu.IMUName())

    if not my_imu.IMUInit():
        print("IMU Init Failed")
        sys.exit(1)
    else:
        print("IMU Init Succeeded")

    my_imu.setSlerpPower(0.02)
    my_imu.setGyroEnable(True)
    my_imu.setAccelEnable(True)
    my_imu.setCompassEnable(True)

    return my_imu


def main():
    """[summary]
    """
    log_file = open('myEKFlogFile.txt', "a+")

    sixfab = init_sixfab_cellulariot()

    try:
        imu = init_imu_breakout()
        poll_interval = 0.001*imu.IMUGetPollInterval()
        print("Recommended poll interval: {}[s]".format(poll_interval))

        t_imu = 10.*poll_interval
        # delete here? fIMU = 1./t_imu
        f_gps = 1.
        t_gps = 1./f_gps

        ekf = GNSSaidedINSwithEKF(t_imu, imu.getIMUData())

        print("Back in main, EKF has been initialized")
        print(imu.IMURead())
        print(imu.getIMUData())

        start_time = time.perf_counter()
        time_gps = start_time - t_gps
        time_imu = start_time - t_imu

        first_flag = True
        read_flag = False

        while True:
            # delete here? prevTimeGPS = time_gps
            time_gps = time.perf_counter()
            # delete here? dtGPS = time_gps - prevTimeGPS

            while True:
                print("In inner loop")
                print(imu.IMURead())
                print(imu.getIMUData())
                prev_time_imu = time_imu
                time_imu = time.perf_counter()
                dt_imu = time_imu - prev_time_imu
                ekf.kf_dt = dt_imu

                if imu.IMURead():
                    read_flag = True
                    ekf.cur_imu_data = imu.getIMUData()
                    if first_flag:
                        print("Initial quaternion set with EA, hope for the "
                              "best - {}".format(
                                  ekf.cur_imu_data['fusionQPose']))
                        ekf.prev_q_est = Quaternion(
                            *ekf.cur_imu_data['fusionQPose']).to_numpy()
                        first_flag = False

                    # q = np.array(ekf.cur_imu_data['fusionQPose']).reshape(
                    #    (4,1))
                    # ekf.rotmat_bn = comp_rot_mat_from_quat(q)
                    # /!\ OUTPUT ([0;0;1][G]) ABOVE PROVES THAT IT IS THE
                    # CORRECT TRANSFORMATION TO GET LEVELED DATA
                    # ekf.acc_B = np.array(ekf.cur_imu_data['accel']).reshape(
                    #    (3,1))
                    # ekf.gyr_B = np.array(ekf.cur_imu_data['gyro']).reshape(
                    #    (3,1))

                    ekf.predict()
                    # sys.stdout.write('\rPred. at {} is x:{}'.format(
                    #   time.ctime(0.000001*ekf.cur_imu_data['timestamp']),
                    #   ekf.x.T))
                    # sys.stdout.flush()

                else:
                    read_flag = False
                    print("DIDNT READ IMU")

                while True:
                    if time.perf_counter() > (time_imu + t_imu):
                        break

                if time.perf_counter() > (time_gps + t_gps):
                    if read_flag and ekf.cur_imu_data['fusionQPoseValid']:
                        # vvv FOR VERIFICATION / DEBUGGING vvv
                        d_gga = sixfab.getNMEAGGA()
                        if d_gga == {}:
                            print("No fix, using defaults "
                                  "(46.51197N, 6.624637E, 404[m])")
                            d_gga = {'lat': 46.51197,
                                     'lon': 6.624637,
                                     'alt': 404.}
                        print(d_gga)
                        ekf.ned_pos = ecef2ned(np.radians(d_gga['lat']),
                                               np.radians(d_gga['lon']),
                                               d_gga['alt'],
                                               ekf.init_ecef,
                                               ekf.ecef2ned_matrix)
                        obs_z = np.array(ekf.ned_pos)
                        # or ned_pos with 46.513519, 6.545234, 398
                        # then obs_z = np.array(ekf.ned_pos)
                        # ^^^ FOR VERIFICATION / DEBUGGING ^^^
                        ekf.measurement_update(0.01, obs_z, log_file)

                        print("State update - saving to log file")
                        timedata = 0.000001*ekf.cur_imu_data['timestamp']
                        log_file.write("Exit GPS on {} - p_est:{} - v_est:{}"
                                       " - q_est:{}\n\n".format(
                                           time.ctime(timedata), ekf.p_est.T,
                                           ekf.v_est.T, ekf.q_est.T))
                    break

    except(KeyboardInterrupt, SystemExit):
        log_file.close()
        print("Stopping GNSS...")
        sixfab.turnOffGNSS()
        print("Done. Bye.\n")

    print("Done.\nExiting now.")


if __name__ == '__main__':
    main()
