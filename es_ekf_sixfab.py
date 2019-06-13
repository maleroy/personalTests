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

import numpy as np
import os.path
import RTIMU
import time
import sys
import os

from cellulariot import cellulariot

np.set_printoptions(precision=4, suppress=True, sign='+', linewidth=204)

""" ------------------------------------------------- GPS CONVERSION FUNCTIONS.
"""
def geodetic2ecef(lat, lon, alt):
    """Converts Geodetic coordinates to ECEF on Earth.

    Args:
        lat (float): Latitude in [rad]
        lon (float): Longitude in [rad]
        alt (float): Altitude in [m]

    Returns:
        np.ndarray((3, 1), dtype='float64'): ECEF coordinates
    """
    xi = 1./np.sqrt(1. - 0.00669437999014 * np.sin(lat) * np.sin(lat))

    ecef = np.zeros((3, 1), dtype=np.float)

    ecef[0, 0] = (6378137*xi + alt) * np.cos(lat) * np.cos(lon)
    ecef[1, 0] = (6378137*xi + alt) * np.cos(lat) * np.sin(lon)
    ecef[2, 0] = (6378137*xi * (1. - 0.00669437999014) + alt) * np.sin(lat)

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

    v = np.zeros((3, 1), dtype=np.float)
    v[0, 0] = xyz[0, 0] - init_ecef[0, 0]
    v[1, 0] = xyz[1, 0] - init_ecef[1, 0]
    v[2, 0] = xyz[2, 0] - init_ecef[2, 0]

    ret = ecef2ned_mat @ v
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
    v = np.array([ned[0, 0], ned[1, 0], -ned[2, 0]]).reshape(3, 1)
    xyz = ned2ecef_mat @ v
    xyz += init_ecef

    x = xyz[0, 0]
    y = xyz[1, 0]
    z = xyz[2, 0]

    k_semimajor_axis = 6378137
    k_semiminor_axis = 6356752.3142
    first_ecc_sq = 0.00669437999014
    second_ecc_sq = 0.00673949674228

    r = np.sqrt(((xyz[0, 0])**2) + ((xyz[1, 0])**2))
    k_esq = k_semimajor_axis**2 - k_semiminor_axis**2
    k_f = 54. * k_semiminor_axis**2 * xyz[2, 0]**2
    k_g = (r**2
           + (1. - first_ecc_sq) * xyz[2, 0]**2
           - first_ecc_sq * k_esq)
    k_c = (first_ecc_sq**2 * k_f * r**2) / (k_g**3)
    k_s = (1 + k_c + np.sqrt(k_c**2 + 2.*k_c))**(1./3.)
    k_p = k_f / (3. * ((k_s + 1./k_s + 1.)**2) * k_g**2)
    k_q = np.sqrt(1. + 2.*first_ecc_sq**2 * k_p)
    r_0 = -(k_p*r*first_ecc_sq) / (1.+k_q) + np.sqrt(0.5*(1.+1./k_q)*(k_semimajor_axis**2) - k_p * (1.-first_ecc_sq) * xyz[2, 0]**2 / (k_q*(1.+k_q)) - 0.5*k_p*(r**2))
    k_u = np.sqrt(((r-first_ecc_sq*r_0)**2) + xyz[2, 0]**2)
    k_v = np.sqrt(((r-first_ecc_sq*r_0)**2) + (1.-first_ecc_sq) * xyz[2, 0]**2)
    z_0 = k_semiminor_axis**2 * xyz[2, 0] / (k_semimajor_axis * k_v)
    lat = np.degrees(np.arctan((xyz[2, 0] + second_ecc_sq*z_0)/r))
    lon = np.degrees(np.arctan2(xyz[1, 0], xyz[0, 0]))
    alt = k_u * (1.- (k_semiminor_axis**2)/(k_semimajor_axis*k_v))

    return np.array([[lat], [lon], [alt]])

def nRe(lat, lon):
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


#######################################################################################################################
#######################################################################################################################
### GEOMETRIC FUNCTIONS (EULER, DCM, QUAT, ETC)                                                                     ###
#######################################################################################################################
#######################################################################################################################
def compRotMatFromRPY(r, p, y):
    """Computes rotation matrix from roll, pitch and yaw values
    
    Args:
        r (float): roll in [rad]
        p (float): pitch in [rad]
        y (float): yaw in [rad]
    
    Returns:
        np.ndarray((3, 3), dtype='float64'): Rotation matrix
    """
    return np.array([ [np.cos(p)*np.cos(y), np.sin(r)*np.sin(p)*np.cos(y) - np.sin(y)*np.cos(r),  np.sin(r)*np.sin(y) + np.sin(p)*np.cos(r)*np.cos(y)],
                      [np.sin(y)*np.cos(p), np.sin(r)*np.sin(p)*np.sin(y) + np.cos(r)*np.cos(y), -np.sin(r)*np.cos(y) + np.sin(p)*np.sin(y)*np.cos(r)],
                      [         -np.sin(p),                                 np.sin(r)*np.cos(p),                                  np.cos(r)*np.cos(p)] ])

def compQuatFromRotMat(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    
    if ( tr>0 ):
        S  = np.sqrt(tr+1.)*2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
        
    elif ( (R[0,0]>R[1,1]) and (R[0,0]>R[2,2]) ):
        S  = np.sqrt(1. + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
        
    elif ( R[1,1]>R[2,2] ):
        S  = np.sqrt(1. + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1])
        
    else:
        S  = np.sqrt(1. + R[2,2] - R[0,0] - R[1,1])
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
      
    q  = np.array([[qw],[qx],[qy],[qz]])
    q /= np.linalg.norm(q)
    return q

def compRotMatFromQuat(q):
    """ COMPUTES R_BtoN AS IT IS USING THE fusionQPose FROM RTIMU """
    q00 = q[0]**2
    q11 = q[1]**2
    q22 = q[2]**2
    q33 = q[3]**2

    q01 = q[0]*q[1]
    q02 = q[0]*q[2]
    q03 = q[0]*q[3]

    q12 = q[1]*q[2]
    q13 = q[1]*q[3]

    q23 = q[2]*q[3]

    R_BtoN = np.zeros((3,3), dtype=np.float)

    R_BtoN[0,0] = q00 + q11 - q22 - q33
    R_BtoN[0,1] = 2 * (q12 - q03) 
    R_BtoN[0,2] = 2 * (q13 + q02)

    R_BtoN[1,0] = 2 * (q12 + q03)
    R_BtoN[1,1] = q00 - q11 + q22 - q33 
    R_BtoN[1,2] = 2 * (q23 - q01)

    R_BtoN[2,0] = 2 * (q13 - q02)
    R_BtoN[2,1] = 2 * (q23 + q01) 
    R_BtoN[2,2] = q00 - q11 - q22 + q33

    return R_BtoN

def compRPYfromQuat(q):
    return compRPYfromRotMat(compRotMatFromQuat(q))

def compRPYdiff(eaAfter, eaBefore):
    return compRPYfromRotMat( compRotMatFromRPY(eaAfter[0,0],eaAfter[1,0],eaAfter[2,0]).T @ compRotMatFromRPY(eaBefore[0,0],eaBefore[1,0],eaBefore[2,0]) )

def compRPYfromRotMat(R):
    r =  np.arctan2(R[1,2],R[2,2])
    p = -np.arcsin(R[0,2])
    y =  np.arctan2(R[0,1],R[0,0])
    return np.array([[r],[p],[y]])

def compQuatFromRPY(ori):
    return compQuatFromRotMat( compRotMatFromRPY(ori['roll'], ori['pitch'], ori['yaw']) )


#######################################################################################################################
#######################################################################################################################
### 3D DYNAMICS FUNCTIONS                                                                                           ###
#######################################################################################################################
#######################################################################################################################

def skew_symmetric(v):
    return np.array([ [      0 , -v[2,0],  v[1,0] ],
                      [  v[2,0],      0 , -v[0,0] ],
                      [ -v[1,0],  v[0,0],      0  ] ])

def getOmegaMatrix(v):
    """ To compute matrix form of gyroscope info """
    return np.array([ [ 0     ,-v[0,0],-v[1,0],-v[2,0] ],
                      [ v[0,0], 0     , v[2,0],-v[1,0] ],
                      [ v[1,0],-v[2,0], 0     , v[0,0] ],
                      [ v[2,0], v[1,0],-v[0,0], 0      ] ])

def compRPYderivativeWithQuaternion(q):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    
    q00 = q0*q0
    q01 = q0*q1
    q02 = q0*q2
    q03 = q0*q3

    q11 = q1*q1
    q12 = q1*q2
    q13 = q1*q3

    q22 = q2*q2
    q23 = q2*q3

    q33 = q3*q3

    rollMultA    = q00 - q11 - q22 + q33
    rollMultB    = q01 + q23
    rollDenom    = 2./((rollMultA**2) + 4*(rollMultB**2))
    rollDerivByW = rollDenom*( q1*rollMultA - 2*q0*rollMultB )
    rollDerivByX = rollDenom*( q0*rollMultA + 2*q1*rollMultB )
    rollDerivByY = rollDenom*( q3*rollMultA + 2*q2*rollMultB )
    rollDerivByZ = rollDenom*( q2*rollMultA - 2*q3*rollMultB )

    pitchMult     = 2*np.sqrt( 1 - 4*((q13-q02)**2) )
    pitchDerivByW =  q2*pitchMult
    pitchDerivByX = -q3*pitchMult
    pitchDerivByY =  q0*pitchMult
    pitchDerivByZ = -q1*pitchMult

    yawMultA    = q00 + q11 - q22 - q33
    yawMultB    = q03 + q12
    yawDenom    = 2./((yawMultA**2) + 4*(yawMultB**2))
    yawDerivByW = yawDenom*( q3*yawMultA - 2*q0*yawMultB )
    yawDerivByX = yawDenom*( q2*yawMultA - 2*q1*yawMultB )
    yawDerivByY = yawDenom*( q1*yawMultA + 2*q2*yawMultB )
    yawDerivByZ = yawDenom*( q0*yawMultA + 2*q3*yawMultB )

    return np.array([ [ rollDerivByW,  rollDerivByX,  rollDerivByY,  rollDerivByZ],
                      [pitchDerivByW, pitchDerivByX, pitchDerivByY, pitchDerivByZ],
                      [  yawDerivByW,   yawDerivByX,   yawDerivByY,   yawDerivByZ] ])


#######################################################################################################################
#######################################################################################################################
### CLASSES                                                                                                         ###
#######################################################################################################################
#######################################################################################################################

class Quaternion():
    """ Class for all things quaternion-related """

    def __init__(self, w=1., x=0., y=0., z=0., axis_angle=None, euler=None):
        if axis_angle is None and euler is None:
            self.w = w
            self.x = x
            self.y = y
            self.z = z

        elif euler is not None and axis_angle is not None:
            raise AttributeError("Only one of axis_angle and euler may be specified")

        elif axis_angle is not None:
            if not (type(axis_angle) == list or type(axis_angle) == np.ndarray) or len(axis_angle) != 3:
                raise ValueError("axis_angle must be a list or an np.ndarray of length 3")
            
            axis_angle = np.array(      axis_angle)
            norm       = np.linalg.norm(axis_angle)
            self.w = np.cos(0.5*norm)
            
            if norm < 1e-50:
                self.x = 0
                self.y = 0
                self.z = 0
            
            else:
                imag = axis_angle / norm * np.sin(0.5*norm)
                self.x = imag[0].item()
                self.y = imag[1].item()
                self.z = imag[2].item()

        else:
            roll  = euler[0]
            pitch = euler[1]
            yaw   = euler[2]

            cr = np.cos(0.5*roll)
            sr = np.sin(0.5*roll)
            cp = np.cos(0.5*pitch)
            sp = np.sin(0.5*pitch)
            cy = np.cos(0.5*yaw)
            sy = np.sin(0.5*yaw)

            self.w = cr * cp * cy + sr * sp * sy
            self.x = sr * cp * cy - cr * sp * sy
            self.y = cr * sp * cy + sr * cp * sy
            self.z = cr * cp * sy - sr * sp * cy

    def __repr__(self):
        return "Quaternion (wxyz): [{}, {}, {}, {}]".format(self.w, self.x, self.y, self.z)

    def to_mat(self):
        v = np.array([self.x, self.y, self.z]).reshape(3,1)
        return (self.w**2 - np.dot(v.T,v)) * np.eye(3) + 2*np.dot(v,v.T) + 2*self.w*skew_symmetric(v)
    
    def to_euler(self):
        roll  = np.arctan2(2 * (self.w * self.x + self.y * self.z), 1 - 2 * (self.x**2 + self.y**2))
        pitch = np.arcsin( 2 * (self.w * self.y - self.z * self.x))
        yaw   = np.arctan2(2 * (self.w * self.z + self.x * self.y), 1 - 2 * (self.y**2 + self.z**2))
        return np.array([roll, pitch, yaw])
    
    def to_numpy(self):
        return np.array([self.w, self.x, self.y, self.z])

    def normalize(self):
        normInv = 1./np.linalg.norm([self.w, self.x, self.y, self.z])
        return Quaternion(self.w*normInv, self.x*normInv, self.y*normInv, self.z*normInv)

    def quat_mult(self, q, out='np'):
        v = np.array([self.x, self.y, self.z]).reshape(3,1)
        sum_term = np.zeros([4,4])
        sum_term[0 ,1:] = -v[:,0]
        sum_term[1:,0 ] =  v[:,0]
        sum_term[1:,1:] = -skew_symmetric(v)
        sigma = self.w * np.eye(4) + sum_term

        if type(q).__name__ == "Quaternion":
            quat_np = np.dot(sigma, q.to_numpy())
        else:
            quat_np = np.dot(sigma, q)

        if out == 'np':
            return quat_np
        elif out == 'Quaternion':
            quat_obj = Quaternion(quat_np[0], quat_np[1], quat_np[2], quat_np[3])
            return quat_obj


class GNSSaidedINSwithEKF():
    """ Class for GNSS-aided INS """
    def __init__(self, dt, curIMUdata):
        self.dt = dt  # Default timestep length in [s]

        # Reference location parameters to be the origin of NED frame
        self.craneCenterLat = np.radians(46.51197)#46.513250)  # [rad]
        self.craneCenterLon = np.radians(6.624637)# 6.546444)  # [rad]
        self.craneCenterAlt = 404#435.                   # [m]

        # Compute base world coordinate matrices
        self.initEcef       = geodetic2ecef(self.craneCenterLat, self.craneCenterLon, self.craneCenterAlt)
        phiP                = np.arctan2(self.initEcef[2], np.sqrt(((self.initEcef[0])**2)+((self.initEcef[1])**2)))
        self.ecef2nedMatrix = nRe(phiP, self.craneCenterLon)
        self.ned2ecefMatrix = self.ecef2nedMatrix.T

        ############################################
        ### vvv FOR VERIFICATION / DEBUGGING vvv ###
        ############################################
        #elf.nedPos = ecef2ned(np.radians(46.513519), np.radians(6.545234), 30., self.initEcef, self.ecef2nedMatrix)
        self.nedPos = ecef2ned(np.radians(46.51187), np.radians(6.624647), 30., self.initEcef, self.ecef2nedMatrix)
        self.nedVel = np.zeros((3,1), dtype=np.float)
        self.llaPos = ned2geodetic(self.nedPos, self.initEcef, self.ned2ecefMatrix)
        ############################################
        ### ^^^ FOR VERIFICATION / DEBUGGING ^^^ ###
        ############################################

        # Initial IMU values
        self.curIMUdata = curIMUdata
        curQuat         = np.array(curIMUdata['fusionQPose']).reshape(4,1)
        acc_B           = np.array(curIMUdata['accel']      ).reshape(3,1)
        gyr_B           = np.array(curIMUdata['gyro']       ).reshape(3,1)
        self.R_BtoN     = compRotMatFromQuat(curQuat)

        # IMU noise specifications (acc at 8[g]:.09[g], gyr at 2000[dps]:30[dps], mag at 4[G]:1[G])
        self.sigmaAcc = 0.01125
        self.sigmaGyr = 0.0003

        self.p_est = np.array(self.nedPos)
        self.v_est = np.array(self.nedVel)
        self.q_est = Quaternion().to_numpy()
        self.p_cov = np.eye(9)

        self.gravity     = np.array([[0.],[0.],[-9.81]])
        self.l_jac       = np.zeros([9,6])
        self.l_jac[3:,:] = np.eye(6)
        self.h_jac       = np.zeros([3,9])
        self.h_jac[:,:3] = np.eye(3)

    def measurement_update(self, sensor_var, y_k, logFile):
        S    = self.h_jac @ self.p_cov @ self.h_jac.T + np.diag([sensor_var, sensor_var, sensor_var])
        Sinv = np.linalg.inv(S)
        K    = self.p_cov @ self.h_jac.T @ Sinv

        logFile.write('Observation:    {}\n'.format(y_k.T))
        logFile.write('Prior estimate: {}\n'.format(self.p_est.T))
        delta_x = (K @ (y_k-self.p_est).reshape((3,1))).reshape(9,1)
        logFile.write('Difference:     {}\n'.format(delta_x.T))
        self.prev_q_est = np.copy(self.q_est)
        
        self.p_est += delta_x[0:3]
        self.v_est += delta_x[3:6]
        self.q_est  = Quaternion(*self.q_est).quat_mult(Quaternion(euler=delta_x[6:]))

        self.p_cov = (np.eye(9) - K@self.h_jac) @ self.p_cov

    def predict(self):
        # Update variable shortcuts (right before calling this function, the curIMUdata field is updated)
        acc_B = np.array(self.curIMUdata['accel']).reshape(3,1)*9.81
        gyr_B = np.array(self.curIMUdata['gyro'] ).reshape(3,1)
        
        if True in np.isnan(self.p_est):
            self.p_est = np.array(self.nedPos)
        if True in np.isnan(self.v_est):
            self.v_est = np.array(self.nedVel)
        if True in np.isnan(self.q_est):
            self.q_est = Quaternion().to_numpy()
    
        self.prev_q_est = np.copy(self.q_est)
        prev_acc        = (Quaternion(*self.q_est.reshape(4)).to_mat() @ acc_B) - self.gravity

        self.p_est += self.dt * self.v_est + 0.5*(self.dt**2)*prev_acc
        self.v_est += self.dt * prev_acc
        self.q_est  = Quaternion(*self.q_est).quat_mult(Quaternion(axis_angle=(gyr_B*self.dt)))

        F          =  np.eye(9)
        F[0:3,3:6] =  self.dt * np.eye(3)
        F[3:6,6:]  = -self.dt * skew_symmetric( (Quaternion(*self.prev_q_est).to_mat() @ acc_B) )
        Q          = (self.dt**2) * np.diag([self.sigmaAcc, self.sigmaAcc, self.sigmaAcc, self.sigmaGyr, self.sigmaGyr, self.sigmaGyr])
        self.p_cov = F @ self.p_cov @ F.T + self.l_jac @ Q @ self.l_jac.T



#######################################################################################################################
#######################################################################################################################
### INITIALIZATION AND MAIN FUNCTION                                                                                ###
#######################################################################################################################
#######################################################################################################################

if __name__ == '__main__':
    logFile = open('myEKFlogFile.txt', "a+")

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
    node.sendATComm("ATE0","OK")
    time.sleep(0.5)
    node.sendATComm("AT+CMEE=2","OK")
    time.sleep(0.5)

    print("Turning GNSS on")
    node.turnOnGNSS()
    time.sleep(1)

    node.sendATComm("AT+QGPSCFG=\"galileonmeatype\",1", "OK")
    time.sleep(0.5)
    node.sendATComm("AT+QGPSCFG=\"nmeasrc\",1", "OK")
    time.sleep(0.5)

    try:
        sys.path.append('.')
        SETTINGS_FILE = "myRTIMULib"
        print("Using settings file " + SETTINGS_FILE + ".ini")
        if not os.path.exists("./myRTIMULib.ini"):
            print("Settings file does not exist, but will be created")

        imuSettings = RTIMU.Settings(SETTINGS_FILE)
        imu         = RTIMU.RTIMU(imuSettings)

        print("IMU Name: " + imu.IMUName())

        if (not imu.IMUInit()):
            print("IMU Init Failed")
            sys.exit(1)
        else:
            print("IMU Init Succeeded")

        imu.setSlerpPower(0.02)
        imu.setGyroEnable(True)
        imu.setAccelEnable(True)
        imu.setCompassEnable(True)

        poll_interval = 0.001*imu.IMUGetPollInterval()
        print("Recommended poll interval: {}[s]".format(poll_interval))
            
        tIMU = 10*poll_interval  
        fIMU = 1./tIMU
        fGPS = 1.
        tGPS = 1./fGPS

        ekf = GNSSaidedINSwithEKF(tIMU, imu.getIMUData())

        startTime = time.perf_counter()
        timeGPS   = startTime - tGPS
        timeIMU   = startTime - tIMU

        firstFlag = True
        readFlag  = False

        while True:
            prevTimeGPS = timeGPS
            timeGPS     = time.perf_counter()
            dtGPS       = timeGPS - prevTimeGPS

            while True:
                prevTimeIMU = timeIMU
                timeIMU     = time.perf_counter()
                dtIMU       = timeIMU - prevTimeIMU
                ekf.dt      = dtIMU

                if imu.IMURead():
                    readFlag = True
                    ekf.curIMUdata = imu.getIMUData()
                    if firstFlag:
                        print('Initial quaternion set with EA, hope for the best - {}'.format(ekf.curIMUdata['fusionQPose']))
                        ekf.prev_q_est = Quaternion(*ekf.curIMUdata['fusionQPose']).to_numpy()
                        firstFlag = False
                    '''
                    q              = np.array(ekf.curIMUdata['fusionQPose']).reshape((4,1)) 
                    ekf.R_BtoN     = compRotMatFromQuat(q)  # /!\ OUTPUT ([0;0;1][G]) PROVES THAT IT IS THE CORRECT TRANSFORMATION TO GET LEVELED DATA
                    ekf.acc_B = np.array(ekf.curIMUdata['accel']).reshape((3,1))
                    ekf.gyr_B = np.array(ekf.curIMUdata['gyro']).reshape((3,1))
                    '''
                    ekf.predict()
                    #ys.stdout.write('\rPred. at {} is x:{}'.format(time.ctime(0.000001*ekf.curIMUdata['timestamp']),ekf.x.T))
                    #ys.stdout.flush()

                else:
                    readFlag = False
                    print("DIDNT READ IMU")

                while True:
                    if time.perf_counter() > (timeIMU + tIMU):
                        break

                if time.perf_counter() > (timeGPS + tGPS):
                    if readFlag and ekf.curIMUdata['fusionQPoseValid']:
                        ############################################
                        ### vvv FOR VERIFICATION / DEBUGGING vvv ###
                        ############################################
                        #kf.nedPos = ecef2ned(np.radians(46.513519), np.radians(6.545234), 398., ekf.initEcef, ekf.ecef2nedMatrix)
                        # = np.array(ekf.nedPos)
                        d_gga = node.getNMEAGGA()
                        if d_gga=={}:
                            print('No fix, using defaults (46.51197N,6.624637E,404[m])')
                            d_gga={'lat':46.51197, 'lon':6.624637, 'alt':404.}
                        print(d_gga)
                        ekf.nedPos = ecef2ned(np.radians(d_gga['lat']), np.radians(d_gga['lon']), d_gga['alt'], ekf.initEcef, ekf.ecef2nedMatrix)
                        z = np.array(ekf.nedPos)
                        ############################################
                        ### ^^^ FOR VERIFICATION / DEBUGGING ^^^ ###
                        ############################################
                        ekf.measurement_update(0.01, z, logFile)

                        print('State update - saving to log file')
                        logFile.write('Exit GPS on {} - p_est:{} - v_est:{} - q_est:{}\n\n'.format(time.ctime(0.000001*ekf.curIMUdata['timestamp']),ekf.p_est.T, ekf.v_est.T, ekf.q_est.T))
                    break

    except(KeyboardInterrupt, SystemExit):
        logFile.close()
        print('Stopping GNSS...')
        node.turnOffGNSS()
        print('Done. Bye.\n')

    print('Done.\nExiting now.')
