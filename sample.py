#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================
# brief        Sample
#
# author       Takuya Niibori
# attention    none
#==============================================================================

import numpy as np
from mylib import limit
import matplotlib.pyplot as plt

class Sample(object):
    """サンプリングclass"""

    def __init__(self, dt, a1, a2, a3, a4, a5, a6):
        """コンストラクタ
        引数：
            a1～a6：雑音パラメータ
        返り値：
            なし
        """
        self.__Dt = dt
        self.__Noise = (a1, a2, a3, a4, a5, a6)


    def sampleMotionModelVelocity(self, aPose, aCtr):
        """速度動作モデルのサンプリング
            状態方程式：x(k+1) = A * x(k) + B * u(k)
        引数：
            aPose：姿勢(x, y, θ)
            aCtr：制御(ν, ω)
        返り値：
            samp1：サンプリング後の姿勢(x, y, θ)
        """
        vel = aCtr[0, 0]
        omg = aCtr[1, 0]
        velSq = vel ** 2
        omgSq = omg ** 2

        velSigma = (self.__Noise[0] * velSq) + (self.__Noise[1] * omgSq)
        omgSigma = (self.__Noise[2] * velSq) + (self.__Noise[3] * omgSq)
        gamSigma = (self.__Noise[4] * velSq) + (self.__Noise[5] * omgSq)
        velHat = vel + np.random.normal(0.0, velSigma ** 2)
        omgHat = omg + np.random.normal(0.0, omgSigma ** 2)
        gamHat = np.random.normal(0.0, gamSigma ** 2)

        a = velHat / omgHat
        b = omgHat * self.__Dt

        px = aPose[0, 0]
        py = aPose[1, 0]
        pt = aPose[2, 0]

        pxs = px - (a * np.sin(pt)) + (a * np.sin(pt + b))
        pys = py + (a * np.cos(pt)) - (a * np.cos(pt + b))
        pts = pt + (omgHat + gamHat) * self.__Dt
        pts = limit.limit_angle(pts)

        samp1 = np.array([[pxs],
                         [pys],
                         [pts]])

        return samp1



if __name__ == '__main__':

    #---------- 状態空間モデルパラメータ定義 ----------
    RADIUS_m = 1.0  # 周回半径[m]
    YAW_RATE_rps = np.deg2rad(90.0)  # 角速度[rad/s]
    VEL_mps = RADIUS_m * YAW_RATE_rps  # 速度[m/s]

    px0 = RADIUS_m  # 位置x[m]
    py0 = 0.0  # 位置y[m]
    yaw0 = 90.0  # 角度yaw[deg]
    Pose = np.array([[px0],
                     [py0],
                     [np.deg2rad(yaw0)]])

    Ctr = np.array([[VEL_mps],
                    [YAW_RATE_rps]])


    fig = plt.figure(figsize=(12, 9))
    ax = plt.subplot2grid((1, 1), (0, 0))

    a1 = [0.01, 0.01, 0.1, 0.1, 0.01, 0.01]
    samp1 = Sample(1.0, a1[0], a1[1], a1[2], a1[3], a1[4], a1[5])

    a2 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    samp2 = Sample(1.0, a2[0], a2[1], a2[2], a2[3], a2[4], a2[5])

    P1 = []
    P2 = []
    for i in range(500):
        SampPose = samp1.sampleMotionModelVelocity(Pose, Ctr)
        P1.append(SampPose[0:2, :])
        SampPose = samp2.sampleMotionModelVelocity(Pose, Ctr)
        P2.append(SampPose[0:2, :])

    a, b = np.array(np.concatenate(P1, axis=1))
    ax.scatter(a, b, c="red", marker='.', alpha=0.5, label='Sampling1')

    a, b = np.array(np.concatenate(P2, axis=1))
    ax.scatter(a, b, c="blue", marker='.', alpha=0.5, label='Sampling2')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Sampling Test')
    ax.axis('equal', adjustable='box')
    ax.grid()
    ax.legend(fontsize=10)

    plt.show()


