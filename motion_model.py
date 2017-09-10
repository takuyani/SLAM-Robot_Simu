#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================
# brief        MotionModel
#
# author       Takuya Niibori
# attention    none
#==============================================================================

import numpy as np
from mylib import limit
import matplotlib.pyplot as plt

class MotionModel(object):
    """動作モデルclass
        モデルの仕様は書籍『確率ロボティクス』の「5章 ロボットの動作」。
        詳細は書籍を参考。
    """

    def __init__(self, dt, a1, a2, a3, a4, a5, a6):
        """コンストラクタ
        引数：
            a1～a6：雑音パラメータ
             ※詳細は書籍を参照
        返り値：
            なし
        """
        self.__mDt = dt
        self.__mNoise = (a1, a2, a3, a4, a5, a6)

    def moveWithNoise(self, aPose, aV, aW):
        """動作モデル（ノイズ有り）
        引数：
            aPose：姿勢（x, y, θ）
            aV：速度ν[m/s]
            aW：角速度ω[rad/s]
        返り値：
            newPose：動作後の姿勢（x, y, θ）
        """
        velSq = aV ** 2
        omgSq = aW ** 2

        velSigma = (self.__mNoise[0] * velSq) + (self.__mNoise[1] * omgSq)
        omgSigma = (self.__mNoise[2] * velSq) + (self.__mNoise[3] * omgSq)
        gamSigma = (self.__mNoise[4] * velSq) + (self.__mNoise[5] * omgSq)
        velHat = aV + np.random.normal(0.0, velSigma ** 2)
        omgHat = aW + np.random.normal(0.0, omgSigma ** 2)
        gamHat = np.random.normal(0.0, gamSigma ** 2)

        a = velHat / omgHat
        b = omgHat * self.__mDt
        pt = aPose[2, 0]

        pxs = aPose[0, 0] - (a * np.sin(pt)) + (a * np.sin(pt + b))
        pys = aPose[1, 0] + (a * np.cos(pt)) - (a * np.cos(pt + b))
        pts = limit.limit_angle(pt + (omgHat + gamHat) * self.__mDt)

        newPose = np.array([[pxs],
                            [pys],
                            [pts]])

        return newPose

    def moveWithoutNoise(self, aPose, aV, aW):
        """動作モデル（ノイズ無し）
        引数：
            aPose：姿勢（x, y, θ）
            aV：速度ν[m/s]
            aW：角速度ω[rad/s]
        返り値：
            newPose：動作後の姿勢（x, y, θ）
        """
        a = aV / aW
        b = limit.limit_angle(aW * self.__mDt)
        yaw = aPose[2, 0]
        yaw_add = limit.limit_angle(yaw + b)

        px = aPose[0, 0] + a * (-np.sin(yaw) + np.sin(yaw_add))
        py = aPose[1, 0] + a * (np.cos(yaw) - np.cos(yaw_add))
        pt = yaw_add

        newPose = np.array([[px],
                            [py],
                            [pt]])

        return newPose




if __name__ == "__main__":

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

    fig = plt.figure(figsize = (12, 9))
    ax = plt.subplot2grid((1, 1), (0, 0))

    a1 = [0.05, 0.05, 0.01, 0.01, 0.01, 0.01]
    samp1 = MotionModel(1.0, a1[0], a1[1], a1[2], a1[3], a1[4], a1[5])

    a2 = [0.01, 0.01, 0.05, 0.05, 0.01, 0.01]
    samp2 = MotionModel(1.0, a2[0], a2[1], a2[2], a2[3], a2[4], a2[5])

    a3 = [0.01, 0.01, 0.01, 0.01, 0.1, 0.1]
    samp3 = MotionModel(1.0, a3[0], a3[1], a3[2], a3[3], a3[4], a3[5])

    P1 = []
    P2 = []
    P3 = []
    for i in range(500):
        SampPose = samp1.moveWithNoise(Pose, VEL_mps, YAW_RATE_rps)
        P1.append(SampPose[0:2, :])
        SampPose = samp2.moveWithNoise(Pose, VEL_mps, YAW_RATE_rps)
        P2.append(SampPose[0:2, :])
        SampPose = samp3.moveWithNoise(Pose, VEL_mps, YAW_RATE_rps)
        P3.append(SampPose[0:2, :])

    a, b = np.array(np.concatenate(P1, axis = 1))
    ax.scatter(a, b, c = "red", marker = "o", alpha = 0.5, label = "Sampling1")

    a, b = np.array(np.concatenate(P2, axis = 1))
    ax.scatter(a, b, c = "green", marker = "o", alpha = 0.5, label = "Sampling2")

    a, b = np.array(np.concatenate(P3, axis = 1))
    ax.scatter(a, b, c = "blue", marker = "o", alpha = 0.5, label = "Sampling3")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Sampling Test")
    ax.axis("equal", adjustable = "box")
    ax.grid()
    ax.legend(fontsize = 10)

    plt.show()


