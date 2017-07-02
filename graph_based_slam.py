#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================
# brief        Graph-based SLAM
#
# author       Takuya Niibori
# attention    none
#==============================================================================

import numpy as np
import scipy as sp
from numpy import matlib as matlib
from matplotlib import animation, mlab
import matplotlib.pyplot as plt
from mylib import transform
from mylib import error_ellipse
from mylib import limit
from mylib import transform as tf
from copy import deepcopy
from numpy.core.defchararray import upper


class Sensor(object):
    """センサclass"""

    def __init__(self, aRange_m, aAngle_rad):
        """"コンストラクタ
        引数：
            aRange_m：走査距離[m]
            aAngle_rad：走査角度[rad]
        """
        self.__mScanRange_m = aRange_m
        self.__mScanAngle_rad = aAngle_rad
        self.__mReslAng = int(np.rad2deg(aAngle_rad))

        ang = np.rad2deg(aAngle_rad)
        ofs = np.rad2deg(tf.BASE_ANG)

        step = 1.0
        self.__p = []
        self.__p.append([aRange_m * np.cos(np.deg2rad(x)) for x  in np.arange(-ang + ofs, ang + ofs + step, step)])
        self.__p.append([aRange_m * np.sin(np.deg2rad(x)) for x  in np.arange(-ang + ofs, ang + ofs + step, step)])
        self.__p[0].append(0.0)
        self.__p[1].append(0.0)
        self.__p[0].append(self.__p[0][0])
        self.__p[1].append(self.__p[1][0])

        self.__local = np.array([self.__p[0],
                                 self.__p[1]])


    def judgeInclusion(self, aPose, aLm):
        """"センサ計測範囲内包含判定
        引数：
            aPose：姿勢
               aPose[0, 0]：x座標[m]
               aPose[1, 0]：y座標[m]
               aPose[2, 0]：方角(rad)
            aLm：ワールド座標系ランドマーク
               aLm[n, 0]：x座標
               aLm[n, 1]：y座標
               n:要素数
        """
        lmLo = tf.world2local(aPose, aLm)

        normLm = np.linalg.norm(lmLo, axis = 1)      # ノルム計算
        radLm = np.arctan2(lmLo[:, 1], lmLo[:, 0])   # 角度計算

        upperRad = tf.BASE_ANG + self.__mScanAngle_rad
        lowerRad = tf.BASE_ANG - self.__mScanAngle_rad
        isInc = [ True if (normLm[i]<=self.__mScanRange_m and (lowerRad <= radLm[i] and radLm[i]<=upperRad)) else False for i in range(lmLo.shape[0])]

        return isInc



    def draw(self, aAx, aColor, aPose):
        """"描写
        引数：
            aPose：姿勢
               aPose[0, 0]：x座標[m]
               aPose[1, 0]：y座標[m]
               aPose[2, 0]：方角(rad)
            aDt：演算周期[sec]
        """
        world = tf.local2world(aPose, self.__local.T)
        aAx.plot(world.T[0], world.T[1], c = aColor, linewidth = 1.0, linestyle = '-')


class Robot(object):
    """ロボットclass"""

    def __init__(self, aPose, aDt):
        """"コンストラクタ
        引数：
            aPose：姿勢
               aPose[0, 0]：x座標[m]
               aPose[1, 0]：y座標[m]
               aPose[2, 0]：方角(rad)
            aDt：演算周期[sec]
        """

        self.__mSensor = Sensor(5.0, np.deg2rad(30.0))

        #---------- 制御周期 ----------
        self.__mDt = aDt

        #---------- 姿勢定義 ----------
        self.__mPose = np.array([[aPose[0, 0]],  # x座標[m]
                                 [aPose[1, 0]],  # y座標[m]
                                 [aPose[2, 0]]])  # 方角yaw[deg]

    def getPose(self):
        """"姿勢取得処理
        引数：
            なし
        返り値：
            x：姿勢

        """
        return deepcopy(self.__mPose)


    def motionModel(self, aV, aW):
        """"動作処理
        引数：
            aV：速度ν[m/s]
            aW：角速度ω[rad/s]
        返り値：
        """
        a = aV / aW
        b = limit.limit_angle(aW * self.__mDt)
        yaw = self.__mPose[2, 0]
        yaw_add = limit.limit_angle(yaw + b)

        self.__mPose[0, 0] += a * (-np.sin(yaw) + np.sin(yaw_add))
        self.__mPose[1, 0] += a * (np.cos(yaw) - np.cos(yaw_add))
        self.__mPose[2, 0] = yaw_add

    def judgeInclusion(self, aLm):
        return self.__mSensor.judgeInclusion(self.__mPose, aLm)

    def draw(self, aAx, aColor):
        self.__mSensor.draw(aAx, "green", self.__mPose)

        x = self.__mPose[0, 0]
        y = self.__mPose[1, 0]
        # 矢印（ベクトル）の成分
        u = np.cos(self.__mPose[2, 0])
        v = np.sin(self.__mPose[2, 0])
        # 矢印描写
        aAx.quiver(x, y, u, v, color = aColor, angles = 'xy', scale_units = 'xy', scale = 1)




time_s = 0
P1 = []

def graph_based_slam(i, aPeriod_ms, aRobot):
    """"Graph-based SLAM処理
    引数：
        period_ms：更新周期[msec]
    返り値：
        なし
    """
    global time_s
    global P1

    #---------- ランドマーク ----------
    """
    LM(j) = [j番目LMのX座標, j番目LMのY座標]
    """
    LM = np.array([[ 5.0, 5.0],
                   [ 2.0, -3.0],
                   [ 0.0, 10.0],
                   [-5.0, -1.0],
                   [ 0.0, 0.0]])

    RADIUS_m = 10.0  # 周回半径[m]

    OMEGA_rps = np.deg2rad(10.0)  # 角速度[rad/s]
    VEL_mps = RADIUS_m * OMEGA_rps  # 速度[m/s]

    col_x_true = 'red'
    time_s += aPeriod_ms / 1000

    aRobot.motionModel(VEL_mps, OMEGA_rps)
    x = aRobot.getPose()

    isSns = aRobot.judgeInclusion(LM)
    actLM = np.array([ LM[i] for i in range(len(isSns))  if isSns[i] == True  ])

    plt.cla()

    # サブプロットを追加
    ax1 = plt.subplot2grid((1, 1), (0, 0), aspect='equal', adjustable='box-forced')

    # ランドマークの描写
    ax1.scatter(LM[:, 0], LM[:, 1], s = 100, c = "yellow", marker = "*", alpha = 0.5, linewidths = "2",
                edgecolors = "orange", label = 'Land Mark')
    if len(actLM) != 0:
        ax1.scatter(actLM[:, 0], actLM[:, 1], s = 100, c = "red", marker = "*", alpha = 0.5, linewidths = "2",
                    edgecolors = "red", label = 'Land Mark')

    aRobot.draw(ax1, "red")

    # 状態x(真値)の描写
    P1.append(x[0:2, :])
    a, b = np.array(np.concatenate(P1, axis = 1))
    ax1.plot(a, b, c = col_x_true, linewidth = 1.0, linestyle = '-', label = 'Ground Truth')

    print('time:{0:.3f}[s] x = {1:.3f}[m], y = {2:.3f}[m], θ = {3:.3f}[deg]'.format(time_s, x[0, 0], x[1, 0], np.rad2deg(x[2, 0])))

    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Graph-based SLAM')
    ax1.axis('equal', adjustable = 'box')
    ax1.grid()
    ax1.legend(fontsize = 10)


#    ax2.scatter(loLM[:, 0], loLM[:, 1], s = 100, c = "yellow", marker = "*", alpha = 0.5, linewidths = "2",
#                edgecolors = "orange", label = 'Land Mark')

#    ax2.set_xlabel('x [m]')
#    ax2.set_ylabel('y [m]')
#    ax2.set_title('Graph-based SLAM')
#    ax2.axis('equal', adjustable = 'box')
#    ax2.grid()
#    ax2.legend(fontsize = 10)


if __name__ == '__main__':

    period_ms = 100  # 更新周期[msec]
    frame_cnt = int(36 * 1000 / period_ms)

    # 描画
    fig = plt.figure(figsize = (18, 9))

    x_base = np.array([[10.0],  # x座標[m]
                       [ 0.0],  # y座標[m]
                       [np.deg2rad(90.0)]])  # 方角yaw[deg]

    oRobot = Robot(x_base, period_ms / 1000)

    ani = animation.FuncAnimation(fig, graph_based_slam, frames = frame_cnt, fargs = (period_ms, oRobot), blit = False,
                                  interval = period_ms, repeat = False)

    # ani.save('Localization_by_pf.mp4', bitrate=6000)

    plt.show()

