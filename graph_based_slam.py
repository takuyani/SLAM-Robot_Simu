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
from mylib import error_ellipse
from mylib import limit
from mylib import transform as tf


class ScanSensor(object):
    """スキャンセンサclass"""

    def __init__(self, aRange_m, aAngle_rad, aLandMarks):
        """"コンストラクタ
        引数：
            aRange_m：走査距離[m]
            aAngle_rad：走査角度[rad]
        """
        self.__mScanRange_m = aRange_m
        self.__mScanAngle_rad = aAngle_rad
        self.__mReslAng = int(np.rad2deg(aAngle_rad))
        self.__mLandMarks = aLandMarks
        self.__mNum = aLandMarks.shape[0]
        self.__mObsFlg = [[False] * self.__mNum]

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


    def judgeInclusion(self, aPose):
        """"センサ計測範囲内包含判定
        引数：
            aPose：姿勢
               aPose[0, 0]：x座標[m]
               aPose[1, 0]：y座標[m]
               aPose[2, 0]：方角(rad)
        返り値：
           なし
        """
        lmLo = tf.world2local(aPose, self.__mLandMarks)

        normLm = np.linalg.norm(lmLo, axis = 1)  # ノルム計算
        radLm = np.arctan2(lmLo[:, 1], lmLo[:, 0])  # 角度計算

        upperRad = tf.BASE_ANG + self.__mScanAngle_rad
        lowerRad = tf.BASE_ANG - self.__mScanAngle_rad
        self.__mObsFlg = [ True if (normLm[i] <= self.__mScanRange_m and (lowerRad <= radLm[i] and radLm[i] <= upperRad)) else False for i in range(lmLo.shape[0])]


    def draw(self, aAx, aColor, aPose):
        """"描写
        引数：
            aPose：姿勢
               aPose[0, 0]：x座標[m]
               aPose[1, 0]：y座標[m]
               aPose[2, 0]：方角(rad)
        """
        world = tf.local2world(aPose, self.__local.T)
        aAx.plot(world.T[0], world.T[1], c = aColor, linewidth = 1.0, linestyle = "-")

        # ランドマークの描写
        actLM = np.array([ self.__mLandMarks[i] for i in range(len(self.__mObsFlg))  if self.__mObsFlg[i] == True  ])
        aAx.scatter(self.__mLandMarks[:, 0], self.__mLandMarks[:, 1], s = 100, c = "yellow", marker = "*", alpha = 0.5, linewidths = "2", edgecolors = "orange", label = "Land Mark")
        if len(actLM) != 0:
            aAx.scatter(actLM[:, 0], actLM[:, 1], s = 100, c = "red", marker = "*", alpha = 0.5, linewidths = "2", edgecolors = "red", label = "Land Mark")



class Robot(object):
    """ロボットclass"""

    def __init__(self, aPose, aDt, aScanRng, aScanAng, aLandMarks):
        """"コンストラクタ
        引数：
            aPose：姿勢
               aPose[0, 0]：x座標[m]
               aPose[1, 0]：y座標[m]
               aPose[2, 0]：方角(rad)
            aDt：演算周期[sec]
        """

        self.__mScnSnsr = ScanSensor(aScanRng, aScanAng, aLandMarks)

        #---------- 制御周期 ----------
        self.__mDt = aDt

        #---------- 姿勢 ----------
        self.__mPosesAct = []
        self.__mPosesAct.append(aPose)


    def getPose(self):
        """"姿勢取得処理
        引数：
            なし
        返り値：
            x：姿勢

        """
        return self.__mPosesAct[-1]


    def motionModel(self, aV, aW):
        """"動作処理
        引数：
            aV：速度ν[m/s]
            aW：角速度ω[rad/s]
        返り値：
        """
        a = aV / aW
        b = limit.limit_angle(aW * self.__mDt)
        yaw = self.__mPosesAct[-1][2, 0]
        yaw_add = limit.limit_angle(yaw + b)

        px = self.__mPosesAct[-1][0, 0] + a * (-np.sin(yaw) + np.sin(yaw_add))
        py = self.__mPosesAct[-1][1, 0] + a * (np.cos(yaw) - np.cos(yaw_add))
        pt = yaw_add

        newPose = np.array([[px],
                            [py],
                            [pt]])

        self.__mPosesAct.append(newPose)

        self.__mScnSnsr.judgeInclusion(newPose)


    def draw(self, aAx, aColor):
        self.__mScnSnsr.draw(aAx, "green", self.__mPosesAct[-1])

        x = self.__mPosesAct[-1][0, 0]
        y = self.__mPosesAct[-1][1, 0]
        # 矢印（ベクトル）の成分
        u = np.cos(self.__mPosesAct[-1][2, 0])
        v = np.sin(self.__mPosesAct[-1][2, 0])
        # 矢印描写
        aAx.quiver(x, y, u, v, color = aColor, angles = "xy", scale_units = "xy", scale = 1)

        # 軌跡描写
        pxa = [e[0, 0] for e in self.__mPosesAct]
        pya = [e[1, 0] for e in self.__mPosesAct]
        aAx.plot(pxa, pya, c = "red", linewidth = 1.0, linestyle = "-", label = "Ground Truth")


# スキャンセンサモデル
SCN_SENS_RANGE_m = 10.0  # 走査距離[m]
SCN_SENS_ANGLE_rps = np.deg2rad(70.0)  # 走査角度[rad]
RADIUS_m = 10.0  # 周回半径[m]

# ロボット動作モデル
OMEGA_rps = np.deg2rad(10.0)  # 角速度[rad/s]
VEL_mps = RADIUS_m * OMEGA_rps  # 速度[m/s]

# ランドマーク
LAND_MARKS = np.array([[ 5.0, 5.0],
                       [ 2.0, -3.0],
                       [ 0.0, 10.0],
                       [-5.0, -1.0],
                       [ 0.0, 0.0]])

# アニメーション更新周期[msec]
PERIOD_ms = 100

x_base = np.array([[10.0],  # x座標[m]
                   [ 0.0],  # y座標[m]
                   [np.deg2rad(90.0)]])  # 方角yaw[deg]

gRbt = Robot(x_base, PERIOD_ms / 1000, SCN_SENS_RANGE_m, SCN_SENS_ANGLE_rps, LAND_MARKS)

time_s = 0

def graph_based_slam(i, aPeriod_ms):
    """"Graph-based SLAM処理
    引数：
        PERIOD_ms：更新周期[msec]
    返り値：
        なし
    """
    global time_s
    global P1
    global RADIUS_m
    global OMEGA_rps
    global VEL_mps

    time_s += aPeriod_ms / 1000

    gRbt.motionModel(VEL_mps, OMEGA_rps)
#    gRbt.judgeInclusion()
    x = gRbt.getPose()

    plt.cla()

    # サブプロットを追加
    ax1 = plt.subplot2grid((1, 1), (0, 0), aspect = "equal", adjustable = "box-forced")

    gRbt.draw(ax1, "red")

    print("time:{0:.3f}[s] x = {1:.3f}[m], y = {2:.3f}[m], θ = {3:.3f}[deg]".format(time_s, x[0, 0], x[1, 0], np.rad2deg(x[2, 0])))

    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Graph-based SLAM")
    ax1.axis("equal", adjustable = "box")
    ax1.grid()
    ax1.legend(fontsize = 10)


#    ax2.scatter(loLM[:, 0], loLM[:, 1], s = 100, c = "yellow", marker =ax", alpha = 0.5, linewidthsax"2",
#                edgaxlors = "orange", label = "Land Mark"ax
#    ax2.set_xlabel("x [m]")
#    ax2axt_ylabel("y [ax)
#    ax2.set_title("Graph-based SLAM")
#    ax2.axis("equal", adjustable = "box")
#    ax2.grid()
#    ax2.legend(fontsize = 10)


if __name__ == "__main__":

    frame_cnt = int(36 * 1000 / PERIOD_ms)

    fig = plt.figure(figsize = (18, 9))

    ani = animation.FuncAnimation(fig, graph_based_slam, frames = frame_cnt, fargs = (PERIOD_ms,), blit = False,
                                  interval = PERIOD_ms, repeat = False)

    # ani.save("Localization_by_pf.mp4", bitrate=6000)

    plt.show()

