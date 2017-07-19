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
import motion_model as mm

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

        self.__R = np.diag([0.001, 0.001, 0.001]) ** 2


    def setNoiseParameter(self, aCovPx_m, aCovPy_m, aCovAng_rad):
        """"観測雑音パラメータ設定
        引数：
            aCovPx_m：x方向[m]の標準偏差
            aCovPy_m：y方向[m]の標準偏差
            aCovAng_rad：方向[rad]の標準偏差
        返り値：
           なし
        """
        self.__R = np.diag([aCovPx_m, aCovPy_m, aCovAng_rad]) ** 2

    def scan(self, aPose):
        """"スキャン結果
        引数：
            aPose：姿勢
               aPose[0, 0]：x座標[m]
               aPose[1, 0]：y座標[m]
               aPose[2, 0]：方角(rad)
        返り値：
           なし
        """
        obs = []

        landMarkTrueDir = limit.limit_angle(tf.BASE_ANG * 2.0 - aPose[2, 0])

        robotLandMarks = tf.world2robot(aPose, self.__mLandMarks)

        normLm = np.linalg.norm(robotLandMarks, axis = 1)  # ノルム計算
        radLm = np.arctan2(robotLandMarks[:, 1], robotLandMarks[:, 0])  # 角度計算

        upperRad = tf.BASE_ANG + self.__mScanAngle_rad
        lowerRad = tf.BASE_ANG - self.__mScanAngle_rad
        self.__mObsFlg = [ True if (normLm[i] <= self.__mScanRange_m and (lowerRad <= radLm[i] and radLm[i] <= upperRad)) else False for i in range(robotLandMarks.shape[0])]

        for i in range(robotLandMarks.shape[0]):
            if (normLm[i] <= self.__mScanRange_m and (lowerRad <= radLm[i] and radLm[i] <= upperRad)):
                n = np.random.multivariate_normal([0.0, 0.0, 0.0], self.__R, 1).T
                xp = robotLandMarks[i, 0] + n[0, 0]
                yp = robotLandMarks[i, 1] + n[1, 0]
                tp = landMarkTrueDir + n[2, 0]
                obs.append([i, xp, yp, tp])

        return obs


    def draw(self, aAx, aColor, aPose):
        """"描写
        引数：
            aPose：姿勢
               aPose[0, 0]：x座標[m]
               aPose[1, 0]：y座標[m]
               aPose[2, 0]：方角(rad)
        """
        world = tf.robot2world(aPose, self.__local.T)
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
#        self.__mMvMdl = mm.MotionModel(aDt, 0.3, 0.3, 0.3, 0.3, 0.1, 0.1)
        self.__mMvMdl = mm.MotionModel(aDt, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001)

        #---------- 制御周期 ----------
        self.__mDt = aDt

        #---------- 姿勢 ----------
        self.__mPosesActu = [aPose]  # 姿勢（実際値）
        self.__mPosesGues = [aPose]  # 姿勢（推定値）
        #---------- 制御 ----------
        self.__mCtr = []
        #---------- 観測 ----------
        self.__mObs = []


    def getPose(self):
        """"姿勢取得処理
        引数：
            なし
        返り値：
            x：姿勢

        """
        return self.__mPosesActu[-1]

    def move(self, aV, aW):

        poseActu = self.__mMvMdl.moveWithNoise(self.__mPosesActu[-1], aV, aW)
        poseGues = self.__mMvMdl.moveWithoutNoise(self.__mPosesGues[-1], aV, aW)

        # 履歴保持
        self.__mCtr.append(np.array([aV, aW]))  # 制御
        self.__mPosesActu.append(poseActu)      # 姿勢（実際値）
        self.__mPosesGues.append(poseGues)      # 姿勢（推定値）
        self.__mObs.append(self.__mScnSnsr.scan(poseActu))  # 観測結果


#    def estimateOpticalTrajectory(self):


    def draw(self, aAx, aAx2):
        self.__mScnSnsr.draw(aAx, "green", self.__mPosesActu[-1])

        self.__drawPoses(aAx, "red", "Guess", self.__mPosesGues)
        self.__drawPoses(aAx, "blue", "Actual", self.__mPosesActu)

        self.__debug(aAx2)


    def __drawPoses(self, aAx, aColor, aLabel, aPoses):
        x = aPoses[-1][0, 0]
        y = aPoses[-1][1, 0]
        # 矢印（ベクトル）の成分
        u = np.cos(aPoses[-1][2, 0])
        v = np.sin(aPoses[-1][2, 0])
        # 矢印描写
        aAx.quiver(x, y, u, v, color = aColor, angles = "xy", scale_units = "xy", scale = 1)

        # 軌跡描写
        pxa = [e[0, 0] for e in aPoses]
        pya = [e[1, 0] for e in aPoses]
        aAx.plot(pxa, pya, c = aColor, linewidth = 1.0, linestyle = "-", label = aLabel)

    def __debug(self, aAx):
        a = self.__mObs[-1]
        if len(a) != 0:
            pxa = [e[1] for e in a]
            pya = [e[2] for e in a]
            pta = [e[3] for e in a]
            aAx.scatter(pxa, pya, c="red", marker='o', alpha=0.5)

            x = pxa
            y = pya
            # 矢印（ベクトル）の成分
            u = np.cos(pta)
            v = np.sin(pta)
            # 矢印描写
            aAx.quiver(x, y, u, v, color = "red", angles = "xy", scale_units = "xy", scale = 1)



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

    gRbt.move(VEL_mps, OMEGA_rps)
    x = gRbt.getPose()

    plt.cla()

    # サブプロットを追加
    ax1 = plt.subplot2grid((1, 2), (0, 0), aspect = "equal", adjustable = "box-forced")
    ax2 = plt.subplot2grid((1, 2), (0, 1), aspect = "equal", adjustable = "box-forced")

    gRbt.draw(ax1, ax2)

    print("time:{0:.3f}[s] x = {1:.3f}[m], y = {2:.3f}[m], θ = {3:.3f}[deg]".format(time_s, x[0, 0], x[1, 0], np.rad2deg(x[2, 0])))

    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Graph-based SLAM")
    ax1.axis("equal", adjustable = "box")
    ax1.grid()
    ax1.legend(fontsize = 10)

    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Debug")
#    ax2.axis("equal", adjustable = "box")
    ax2.axis([-15, 15, -15, 15])
    ax2.grid()
    ax2.legend(fontsize = 10)

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

