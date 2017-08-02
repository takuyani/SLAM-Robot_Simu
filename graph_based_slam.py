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
from docutils.parsers import null
from matplotlib import animation, patches

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
        self.__mLandMarksNum = aLandMarks.shape[0]
        self.__mObsFlg = [[False] * self.__mLandMarksNum]

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

        # 観測雑音定義
        self.__DIST_NOISE = 10  # ランドマーク距離雑音[%]
        self.__R_DIST = self.__DIST_NOISE / 100  # ランドマーク距離雑音ゲイン
        self.__R_DIR_SIGMA = 3 * np.pi / 180  # ランドマーク観測方向雑音標準偏差[rad]
        self.__R_ORIENT_SIGMA = 3 * np.pi / 180  # ランドマーク向き雑音標準偏差[rad]

#        self.__R = np.diag([0.001, 0.001, 0.001]) ** 2


#    def setNoiseParameter(self, aCovPx_m, aCovPy_m, aCovAng_rad):
        """"観測雑音パラメータ設定
        引数：
            aCovPx_m：x方向[m]の標準偏差
            aCovPy_m：y方向[m]の標準偏差
            aCovAng_rad：方向[rad]の標準偏差
        返り値：
           なし
        """
#        self.__R = np.diag([aCovPx_m, aCovPy_m, aCovAng_rad]) ** 2

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
        obsWithNoise = []
        obsWithoutNoise = []
        robotLandMarks = tf.world2robot(aPose, self.__mLandMarks)  # 世界座標系→ロボット座標系変換
        distLm = np.linalg.norm(robotLandMarks, axis = 1)  # ランドマーク距離算出
        dirLm_rad = np.arctan2(robotLandMarks[:, 1], robotLandMarks[:, 0])  # ランドマーク観測方向算出
        orientLm_rad = np.ones(robotLandMarks.shape[0]) * (tf.BASE_ANG - aPose[2, 0])  # ランドマーク向き算出

        upperRad = tf.BASE_ANG + self.__mScanAngle_rad
        lowerRad = tf.BASE_ANG - self.__mScanAngle_rad
        self.__mObsFlg = [ True if (distLm[i] <= self.__mScanRange_m and (lowerRad <= dirLm_rad[i] and dirLm_rad[i] <= upperRad)) else False for i in range(len(dirLm_rad))]

        for i, flg in enumerate(self.__mObsFlg):
            if (flg == True):
                distActu = np.random.normal(distLm[i], distLm[i] * self.__R_DIST)
                dirActu = limit.limit_angle(np.random.normal(dirLm_rad[i], self.__R_DIR_SIGMA))
                orientActu = limit.limit_angle(np.random.normal(orientLm_rad[i], self.__R_ORIENT_SIGMA))
                obsWithNoise.append([i, [distActu, dirActu, orientActu]])
                obsWithoutNoise.append([i, [distLm[i], dirLm_rad[i], orientLm_rad[i]]])

        return obsWithNoise, obsWithoutNoise

    def rotateCovariance(self, aCov, aRad):
        c = np.cos(aRad)
        s = np.sin(aRad)

        rotmat = np.array([[  c,  -s, 0.0],
                           [  s,   c, 0.0],
                           [0.0, 0.0, 1.0]])

        return rotmat @ aCov @ rotmat.T

    def getLandMarkCovMatrixOnMeasurementSys(self, aLandMark):
        """計測座標系におけるランドマークの共分散行列取得
        引数：
            aPose：世界座標系でのロボット姿勢
               aPose[0, 0]：x座標[m]
               aPose[1, 0]：y座標[m]
               aPose[2, 0]：方角(rad)
            aLandMark：計測座標家でのランドマーク姿勢
                aLandMark[0]：ユークリッド距離
                aLandMark[1]：観測方向
                aLandMark[2]：ランドマーク向き
        返り値：
            covMat：計測座標系での共分散行列(3×3)
                対角成分
                1：x軸の共分散
                2：y軸の共分散
                3：θ方向の共分散
        """
        dist = aLandMark[0] * self.__R_DIST
        dir_cov = (aLandMark[0] * np.sin(self.__R_DIR_SIGMA)) ** 2
        orient_cov = self.__R_DIR_SIGMA ** 2 + self.__R_ORIENT_SIGMA ** 2
        covMat = np.array([[dist ** 2, 0, 0         ],
                           [0, dir_cov, 0         ],
                           [0, 0, orient_cov]])

        return covMat

    def tfMeasurement2World(self, aCovMat, aLandMarkDir, aRobotDir):
        """計測座標系→世界座標系変換
        引数：
            aCovMat：共分散行列
            aLandMarkDir：ロボット座標系でのランドマーク観測方向[rad]
            aRobotDir：世界座標系でのロボット方角[rad]
        返り値：
            covMatWorld：世界座標系での共分散行列(3×3)
        """
        ang = aLandMarkDir + aRobotDir - tf.BASE_ANG
        c = np.cos(ang)
        s = np.sin(ang)
        rotMat = np.array([[c, -s, 0],
                           [s,  c, 0],
                           [0,  0, 1]])

        covMatWorld = rotMat @ aCovMat @ rotMat.T

        return covMatWorld


    def tfMeasurement2Robot(self, aCovMat, aLandMarkDir):
        """計測座標系→ロボット座標系変換
        引数：
            aCovMat：共分散行列
            aLandMarkDir：ロボット座標系でのランドマーク観測方向[rad]
        返り値：
            covMatRobot：ロボット座標系での共分散行列(3×3)
        """
        c = np.cos(aLandMarkDir)
        s = np.sin(aLandMarkDir)
        rotMat = np.array([[c, -s, 0],
                           [s,  c, 0],
                           [0,  0, 1]])

        covMatRobot = rotMat @ aCovMat @ rotMat.T

        return covMatRobot

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
        aAx.scatter(self.__mLandMarks[:, 0], self.__mLandMarks[:, 1], s = 100, c = "yellow", marker = "*", alpha = 0.5, linewidths = "2", edgecolors = "orange", label = "Land Mark(True)")



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
#        self.__mMvMdl = mm.MotionModel(aDt, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
        self.__mMvMdl = mm.MotionModel(aDt, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001)

        #---------- 制御周期 ----------
        self.__mDt = aDt

        #---------- 姿勢 ----------
        self.__mPosesActu = [aPose]  # 姿勢（実際値）
        self.__mPosesGues = [aPose]  # 姿勢（推定値）
        #---------- 制御 ----------
        self.__mCtr = [[]]
        #---------- 観測 ----------
        self.__mObsActu = [[]]
        self.__mObsTrue = [[]]

        #---------- 情報行列 ----------
        self.__mInfoMat = [[]]


        # 誤差楕円の信頼区間[%]
        self.__mConfidence_interval = 99.0
        self.__mEllipse = error_ellipse.ErrorEllipse(self.__mConfidence_interval)


    def getPose(self):
        """"姿勢取得処理
        引数：
            なし
        返り値：
            x：姿勢

        """
        return self.__mPosesActu[-1]

    def move(self, aV, aW):
        print("[移動]")

        poseActu = self.__mMvMdl.moveWithNoise(self.__mPosesActu[-1], aV, aW)
        poseGues = self.__mMvMdl.moveWithoutNoise(self.__mPosesGues[-1], aV, aW)

        # 履歴保持
        self.__mCtr.append(np.array([aV, aW]))  # 制御
        self.__mPosesActu.append(poseActu)  # 姿勢（実際値）
        self.__mPosesGues.append(poseGues)  # 姿勢（推定値）

        obsWithNoise, obsWithoutNoise = self.__mScnSnsr.scan(poseActu)
        self.__mObsActu.append(obsWithNoise)  # 観測結果
        self.__mObsTrue.append(obsWithoutNoise)  # 観測結果


    def estimateOpticalTrajectory(self):
        print("[軌跡推定]Ctr:{0}, PosesActu = {1}, PosesGues = {2}, Obs = {3}".format(len(self.__mCtr), len(self.__mPosesActu),
                                                                            len(self.__mPosesGues), len(self.__mObsActu)))

        obsNext = []
        poseNext = []
        self.__mInfoMat = []
        for obsCrnt, poseCrnt in zip(reversed(self.__mObsActu), reversed(self.__mPosesGues)):
            infoMat = []
            if len(obsCrnt) > 0 and len(obsNext) > 0:
                for j in range(len(obsCrnt)):
                    for k in range(len(obsNext)):
                        if obsCrnt[j][0] == obsNext[k][0]:
                            obsPoseCrnt = obsCrnt[j][1]
                            obsPoseNext = obsNext[k][1]

                            # 観測結果によるエッジ算出
                            lmCrntWorld = self.__tfRobot2LandMark(obsPoseCrnt)
                            lmNextWorld = self.__tfRobot2LandMark(obsPoseNext)
                            relPose = self.__calcRelativePoseByObservation(lmCrntWorld, lmNextWorld)
                            print("<相対姿勢>(LandMark ID:{0})>t[{1:.2f}[m], {2:.2f}[deg], {3:.2f}[deg], t+1[{4:.2f}[m], {5:.2f}[deg], {6:.2f}[deg]], Rel[x={7:.2f}[m], y={8:.2f}[m], t={9:.2f}[deg]]"
                                  .format(j,
                                          lmCrntWorld[0], np.rad2deg(lmCrntWorld[1]), np.rad2deg(lmCrntWorld[2]),
                                          lmNextWorld[0], np.rad2deg(lmNextWorld[1]), np.rad2deg(lmNextWorld[2]),
                                          relPose[0, 0], relPose[1, 0], np.rad2deg(relPose[2, 0])))

                            # 計測座標系での情報行列算出
                            lmCovCrntM = self.__mScnSnsr.getLandMarkCovMatrixOnMeasurementSys(obsPoseCrnt)
                            lmCovCrntW = self.__mScnSnsr.tfMeasurement2World(lmCovCrntM, obsPoseCrnt[1], poseCrnt[2, 0])

                            lmCovNextM = self.__mScnSnsr.getLandMarkCovMatrixOnMeasurementSys(obsPoseNext)
                            lmCovNextW = self.__mScnSnsr.tfMeasurement2World(lmCovNextM, obsPoseNext[1], poseNext[2, 0])

                            infoMat = (obsCrnt[j][0], np.linalg.inv(lmCovCrntW + lmCovNextW))

            else:
                print("なし")

            obsNext = obsCrnt
            poseNext = poseCrnt
            self.__mInfoMat.insert(-1, infoMat)


    def __calcRelativePoseByObservation(self, aOrigin, aTarget):
        """観測結果による、相対姿勢算出
        引数：
            aOrigin：基準ロボット姿勢
                aOrigin[0]：ユークリッド距離
                aOrigin[1]：観測方向
                aOrigin[2]：ランドマーク向き
            aTarget：目標ロボット姿勢
                aTarget[0]：ユークリッド距離
                aTarget[1]：観測方向
                aTarget[2]：ランドマーク向き
        返り値：
            rel：相対ロボット姿勢
                rel[0, 0]：x座標[m]
                rel[1, 0]：y座標[m]
                rel[2, 0]：方角(rad)
        """
        px = aTarget[0] * np.cos(aTarget[1]) - aOrigin[0] * np.cos(aOrigin[1])
        py = aTarget[0] * np.sin(aTarget[1]) - aOrigin[0] * np.sin(aOrigin[1])
        pt = aTarget[2] - aOrigin[2]

        rel = np.array([[px],
                        [py],
                        [pt]])
        return rel


    def __tfRobot2LandMark(self, aLandMark):
        """ロボット座標系→ランドマーク世界座標系変換
            ロボットを原点とした座標系からランドマークを原点とした世界座標系に変換し、
            変換後のロボット姿勢を戻り値として返す。
        引数：
            aLandMark：ランドマーク姿勢
                aLandMark[0]：ユークリッド距離
                aLandMark[1]：観測方向
                aLandMark[2]：ランドマーク向き
        返り値：
            robot：ランドマークを原点とした世界座標系でのロボット姿勢
                robot[0]：ユークリッド距離
                robot[1]：観測方向
                robot[2]：ランドマーク向き
        """
        dist = aLandMark[0]
        direct = limit.limit_angle(np.pi + aLandMark[1] - aLandMark[2])
        orient = limit.limit_angle(tf.BASE_ANG - aLandMark[2])
        robot = [dist, direct, orient]
        return robot


    def draw(self, aAx1, aAx2):
        self.__mScnSnsr.draw(aAx1, "green", self.__mPosesActu[-1])

        self.__drawPoses(aAx1, "red", "Guess", self.__mPosesGues)
        self.__drawPoses(aAx1, "blue", "Actual", self.__mPosesActu)
        self.__drawActualLandMark(aAx1)

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

    def __drawActualLandMark(self, aAx):
        obsCrnt = self.__mObsActu[-1]
        poseCrnt = self.__mPosesActu[-1]
        if len(obsCrnt) != 0:
            for obs in obsCrnt:
                obsPose = obs[1]
                lmCovM = self.__mScnSnsr.getLandMarkCovMatrixOnMeasurementSys(obsPose)
                lmCovW = self.__mScnSnsr.tfMeasurement2World(lmCovM, obsPose[1], poseCrnt[2, 0])
                Pxy = lmCovW[0:2, 0:2]
                x, y, ang_rad = self.__mEllipse.calc_error_ellipse(Pxy)
                px = (obsPose[0] * np.cos(obsPose[1] + poseCrnt[2, 0] - tf.BASE_ANG)) + poseCrnt[0, 0]
                py = (obsPose[0] * np.sin(obsPose[1] + poseCrnt[2, 0] - tf.BASE_ANG)) + poseCrnt[1, 0]
                p = ( px, py )
                # 誤差楕円描写
                e = patches.Ellipse(p, x, y, angle = np.rad2deg(ang_rad), linewidth = 2, alpha = 0.2,
                             facecolor = 'yellow', edgecolor = 'black', label = 'Error Ellipse: %.2f[%%]' %
                             self.__mConfidence_interval)
                aAx.add_patch(e)
                # 実測ランドマーク描写
                aAx.scatter(px, py, s = 100, c = "red", marker = "*", alpha = 0.5, linewidths = "2", edgecolors = "red", label = "Land Mark(Actual)")

                # ロボット-ランドマーク間線分描写
                ps = poseCrnt[0:2, 0].T
                xl = np.array([ps[0], px])
                yl = np.array([ps[1], py])
                aAx.plot(xl, yl, '--', c='green')


    def __debug(self, aAx):

        gain = 2

        obsCrnt = self.__mObsTrue[-1]
        if len(obsCrnt) != 0:
            pxa = [obs[1][0] * np.cos(obs[1][1]) for obs in obsCrnt]
            pya = [obs[1][0] * np.sin(obs[1][1]) for obs in obsCrnt]
            pta = [obs[1][2] for obs in obsCrnt]
            # ランドマーク描写
            aAx.scatter(pxa, pya, s = 100, c = "yellow", marker = "*", alpha = 0.5, linewidths = "2",
                        edgecolors = "orange", label = "Land Mark(True)")
            x = pxa
            y = pya
            # 矢印（ベクトル）の成分
            u = gain * np.cos(pta)
            v = gain * np.sin(pta)
            # 矢印描写
            aAx.quiver(x, y, u, v, color = "orange", angles = "xy", scale_units = "xy", scale = 1)


        obsCrnt = self.__mObsActu[-1]
        if len(obsCrnt) != 0:
            pxa = [obs[1][0] * np.cos(obs[1][1]) for obs in obsCrnt]
            pya = [obs[1][0] * np.sin(obs[1][1]) for obs in obsCrnt]
            pta = [obs[1][2] for obs in obsCrnt]
            # ランドマーク描写
            aAx.scatter(pxa, pya, s = 100, c = "red", marker = "*", alpha = 0.5, linewidths = "2",
                        edgecolors = "red", label = "Land Mark(Actual)")

            x = pxa
            y = pya
            # 矢印（ベクトル）の成分
            u = gain * np.cos(pta)
            v = gain * np.sin(pta)
            # 矢印描写
            aAx.quiver(x, y, u, v, color = "red", angles = "xy", scale_units = "xy", scale = 1)

            for obs in obsCrnt:
                obsPose = obs[1]
                lmCovM = self.__mScnSnsr.getLandMarkCovMatrixOnMeasurementSys(obsPose)
                lmCovR = self.__mScnSnsr.tfMeasurement2Robot(lmCovM, obsPose[1])
                Pxy = lmCovR[0:2, 0:2]
                x, y, ang_rad = self.__mEllipse.calc_error_ellipse(Pxy)
                p = ( obsPose[0] * np.cos(obsPose[1]), obsPose[0] * np.sin(obsPose[1]) )
                ell = patches.Ellipse(p, x, y, angle = np.rad2deg(ang_rad), linewidth = 2, alpha = 0.2,
                             facecolor = 'yellow', edgecolor = 'black', label = 'Error Ellipse: %.2f[%%]' %
                             self.__mConfidence_interval)
                aAx.add_patch(ell)

                # ロボット-ランドマーク間線分描写
                xl = np.array([0, p[0]])
                yl = np.array([0, p[1]])
                aAx.plot(xl, yl, '--', c='green')

        # ロボット描写
        aAx.scatter(0, 0, s = 100, c = "blue", marker = "o", alpha = 0.5, label = "Robot")
        aAx.quiver(0, 0, 0, 1, color = "blue", angles = "xy", scale_units = "xy", scale = 1)



# スキャンセンサモデル
SCN_SENS_RANGE_m = 10.0  # 走査距離[m]
SCN_SENS_ANGLE_rps = np.deg2rad(70.0)  # 走査角度[rad]
RADIUS_m = 10.0  # 周回半径[m]

# ロボット動作モデル
OMEGA_rps = np.deg2rad(10.0)  # 角速度[rad/s]
VEL_mps = RADIUS_m * OMEGA_rps  # 速度[m/s]

# ランドマーク
LAND_MARKS = np.array([[ 0.0, 10.0],
                       [ 2.0, -3.0],
                       [ 5.0, 5.0],
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
    gRbt.estimateOpticalTrajectory()

    plt.cla()

    # サブプロットを追加
    ax1 = plt.subplot2grid((1, 2), (0, 0), aspect = "equal", adjustable = "box-forced")
    ax2 = plt.subplot2grid((1, 2), (0, 1), aspect = "equal", adjustable = "box-forced")

    gRbt.draw(ax1, ax2)

    print("time:{0:.3f}[s] x = {1:.3f}[m], y = {2:.3f}[m], θ = {3:.3f}[deg]".format(time_s, x[0, 0], x[1, 0], np.rad2deg(x[2, 0])))

    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("World")
    ax1.axis("equal", adjustable = "box")
    ax1.grid()
    ax1.legend(fontsize = 10)




    # データ数
    data_num = 1000

    # 平均
    mu = np.array([[0.0],
                   [0.0]])
    # 共分散
    cov_x = 8.00
    cov_y = 1.00
    cov_t = 1.0
    cov = np.array([[ cov_x, 0.0  , 0.0  ],
                    [ 0.0  , cov_y, 0.0  ],
                    [ 0.0  , 0.0  , cov_t]])

#    P = np.random.multivariate_normal([0.0, 0.0, 0.0], cov, data_num).T
#    scn = ScanSensor(0, 0, LAND_MARKS)

#    aRad = np.deg2rad(45)
#    cov2 = scn.rotateCovariance(cov, aRad - tf.BASE_ANG)
#    P2 = np.random.multivariate_normal([0.0, 0.0, 0.0], cov2, data_num).T
    # 散布図をプロットす
#    ax2.scatter(P[0], P[1], color='r', marker='x', label="$K_1$")
#    ax2.scatter(P2[0], P2[1], color='b', marker='x', label="$K_1$")



    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Robot")
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

