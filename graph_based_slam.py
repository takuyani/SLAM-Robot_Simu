#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================
# brief        Graph-based SLAM
#
# author       Takuya Niibori
# attention    none
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import itertools
from mylib import error_ellipse
from mylib import limit
from mylib import transform as tf
import motion_model as mm
from matplotlib import animation, patches
from copy import deepcopy

class Observation:
    """観測class"""

    def __init__(self, aLandMarkId, aDist_m, aDir_rad, aOrient_rad):
        """"コンストラクタ
        引数：
            aLandMarkId：ランドマーク識別ID
            aDist_m：ランドマーク距離[m]
            aDir_rad：ランドマーク観測方向[rad]
            aOrient_rad：ランドマーク向き[rad]
        """
        self.__mLandMarkId = aLandMarkId
        self.__mDist_m = aDist_m
        self.__mDir_rad = aDir_rad
        self.__mOrient_rad = aOrient_rad

    def getLandMarkId(self):
        """"ランドマーク識別ID取得
        引数：
            なし
        戻り値：
            ランドマーク識別ID
        """
        return self.__mLandMarkId

    def getDist(self):
        """"ランドマーク距離取得
        引数：
            なし
        戻り値：
            ランドマーク距離[m]
        """
        return self.__mDist_m

    def getDir(self):
        """"ランドマーク観測方向取得
        引数：
            なし
        戻り値：
            ランドマーク観測方向[rad]
        """
        return self.__mDir_rad

    def getOrient(self):
        """"ランドマーク向き取得
        引数：
            なし
        戻り値：
            ランドマーク向き
        """
        return self.__mOrient_rad


class ScanSensor(object):
    """スキャンセンサclass
        扇型にスキャンできるセンサを想定。
    """

    # 観測雑音定義(デフォルト値)
    __R_Dist = 10 / 100  # ランドマーク距離雑音ゲイン
    __R_DirSigma = np.deg2rad(3.0)  # ランドマーク観測方向雑音標準偏差[rad]
    __R_OrientSigma = np.deg2rad(3.0)  # ランドマーク向き雑音標準偏差[rad]

    def __init__(self, aRange_m, aAngle_rad, aLandMarks):
        """"コンストラクタ
        引数：
            aRange_m：走査距離[m]
            aAngle_rad：走査角度[rad]
            aLandMarks：ランドマーク
                        [[1番目LMのX座標, 1番目LMのY座標]
                         [2番目LMのX座標, 2番目LMのY座標]
                                         ：
                         [n番目LMのX座標, n番目LMのY座標]]
        """
        self.__mScanRange_m = aRange_m
        self.__mScanAngle_rad = aAngle_rad
        self.__mReslAng = int(np.rad2deg(aAngle_rad))
        self.__mLandMarks = aLandMarks

        ang = np.rad2deg(aAngle_rad)
        ofs = np.rad2deg(tf.BASE_ANG)

        step = 1.0  # スキャン測定範囲の円弧を描くための間隔[deg]
        self.__p = []
        self.__p.append([aRange_m * np.cos(np.deg2rad(x)) for x  in np.arange(-ang + ofs, ang + ofs + step, step)])
        self.__p.append([aRange_m * np.sin(np.deg2rad(x)) for x  in np.arange(-ang + ofs, ang + ofs + step, step)])
        self.__p[0].append(0.0)
        self.__p[1].append(0.0)
        self.__p[0].append(self.__p[0][0])
        self.__p[1].append(self.__p[1][0])

        self.__local = np.array([self.__p[0],
                                 self.__p[1]])


    def setNoiseParam(self, aDist, aDirSigma, aOrientSigma):
        """"ノイズパラメータ設定
        引数：
            aDist：ランドマーク距離雑音[%]
            aDirSigma：ランドマーク観測方向雑音標準偏差[deg]
            aOrientSigma：ランドマーク向き雑音標準偏差[deg]
        戻り値：
           なし
        """
        ScanSensor.__R_Dist = aDist / 100  # ランドマーク距離雑音ゲイン
        ScanSensor.__R_DirSigma = np.deg2rad(aDirSigma)  # ランドマーク観測方向雑音標準偏差[rad]
        ScanSensor.__R_OrientSigma = np.deg2rad(aOrientSigma)  # ランドマーク向き雑音標準偏差[rad]

    def scan(self, aRobotPose):
        """"スキャン結果
            ノイズ有りと無しの2つの観測結果を返却する
        引数：
            aRobotPose：ロボット姿勢
               aRobotPose[0, 0]：x座標[m]
               aRobotPose[1, 0]：y座標[m]
               aRobotPose[2, 0]：方角(rad)
        戻り値：
            obsWithNoise：ノイズ有り観測結果
                            [[観測classインスタンス(1)]
                             [観測classインスタンス(2)]
                                             ：
                             [観測classインスタンス(n)]]
            obsWithoutNoise：ノイズ無し観測結果
                            [[観測classインスタンス(1)]
                             [観測classインスタンス(2)]
                                             ：
                             [観測classインスタンス(n)]]
        """
        obsWithNoise = []
        obsWithoutNoise = []
        robotLandMarks = tf.world2robot(aRobotPose, self.__mLandMarks)  # 世界座標系→ロボット座標系変換
        distLm = np.linalg.norm(robotLandMarks, axis=1)  # ランドマーク距離算出
        dirLm_rad = np.arctan2(robotLandMarks[:, 1], robotLandMarks[:, 0])  # ランドマーク観測方向算出
        orientLm_rad = np.ones(robotLandMarks.shape[0]) * (tf.BASE_ANG - aRobotPose[2, 0])  # ランドマーク向き算出

        # センサの測定範囲内にランドマークが存在するか否かの判定
        scanRad = tf.BASE_ANG - self.__mScanAngle_rad
        obsDetectFlg = [ True
                          if (distLm[i] <= self.__mScanRange_m and (robotLandMarks[i, 1] >= np.absolute(robotLandMarks[i, 0]) * np.tan(scanRad)))
                          else False
                          for i in range(len(robotLandMarks))]

        for i, flg in enumerate(obsDetectFlg):
            if (flg == True):
                # ノイズ付与
                distActu = np.random.normal(distLm[i], distLm[i] * ScanSensor.__R_Dist)
                dirActu = limit.limit_angle(np.random.normal(dirLm_rad[i], ScanSensor.__R_DirSigma))
                orientActu = limit.limit_angle(np.random.normal(orientLm_rad[i], ScanSensor.__R_OrientSigma))
                # 観測結果格納
                obsWithNoise.append(Observation(i, distActu, dirActu, orientActu))
                obsWithoutNoise.append(Observation(i, distLm[i], dirLm_rad[i], orientLm_rad[i]))

        return obsWithNoise, obsWithoutNoise


    @classmethod
    def getLandMarkCovMatrixOnMeasurementSys(cls, aLandMarkDist):
        """計測座標系におけるランドマークの共分散行列取得
        引数：
            aLandMarkDist：ランドマーク距離
        戻り値：
            covMat：計測座標系での共分散行列(3×3)
                対角成分
                1：x軸の共分散
                2：y軸の共分散
                3：θ方向の共分散
        """
        dist = aLandMarkDist * ScanSensor.__R_Dist
        dir_cov = (aLandMarkDist * np.sin(ScanSensor.__R_DirSigma)) ** 2
        orient_cov = ScanSensor.__R_DirSigma ** 2 + ScanSensor.__R_OrientSigma ** 2
        covMat = np.array([[dist ** 2, 0, 0         ],
                           [        0, dir_cov, 0         ],
                           [        0, 0, orient_cov]])

        return covMat

    @classmethod
    def tfMeasurement2World(cls, aCovMat, aLandMarkDir, aRobotDir):
        """計測座標系→世界座標系変換
        引数：
            aCovMat：共分散行列
            aLandMarkDir：ロボット座標系でのランドマーク観測方向[rad]
            aRobotDir：世界座標系でのロボット方角[rad]
        戻り値：
            covMatWorld：世界座標系での共分散行列(3×3)
        """
        ang = aLandMarkDir + aRobotDir - tf.BASE_ANG
        c = np.cos(ang)
        s = np.sin(ang)
        rotMat = np.array([[c, -s, 0],
                           [s, c, 0],
                           [0, 0, 1]])

        covMatWorld = rotMat @ aCovMat @ rotMat.T

        return covMatWorld


    def tfMeasurement2Robot(self, aCovMat, aLandMarkDir):
        """計測座標系→ロボット座標系変換
        引数：
            aCovMat：共分散行列
            aLandMarkDir：ロボット座標系でのランドマーク観測方向[rad]
        戻り値：
            covMatRobot：ロボット座標系での共分散行列(3×3)
        """
        c = np.cos(aLandMarkDir)
        s = np.sin(aLandMarkDir)
        rotMat = np.array([[c, -s, 0],
                           [s, c, 0],
                           [0, 0, 1]])

        covMatRobot = rotMat @ aCovMat @ rotMat.T

        return covMatRobot

    def draw(self, aAx, aColor, aPose):
        """"描写
        引数：
            aAx：描写位置
            aColor：描写色
            aPose：姿勢
               aPose[0, 0]：x座標[m]
               aPose[1, 0]：y座標[m]
               aPose[2, 0]：方角(rad)
        """
        world = tf.robot2world(aPose, self.__local.T)
        aAx.plot(world.T[0], world.T[1], c=aColor, linewidth=1.0, linestyle="-")

        # ランドマークの描写
        aAx.scatter(self.__mLandMarks[:, 0], self.__mLandMarks[:, 1], s=100, c="yellow", marker="*", alpha=0.5, linewidths="2", edgecolors="orange", label="Land Mark(True)")

    def getLandMarkNum(self):
        """"ランドマーク個数取得
        引数：
            なし
        戻り値：
            ランドマーク個数
        """
        return len(self.__mLandMarks)


class HalfEdge(object):
    """ハーフエッジclass"""

    def __init__(self, aTime, aObs, aRobotPoseId):
        """"コンストラクタ
        引数：
            aTime：時刻
            aObs：Observationインスタンス
            aRobotPoseId：ロボット姿勢識別ID
        """
        self.__mTime = aTime
        self.__mObs = aObs
        self.__mRobotPoseId = aRobotPoseId

    def getTime(self):
        """"時刻取得
        引数：
            なし
        戻り値：
            時刻
        """
        return self.__mTime

    def getObs(self):
        """"Observationインスタンス取得
        引数：
            なし
        戻り値：
            Observationインスタンス
        """
        return self.__mObs

    def getRobotPoseId(self):
        """"ロボット姿勢識別ID取得
        引数：
            なし
        戻り値：
            ロボット姿勢識別ID
        """
        return self.__mRobotPoseId


class Edge(object):
    """エッジclass"""

    def __init__(self, aTimeBfr, aTimeAft, aLandMarkId, aMatH_BB, aMatH_BA, aMatH_AB, aMatH_AA, aVecB_B, aVecB_A):
        """"コンストラクタ
        引数：
            aTimeBfr：時刻(前)[s]
            aTimeAft：時刻(後)[s]
            aLandMarkId：ランドマーク識別ID
            aMatH_BB：情報行列(BforeJacb.T・情報行列Ω・BforeJacb)
            aMatH_BA：情報行列(BforeJacb.T・情報行列Ω・AfterJacb)
            aMatH_AB：情報行列(AfterJacb.T・情報行列Ω・BforeJacb)
            aMatH_AA：情報行列(AfterJacb.T・情報行列Ω・AfterJacb)
            aVecB_B：情報ベクトル(BeforeJacb.T・情報行列Ω・err)
            aVecB_A：情報ベクトル(AfterJacb.T・情報行列Ω・err)
        """
        self.mTimeBfr = aTimeBfr
        self.mTimeAft = aTimeAft
        self.mLandMarkId = aLandMarkId
        self.mMatH_BfrBfr = aMatH_BB
        self.mMatH_BfrAft = aMatH_BA
        self.mMatH_AftBfr = aMatH_AB
        self.mMatH_AftAft = aMatH_AA
        self.mVecB_Bfr = aVecB_B
        self.mVecB_Aft = aVecB_A


class TrajectoryEstimator(object):
    """軌跡推定器class"""

    def __init__(self, aPose):
        """"コンストラクタ
        引数：
            aPose：ロボット姿勢
                aPose[0, 0]：x座標[m]
                aPose[1, 0]：y座標[m]
                aPose[2, 0]：方角(rad)
        """
        self.__mPosesEst = [aPose]
        self.__mIsObs = [True]

        self.__mEdge = []

        # 空の情報行列と情報ベクトルを作成
        self.__mMatH = np.zeros((3, 3))
        self.__mVecB = np.zeros((3, 1))
        self.__KeepLandMarkId = []
        self.__KeepLandMarkTime = []

    def addPose(self, aPose, aIsObs):
        """"ロボット姿勢追加
        引数：
            aPose：ロボット姿勢
            aIsObs：観測可否フラグ
        """
        self.__mPosesEst.append(aPose)
        self.__mIsObs.append(aIsObs)

    def setPairObs(self, aHalfEdge1, aHalfEdge2):
        """"観測ペア設定
        引数：
            aHalfEdge1：ハーフエッジインスタンス1
            aHalfEdge2：ハーフエッジインスタンス2
        """
        landMarkId = aHalfEdge1.getObs().getLandMarkId()

        # ２つの引数のインスタンスにおいて、どちらの時刻が前、後であるかを判定する
        if aHalfEdge1.getTime() > aHalfEdge2.getTime():
            timeAft = aHalfEdge1.getTime()
            timeBfr = aHalfEdge2.getTime()
            obsPoseAft = aHalfEdge1.getObs()
            obsPoseBfr = aHalfEdge2.getObs()
            rbtPoseAft = self.__mPosesEst[aHalfEdge1.getRobotPoseId()]
            rbtPoseBfr = self.__mPosesEst[aHalfEdge2.getRobotPoseId()]
        else:
            timeAft = aHalfEdge2.getTime()
            timeBfr = aHalfEdge1.getTime()
            obsPoseAft = aHalfEdge2.getObs()
            obsPoseBfr = aHalfEdge1.getObs()
            rbtPoseAft = self.__mPosesEst[aHalfEdge2.getRobotPoseId()]
            rbtPoseBfr = self.__mPosesEst[aHalfEdge1.getRobotPoseId()]


        # 新規に検出されたランドマーク
        if landMarkId not in self.__KeepLandMarkId:
            self.__KeepLandMarkId.append(landMarkId)

        # 検出された時間を保持
        if timeBfr not in self.__KeepLandMarkTime:
            self.__KeepLandMarkTime.append(timeBfr)
        if timeAft not in self.__KeepLandMarkTime:
            self.__KeepLandMarkTime.append(timeAft)

        # ロボット推定姿勢によるエッジ(相対姿勢)算出
        relPoseRbt = self.__calcRelativePoseByRobotPose(rbtPoseAft, rbtPoseBfr)

        # 観測結果によるエッジ(相対姿勢)算出
        lmCrntWorld = self.__tfRobot2LandMark(obsPoseAft)
        lmPrevWorld = self.__tfRobot2LandMark(obsPoseBfr)
        relPoseObs = self.__calcRelativePoseByObservation(lmCrntWorld, lmPrevWorld)

        # 姿勢誤差算出
        err = relPoseRbt - relPoseObs
        err[2, 0] = limit.limit_angle(err[2, 0])

        # print(" err = {0}".format(np.rad2deg(err[2, 0])))

        # 計測座標系での情報行列算出
        lmCovCrntM = ScanSensor.getLandMarkCovMatrixOnMeasurementSys(obsPoseAft.getDist())
        lmCovCrntW = ScanSensor.tfMeasurement2World(lmCovCrntM, obsPoseAft.getDir(), rbtPoseAft[2, 0])
        lmCovPrevM = ScanSensor.getLandMarkCovMatrixOnMeasurementSys(obsPoseBfr.getDist())
        lmCovPrevW = ScanSensor.tfMeasurement2World(lmCovPrevM, obsPoseBfr.getDir(), rbtPoseBfr[2, 0])
        sumCov = lmCovCrntW + lmCovPrevW
        infoMat = np.linalg.inv(sumCov)

        # ヤコビアン算出
        theta = limit.limit_angle(rbtPoseBfr[2, 0] + obsPoseBfr.getDir())
        jacobMatBfr = np.array([[-1, 0, obsPoseBfr.getDist() * np.sin(theta)],
                                [ 0, -1, -obsPoseBfr.getDist() * np.cos(theta)],
                                [ 0, 0, -1                            ]])
        theta = limit.limit_angle(rbtPoseAft[2, 0] + obsPoseAft.getDir())
        jacobMatAft = np.array([[ 1, 0, -obsPoseAft.getDist() * np.sin(theta)],
                                [ 0, 1, obsPoseAft.getDist() * np.cos(theta)],
                                [ 0, 0, 1                            ]])

        # 情報行列と情報ベクトルを追加する
        self.__mEdge.append(Edge(timeBfr,
                                 timeAft,
                                 landMarkId,
                                 jacobMatBfr.T @ infoMat @ jacobMatBfr,
                                 jacobMatBfr.T @ infoMat @ jacobMatAft,
                                 jacobMatAft.T @ infoMat @ jacobMatBfr,
                                 jacobMatAft.T @ infoMat @ jacobMatAft,
                                 jacobMatBfr.T @ infoMat @ err,
                                 jacobMatAft.T @ infoMat @ err
                                 ))

    def getEstTrajPose(self):
        """"ロボット推定姿勢取得
            観測された時刻のみのロボット推定位置を取得する
        引数：
            なし
        戻り値：
            ロボット推定姿勢
        """
        return [ pe for (i, pe) in enumerate(self.__mPosesEst) if self.__mIsObs[i] == True ]


    def updateEstPose(self):
        """"ロボット推定姿勢の更新
        引数：
            なし
        戻り値：
            is_calc：算出可否判定
            delta_sum：Δxの２乗和
            det：情報行列Hの行列式
            cond：情報行列Hの条件数

        """
        is_calc = False
        delta_sum = 0.0
        det = 0.0
        cond = 0.0
        leng = len(self.__KeepLandMarkTime) * 3

        if leng > 3:

            self.__mMatH = np.zeros((leng, leng))
            self.__mVecB = np.zeros((leng, 1))

            # TODO:後で削除
            self.__mMatH[0:3, 0:3] += np.identity(3) * (10 ** 4)

            # 昇順でソート
            timeList = sorted(self.__KeepLandMarkTime)

            for edg in self.__mEdge:
                pb = timeList.index(edg.mTimeBfr) * 3
                pa = timeList.index(edg.mTimeAft) * 3

                # 情報行列更新
                self.__mMatH[pb:pb + 3, pb:pb + 3] += edg.mMatH_BfrBfr
                self.__mMatH[pb:pb + 3, pa:pa + 3] += edg.mMatH_BfrAft
                self.__mMatH[pa:pa + 3, pb:pb + 3] += edg.mMatH_AftBfr
                self.__mMatH[pa:pa + 3, pa:pa + 3] += edg.mMatH_AftAft

                # 情報ベクトル更新
                self.__mVecB[pb:pb + 3, 0][:, np.newaxis] += edg.mVecB_Bfr
                self.__mVecB[pa:pa + 3, 0][:, np.newaxis] += edg.mVecB_Aft

            det = np.linalg.det(self.__mMatH)
            cond = np.linalg.cond(self.__mMatH)
            if (0.1 < det) and (cond < 10 ** 15):
                inv = np.linalg.inv(self.__mMatH)
                delta = -inv @ self.__mVecB
                for i, tm in enumerate(timeList):
                    self.__mPosesEst[tm][0, 0] += delta[i * 3]
                    self.__mPosesEst[tm][1, 0] += delta[i * 3 + 1]
                    self.__mPosesEst[tm][2, 0] = limit.limit_angle(self.__mPosesEst[tm][2, 0] + delta[i * 3 + 2])
                    # print("delta = {0}, tm = {1}, det = {2}, cond = {3}".format(np.rad2deg(delta[i * 3 + 2]), tm, det, cond))
                delta_sum = float(delta.T @ delta)
                is_calc = True
            else:
                print("can Not calculate trajectory!")

            # クリア
            self.__mEdge = []
            self.__KeepLandMarkId = []
            self.__KeepLandMarkTime = []

        return is_calc, delta_sum, det, cond


    def __calcRelativePoseByRobotPose(self, aPoseAft, aPoseBfr):
        """ロボットの推定姿勢による、相対姿勢算出
        引数：
            aPoseAft：ロボット姿勢(後)
                aPoseAft[0, 0]：x座標[m]
                aPoseAft[1, 0]：y座標[m]
                aPoseAft[2, 0]：方角(rad)
            aPoseBfr：ロボット姿勢(前)
                aPoseAft[0, 0]：x座標[m]
                aPoseAft[1, 0]：y座標[m]
                aPoseAft[2, 0]：方角(rad)
        戻り値：
            rel：相対ロボット姿勢
                rel[0, 0]：x座標[m]
                rel[1, 0]：y座標[m]
                rel[2, 0]：方角(rad)
        """
        rel = aPoseAft - aPoseBfr
        rel[2, 0] = limit.limit_angle(rel[2, 0])

        return rel

    def __tfRobot2LandMark(self, aObs):
        """ロボット座標系→ランドマーク世界座標系変換
            ロボットを原点とした座標系からランドマークを原点とした世界座標系に変換し、
            変換後のロボット姿勢を戻り値として返す。
        引数：
            aObs：Observationインスタンス
        戻り値：
            robot：ランドマークを原点とした世界座標系でのロボット姿勢
                robot[0]：ユークリッド距離
                robot[1]：観測方向
                robot[2]：ランドマーク向き
        """
        dist = aObs.getDist()
        direct = limit.limit_angle(np.pi + aObs.getDir() - aObs.getOrient())
        orient = limit.limit_angle(tf.BASE_ANG - aObs.getOrient())
        robot = [dist, direct, orient]
        return robot

    def __calcRelativePoseByObservation(self, aObsPoseAft, aObsPoseBfr):
        """観測結果による、相対姿勢算出
        引数：
            aObsPoseAft：ロボット姿勢(後)
                aObsPoseAft[0]：ユークリッド距離
                aObsPoseAft[1]：観測方向
                aObsPoseAft[2]：ランドマーク向き
            aObsPoseBfr：ロボット姿勢(前)
                aObsPoseBfr[0]：ユークリッド距離
                aObsPoseBfr[1]：観測方向
                aObsPoseBfr[2]：ランドマーク向き
        戻り値：
            rel：相対ロボット姿勢
                rel[0, 0]：x座標[m]
                rel[1, 0]：y座標[m]
                rel[2, 0]：方角(rad)
        """
        px = aObsPoseAft[0] * np.cos(aObsPoseAft[1]) - aObsPoseBfr[0] * np.cos(aObsPoseBfr[1])
        py = aObsPoseAft[0] * np.sin(aObsPoseAft[1]) - aObsPoseBfr[0] * np.sin(aObsPoseBfr[1])
        pt = limit.limit_angle(aObsPoseAft[2] - aObsPoseBfr[2])

        rel = np.array([[px],
                        [py],
                        [pt]])
        return rel


class Robot(object):
    """ロボットclass"""

    def __init__(self, aPose, aDt, aScanRng_m, aScanAng_rad, aLandMarks):
        """"コンストラクタ
        引数：
            aPose：姿勢
               aPose[0, 0]：x座標[m]
               aPose[1, 0]：y座標[m]
               aPose[2, 0]：方角(rad)
            aDt：演算周期[sec]
            aScanRng_m：走査距離[m]
            aScanAng_rad：走査角度[rad]
            aLandMarks：ランドマーク
                        [[1番目LMのX座標, 1番目LMのY座標]
                         [2番目LMのX座標, 2番目LMのY座標]
                                         ：
                         [n番目LMのX座標, n番目LMのY座標]]
        """
        self.__mScnSnsr = ScanSensor(aScanRng_m, aScanAng_rad, aLandMarks)
        self.__mScnSnsr.setNoiseParam(5, 2, 2)
        self.__mMvMdl = mm.MotionModel(aDt, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
        self.__mTrjEst = TrajectoryEstimator(aPose)

        #---------- 制御周期 ----------
        self.__mDt = aDt
        self.__mTime = 0

        #---------- 姿勢 ----------
        self.__mPosesActu = [aPose]  # 姿勢（実際値）
        #---------- 制御 ----------
        self.__mCtr = []
        #---------- 観測 ----------
        self.__mObsActu = []
        self.__mObsTrue = []
        self.__mHalfEdges = []

        # 誤差楕円の信頼区間[%]
        self.__mConfidence_interval = 99.0
        self.__mEllipse = error_ellipse.ErrorEllipse(self.__mConfidence_interval)

        self.__mScnSnsr.scan(aPose)
        self.__observe(aPose, len(self.__mPosesActu) - 1, self.__mTime)

        # ガウス・ニュートン法による、反復回数を決定する閾値
        #  Δxの２乗和(Δx.T・Δx)の値が閾値以下となったら計算終了
        self.__DELTA_SUM_TH = 0.01

        self.__isCalc = False  # 軌跡算出可否判定
        self.__loopCnt = 0.0  # 計算反復回数
        self.__deltaSum = 0.0  # Δxの２乗和(Δx.T・Δx)
        self.__det = 0.0  # 情報行列Hの行列式
        self.__cond = 0.0  # 情報行列Hの条件数

    def move(self, aV, aW):
        """"移動
        引数：
            aV：速度ν[m/s]
            aW：角速度ω[rad/s]
        戻り値：
            なし
        """
        # 移動
        poseActu = self.__mMvMdl.moveWithNoise(self.__mPosesActu[-1], aV, aW)
        poseTrue = self.__mMvMdl.moveWithoutNoise(self.__mPosesActu[-1], aV, aW)

        # 履歴保持
        self.__mCtr.append(np.array([aV, aW]))  # 制御
        self.__mPosesActu.append(poseActu)

        self.__mTime += 1
        isObs = self.__observe(poseActu, len(self.__mPosesActu) - 1, self.__mTime)
        self.__mTrjEst.addPose(deepcopy(poseTrue), isObs)

    def __observe(self, aPoseActuCrnt, aRobotPoseId, aTime):
        """"観測
        引数：
            aPoseActuCrnt：ロボット姿勢
               aRobotPose[0, 0]：x座標[m]
               aRobotPose[1, 0]：y座標[m]
               aRobotPose[2, 0]：方角(rad)
            aRobotPoseId：ロボット姿勢識別ID
            aTime：時刻
        戻り値：
            観測結果有無フラグ
                True：観測できたランドマーク有り
                False：観測できたランドマーク無し
        """
        obsActuCrnt, obsTrueCrnt = self.__mScnSnsr.scan(aPoseActuCrnt)

        isObs = False
        for obs in obsActuCrnt:
            self.__mHalfEdges.append(HalfEdge(aTime, obs, aRobotPoseId))
            isObs = True

        self.__mObsActu.append(obsActuCrnt)  # 観測結果（実際値）
        self.__mObsTrue.append(obsTrueCrnt)  # 観測結果（真値）

        return isObs


    def estimateOpticalTrajectory(self):
        """"軌跡推定
        引数：
            なし
        戻り値：
            なし
        """
        delta_sum = self.__DELTA_SUM_TH
        loop_cnt = 0
        det = 0
        cond = 0

        while self.__DELTA_SUM_TH <= delta_sum:
            lm_num = self.__mScnSnsr.getLandMarkNum()
            for i in range(lm_num):
                heobj_list = list(filter(lambda obj: obj.getObs().getLandMarkId() == i, self.__mHalfEdges))
                pair = list(itertools.combinations(heobj_list, 2))
                for p in pair:
                    self.__mTrjEst.setPairObs(p[0], p[1])

            # 情報行列と情報ベクトル更新
            is_calc, delta_sum, det, cond = self.__mTrjEst.updateEstPose()
            loop_cnt += 1

            print(" Loop({0}):Δxの２乗和 = {1}, 行列式 = {2}, 条件数 = {3}".format(loop_cnt, delta_sum, det, cond))

        self.__isCalc = is_calc
        self.__loopCnt = loop_cnt
        self.__deltaSum = delta_sum
        self.__det = det
        self.__cond = cond

    def draw(self, aAx1, aAx2):
        """"描写
        引数：
            aAx1：描写位置1
            aAx2：描写位置2
        """
        self.__drawAx1(aAx1)
        self.__drawAx2(aAx2)

    def __drawAx1(self, aAx):
        """"描写(軸1)
        引数：
            aAx：描写位置
        """
        self.__mScnSnsr.draw(aAx, "green", self.__mPosesActu[-1])

        self.__drawPoses(aAx, "red", "Actual Trajectory", self.__mPosesActu)
        self.__drawActualLandMark(aAx)

        estTrajPose = self.__mTrjEst.getEstTrajPose()
        self.__drawPoses(aAx, "blue", "Estimated Trajectory", estTrajPose)

        if self.__isCalc == True:
            kdgMsg = "OK"
        else:
            kdgMsg = "NG"

        # ラベル描写
        txtstr = "<Status>\n Calculated Propriety: %s\n Number of Iterations: %d\n $\sum_{} \, \Delta{x}^T \Delta{x}$: %e\n $det(H)$:%e\n Condition Number:%e" % (kdgMsg, self.__loopCnt, self.__deltaSum, self.__det, self.__cond)

        # these are matplotlib.patch.Patch propertie
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        x, y = 0.01, 0.99
        aAx.text(x, y, txtstr, transform=aAx.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)


    def __drawPoses(self, aAx, aColor, aLabel, aPoses):
        """"ロボット姿勢描写
        引数：
            aAx：描写位置
            aColor：描写色
            aLabel：描写ラベル
            aPoses：ロボット姿勢
        戻り値：
            なし
        """
        for p in aPoses:
            x = p[0, 0]
            y = p[1, 0]
            # 矢印（ベクトル）の成分
            u = np.cos(p[2, 0])
            v = np.sin(p[2, 0])
            # 矢印描写
            aAx.quiver(x, y, u, v, color=aColor, angles="xy", scale_units="xy", scale=1)

        # 軌跡描写
        pxa = [e[0, 0] for e in aPoses]
        pya = [e[1, 0] for e in aPoses]
        aAx.plot(pxa, pya, c=aColor, linewidth=1.0, linestyle="-", label=aLabel)

    def __drawActualLandMark(self, aAx):
        """"実測ランドマーク描写
        引数：
            aAx：描写位置
        戻り値：
            なし
        """
        flag = False
        obsCrnt = self.__mObsActu[-1]
        poseCrnt = self.__mPosesActu[-1]

        lst_px = []
        lst_py = []
        soa = []

        if len(obsCrnt) != 0:
            for obs in obsCrnt:
                obsDist = obs.getDist()
                obsDir = obs.getDir()
                lmCovM = self.__mScnSnsr.getLandMarkCovMatrixOnMeasurementSys(obsDist)
                lmCovW = self.__mScnSnsr.tfMeasurement2World(lmCovM, obsDir, poseCrnt[2, 0])
                Pxy = lmCovW[0:2, 0:2]
                x, y, ang_rad = self.__mEllipse.calc_error_ellipse(Pxy)
                px = (obsDist * np.cos(obsDir + poseCrnt[2, 0] - tf.BASE_ANG)) + poseCrnt[0, 0]
                py = (obsDist * np.sin(obsDir + poseCrnt[2, 0] - tf.BASE_ANG)) + poseCrnt[1, 0]
                p = (px, py)
                # 誤差楕円描写
                if flag == False:
                    ellLbl = "Error Ellipse: %.2f[%%]" % self.__mConfidence_interval
                else:
                    ellLbl = ""
                e = patches.Ellipse(p, x, y, angle=np.rad2deg(ang_rad), linewidth=2, alpha=0.2,
                             facecolor='yellow', edgecolor='black', label=ellLbl)
                aAx.add_patch(e)

                lst_px.append(px)
                lst_py.append(py)

                # ロボット-ランドマーク間線分描写
                ps = poseCrnt[0:2, 0].T
                xl = np.array([ps[0], px])
                yl = np.array([ps[1], py])

                soa.append([ps[0], ps[1], px - ps[0], py - ps[1]])
                aAx.plot(xl, yl, '--', c='green')
                flag = True

            # 実測ランドマーク描写
            aAx.scatter(lst_px, lst_py, s=100, c="red", marker="*", alpha=0.5, linewidths="2", edgecolors="red", label="Land Mark(Actual)")

    def __drawAx2(self, aAx):
        """"描写(軸2)
        引数：
            aAx：描写位置
        """
        gain = 2

        obsCrnt = self.__mObsTrue[-1]
        if len(obsCrnt) != 0:
            pxa = [obs.getDist() * np.cos(obs.getDir()) for obs in obsCrnt]
            pya = [obs.getDist() * np.sin(obs.getDir()) for obs in obsCrnt]
            pta = [obs.getOrient() for obs in obsCrnt]
            # ランドマーク描写
            aAx.scatter(pxa, pya, s=100, c="yellow", marker="*", alpha=0.5, linewidths="2",
                        edgecolors="orange", label="Land Mark(True)")
            x = pxa
            y = pya
            # 矢印（ベクトル）の成分
            u = gain * np.cos(pta)
            v = gain * np.sin(pta)
            # 矢印描写
            aAx.quiver(x, y, u, v, color="orange", angles="xy", scale_units="xy", scale=1)

        obsCrnt = self.__mObsActu[-1]
        if len(obsCrnt) != 0:
            pxa = [obs.getDist() * np.cos(obs.getDir()) for obs in obsCrnt]
            pya = [obs.getDist() * np.sin(obs.getDir()) for obs in obsCrnt]
            pta = [obs.getOrient() for obs in obsCrnt]
            # ランドマーク描写
            aAx.scatter(pxa, pya, s=100, c="red", marker="*", alpha=0.5, linewidths="2",
                        edgecolors="red", label="Land Mark(Actual)")

            x = pxa
            y = pya
            # 矢印（ベクトル）の成分
            u = gain * np.cos(pta)
            v = gain * np.sin(pta)
            # 矢印描写
            aAx.quiver(x, y, u, v, color="red", angles="xy", scale_units="xy", scale=1)

            flag = False
            for obs in obsCrnt:
                obsDist = obs.getDist()
                obsDir = obs.getDir()
                lmCovM = self.__mScnSnsr.getLandMarkCovMatrixOnMeasurementSys(obsDist)
                lmCovR = self.__mScnSnsr.tfMeasurement2Robot(lmCovM, obsDir)
                Pxy = lmCovR[0:2, 0:2]
                x, y, ang_rad = self.__mEllipse.calc_error_ellipse(Pxy)
                p = (obsDist * np.cos(obsDir), obsDist * np.sin(obsDir))

                if flag == False:
                    ellLbl = 'Error Ellipse: %.2f[%%]' % self.__mConfidence_interval
                else:
                    ellLbl = ""
                ell = patches.Ellipse(p, x, y, angle=np.rad2deg(ang_rad), linewidth=2, alpha=0.2,
                             facecolor='yellow', edgecolor='black', label=ellLbl)
                aAx.add_patch(ell)

                # ロボット-ランドマーク間線分描写
                xl = np.array([0, p[0]])
                yl = np.array([0, p[1]])
                aAx.plot(xl, yl, '--', c='green')
                flag = True

        # ロボット描写
        aAx.scatter(0, 0, s=100, c="blue", marker="o", alpha=0.5, label="Robot")
        aAx.quiver(0, 0, 0, 1, color="blue", angles="xy", scale_units="xy", scale=1)



# スキャンセンサモデル
SCN_SENS_RANGE_m = 15.0  # 走査距離[m]
SCN_SENS_ANGLE_rps = np.deg2rad(80.0)  # 走査角度[rad]
RADIUS_m = 10.0  # 周回半径[m]

# ロボット動作モデル
OMEGA_rps = np.deg2rad(10.0)  # 角速度[rad/s]
VEL_mps = RADIUS_m * OMEGA_rps  # 速度[m/s]

# ランドマーク
LAND_MARKS = np.array([[ 0.0, 0.0],
                       [ 14.0, 1.0],
                       [ 9.0, 9.0],
                       [ 0.0, 15.0],
                       [ -11.0, 10.0],
                       [ -14.0, 1.0],
                       [ -10.0, -9.0],
                       [ 0.0, -16.0],
                       [ 10.0, -11.0]])

# アニメーション更新周期[msec]
PERIOD_ms = 2000

x_base = np.array([[10.0],  # x座標[m]
                   [ 0.0],  # y座標[m]
                   [np.deg2rad(90.0)]])  # 方角yaw[deg]

gRbt = Robot(x_base, PERIOD_ms / 1000, SCN_SENS_RANGE_m, SCN_SENS_ANGLE_rps, LAND_MARKS)

time_s = 0

def graph_based_slam(i, aPeriod_ms):
    """"Graph-based SLAM処理
    引数：
        PERIOD_ms：更新周期[msec]
    戻り値：
        なし
    """
    global time_s
    global P1
    global RADIUS_m
    global OMEGA_rps
    global VEL_mps

    print("TIME:{0:.3f}[s]".format(time_s))

    time_s += aPeriod_ms / 1000

    # 動作モデルによる、ロボットの移動
    gRbt.move(VEL_mps, OMEGA_rps)

    # ロボット移動軌跡推定
    gRbt.estimateOpticalTrajectory()

    plt.cla()

    # サブプロットを追加
    ax1 = plt.subplot2grid((1, 2), (0, 0), aspect="equal", adjustable="box-forced")
    ax2 = plt.subplot2grid((1, 2), (0, 1), aspect="equal", adjustable="box-forced")

    gRbt.draw(ax1, ax2)

    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("World System")
    ax1.axis("equal", adjustable="box")
    ax1.grid()
    ax1.legend(fontsize=10)

    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Robot System")
    rng = SCN_SENS_RANGE_m + 5.0
    ax2.axis([-rng, rng, -rng, rng])
    ax2.grid()
    ax2.legend(fontsize=10)

if __name__ == "__main__":

    frame_cnt = int(36 * 1000 / PERIOD_ms)
    fig = plt.figure(figsize=(18, 9))
    ani = animation.FuncAnimation(fig, graph_based_slam, frames=frame_cnt, fargs=(PERIOD_ms,), blit=False,
                                  interval=PERIOD_ms, repeat=False)

    ani.save("graph-based_slam.mp4", bitrate=6000)

    plt.show()

