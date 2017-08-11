#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================
# brief        Graph-based SLAM
#
# author       Takuya Niibori
# attention    none
#==============================================================================

import numpy.matlib
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools
from mylib import error_ellipse
from mylib import limit
from mylib import transform as tf
import motion_model as mm
from matplotlib import animation, patches

class ScanSensor(object):
    """スキャンセンサclass"""

    __DIST_NOISE = 0.001  # ランドマーク距離雑音[%]
    __R_DIST = __DIST_NOISE / 100  # ランドマーク距離雑音ゲイン
    __R_DIR_SIGMA = 0.0000001  # ランドマーク観測方向雑音標準偏差[rad]
    __R_ORIENT_SIGMA = 0.0000001  # ランドマーク向き雑音標準偏差[rad]

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
#        self.__DIST_NOISE = 10  # ランドマーク距離雑音[%]
#        self.__R_DIST = self.__DIST_NOISE / 100  # ランドマーク距離雑音ゲイン
#        self.__R_DIR_SIGMA = 3 * np.pi / 180  # ランドマーク観測方向雑音標準偏差[rad]
#        self.__R_ORIENT_SIGMA = 3 * np.pi / 180  # ランドマーク向き雑音標準偏差[rad]


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

    @classmethod
    def getLandMarkCovMatrixOnMeasurementSys(cls, aLandMark):
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
        dist = aLandMark[0] * ScanSensor.__R_DIST
        dir_cov = (aLandMark[0] * np.sin(ScanSensor.__R_DIR_SIGMA)) ** 2
        orient_cov = ScanSensor.__R_DIR_SIGMA ** 2 + ScanSensor.__R_ORIENT_SIGMA ** 2
        covMat = np.array([[dist ** 2, 0, 0         ],
                           [0, dir_cov, 0         ],
                           [0, 0, orient_cov]])

        return covMat

    @classmethod
    def tfMeasurement2World(cls, aCovMat, aLandMarkDir, aRobotDir):
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

    def getLandMarkNum(self):
        return self.__mLandMarksNum


class Observation(object):
    """観測class"""
    def __init__(self, aTime, aLandMarkId, aLandMarkPose, aRobotPoseId):
        self.mTime = aTime
        self.mLandMarkId = aLandMarkId
        self.mLandMarkPose = aLandMarkPose
        self.mRobotPoseId = aRobotPoseId


class Edge(object):

    def __init__(self, aTimeBfr, aTimeAft, aLandMarkId, aMatH_PP, aMatH_PC, aMatH_CP, aMatH_CC, aVecB_P, aVecB_C):
        self.mTimeBfr = aTimeBfr
        self.mTimeAft = aTimeAft
        self.mLandMarkId = aLandMarkId
        self.mMatH_PrevPrev = aMatH_PP
        self.mMatH_PrevCrnt = aMatH_PC
        self.mMatH_CrntPrev = aMatH_CP
        self.mMatH_CrntCrnt = aMatH_CC
        self.mVecB_Prev = aVecB_P
        self.mVecB_Crnt = aVecB_C


class TrajectoryEstimator(object):
    """軌跡推定器class"""


    def __init__(self):

        self.__mEdge = []

        # 空の情報行列と情報ベクトルを作成
        self.__mMatH = np.zeros((3, 3))
        self.__mVecB = np.zeros((3, 1))
        self.__KeepLandMarkId = []
        self.__KeepLandMarkTime = []

    def setPairObs(self, aObs1, aObs2, aRobotPose):
        landMarkId = aObs1.mLandMarkId

        if aObs1.mTime > aObs2.mTime:
            timeAft = aObs1.mTime
            timeBfr = aObs2.mTime
            obsPoseAft = aObs1.mLandMarkPose
            obsPoseBfr = aObs2.mLandMarkPose
            rbtPoseAft = aRobotPose[aObs1.mRobotPoseId]
            rbtPoseBfr = aRobotPose[aObs2.mRobotPoseId]
        else:
            timeAft = aObs2.mTime
            timeBfr = aObs1.mTime
            obsPoseAft = aObs2.mLandMarkPose
            obsPoseBfr = aObs1.mLandMarkPose
            rbtPoseAft = aRobotPose[aObs2.mRobotPoseId]
            rbtPoseBfr = aRobotPose[aObs1.mRobotPoseId]


        # 新規に検出されたランドマーク
        if landMarkId not in self.__KeepLandMarkId:
            self.__KeepLandMarkId.append(landMarkId)
            if timeBfr not in self.__KeepLandMarkTime:
                self.__KeepLandMarkTime.append(timeBfr)

        # 検出された時間を保持
        self.__KeepLandMarkTime.append(timeAft)


        # ロボット推定姿勢によるエッジ(相対姿勢)算出
        relPoseRbt = self.__calcRelativePoseByRobotPose(rbtPoseAft, rbtPoseBfr)

        # 観測結果によるエッジ(相対姿勢)算出
        lmCrntWorld = self.__tfRobot2LandMark(obsPoseAft)
        lmPrevWorld = self.__tfRobot2LandMark(obsPoseBfr)
        relPoseObs = self.__calcRelativePoseByObservation(lmCrntWorld, lmPrevWorld)

        # 姿勢誤差算出
        err = relPoseRbt - relPoseObs
        print("error:ID<{0}>,  x = {1:.3f}[m], y = {2:.3f}[m], θ = {3:.3f}[deg]".format(landMarkId, err[0, 0], err[1, 0], np.rad2deg(err[2, 0])))


        # 計測座標系での情報行列算出
        lmCovCrntM = ScanSensor.getLandMarkCovMatrixOnMeasurementSys(obsPoseAft)
        lmCovCrntW = ScanSensor.tfMeasurement2World(lmCovCrntM, obsPoseAft[1], rbtPoseAft[2, 0])
        lmCovPrevM = ScanSensor.getLandMarkCovMatrixOnMeasurementSys(obsPoseBfr)
        lmCovPrevW = ScanSensor.tfMeasurement2World(lmCovPrevM, obsPoseBfr[1], rbtPoseBfr[2, 0])
        infoMat = np.linalg.inv(lmCovCrntW + lmCovPrevW)

        # ヤコビアン算出
        theta = rbtPoseBfr[2, 0] + obsPoseBfr[1]
        jacobMatPrev = np.array([[-1,  0,  obsPoseBfr[0] * np.sin(theta)],
                                 [ 0, -1, -obsPoseBfr[0] * np.cos(theta)],
                                 [ 0,  0, -1                            ]])
        theta = rbtPoseAft[2, 0] + obsPoseAft[1]
        jacobMatCrnt = np.array([[ 1,  0, -obsPoseAft[0] * np.sin(theta)],
                                 [ 0,  1,  obsPoseAft[0] * np.cos(theta)],
                                 [ 0,  0,  1                            ]])


        self.__mEdge.append(Edge(timeBfr,
                                 timeAft,
                                 landMarkId,
                                 jacobMatPrev.T @ infoMat @ jacobMatPrev,
                                 jacobMatPrev.T @ infoMat @ jacobMatCrnt,
                                 jacobMatCrnt.T @ infoMat @ jacobMatPrev,
                                 jacobMatCrnt.T @ infoMat @ jacobMatCrnt,
                                 jacobMatPrev.T @ infoMat @ err,
                                 jacobMatCrnt.T @ infoMat @ err
                                 ))

        """
        pp = (t-1) * 3
        pc = t * 3
        # 情報行列更新
        self.__matH[pp:pp + 3, pp:pp + 3] += jacobMatPrev.T @ infoMat @ jacobMatPrev
        self.__matH[pp:pp + 3, pc:pc + 3] += jacobMatPrev.T @ infoMat @ jacobMatCrnt
        self.__matH[pc:pc + 3, pp:pp + 3] += jacobMatCrnt.T @ infoMat @ jacobMatPrev
        self.__matH[pc:pc + 3, pc:pc + 3] += jacobMatCrnt.T @ infoMat @ jacobMatCrnt

        # 情報ベクトル更新
        self.__vecB[pp:pp + 3, 0][:, np.newaxis] += jacobMatPrev.T @ infoMat @ err
        self.__vecB[pc:pc + 3, 0][:, np.newaxis] += jacobMatCrnt.T @ infoMat @ err
        """

    def updateInfoMatAndVec(self):
        """"情報行列と情報ベクトルのリサイズ処理
        引数：
            なし
        返り値：
            なし

        """

        leng = (len(self.__KeepLandMarkTime) + 1) * 3

        if leng > 3:

            self.__mMatH = np.zeros((leng, leng))
            self.__mVecB = np.zeros((leng, 1))

            # 昇順でソート
            timeList = sorted(self.__KeepLandMarkTime)

            for edg in self.__mEdge:
                pp = (timeList.index(edg.mTimeBfr) + 1) * 3
                pc = (timeList.index(edg.mTimeAft) + 1) * 3


                # 情報行列更新
                self.__mMatH[pp:pp + 3, pp:pp + 3] += edg.mMatH_PrevPrev
                self.__mMatH[pp:pp + 3, pc:pc + 3] += edg.mMatH_PrevCrnt
                self.__mMatH[pc:pc + 3, pp:pp + 3] += edg.mMatH_CrntPrev
                self.__mMatH[pc:pc + 3, pc:pc + 3] += edg.mMatH_CrntCrnt

                # 情報ベクトル更新
                self.__mVecB[pp:pp + 3, 0][:, np.newaxis] += edg.mVecB_Prev
                self.__mVecB[pc:pc + 3, 0][:, np.newaxis] += edg.mVecB_Crnt


            # クリア
            self.__KeepLandMarkId = []
            self.__KeepLandMarkTime = []


    def __calcRelativePoseByRobotPose(self, aPoseCrnt, aPosePrev):
        """ロボットの推定姿勢による、相対姿勢算出
        引数：
            aPoseCrnt：ロボット現在姿勢
                aPoseCrnt[0, 0]：x座標[m]
                aPoseCrnt[1, 0]：y座標[m]
                aPoseCrnt[2, 0]：方角(rad)
            aPosePrev：ロボット過去姿勢
                aPoseCrnt[0, 0]：x座標[m]
                aPoseCrnt[1, 0]：y座標[m]
                aPoseCrnt[2, 0]：方角(rad)
        返り値：
            rel：相対ロボット姿勢
                rel[0, 0]：x座標[m]
                rel[1, 0]：y座標[m]
                rel[2, 0]：方角(rad)
        """
        rel = aPoseCrnt - aPosePrev

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

    def __calcRelativePoseByObservation(self, aObsPoseCrnt, aObsPosePrev):
        """観測結果による、相対姿勢算出
        引数：
            aObsPoseCrnt：ロボット現在姿勢
                aObsPoseCrnt[0]：ユークリッド距離
                aObsPoseCrnt[1]：観測方向
                aObsPoseCrnt[2]：ランドマーク向き
            aObsPosePrev：ロボット過去姿勢
                aObsPosePrev[0]：ユークリッド距離
                aObsPosePrev[1]：観測方向
                aObsPosePrev[2]：ランドマーク向き
        返り値：
            rel：相対ロボット姿勢
                rel[0, 0]：x座標[m]
                rel[1, 0]：y座標[m]
                rel[2, 0]：方角(rad)
        """
        px = aObsPoseCrnt[0] * np.cos(aObsPoseCrnt[1]) - aObsPosePrev[0] * np.cos(aObsPosePrev[1])
        py = aObsPoseCrnt[0] * np.sin(aObsPoseCrnt[1]) - aObsPosePrev[0] * np.sin(aObsPosePrev[1])
        pt = aObsPoseCrnt[2] - aObsPosePrev[2]

        rel = np.array([[px],
                        [py],
                        [pt]])
        return rel



    @classmethod
    def clearInfoMatAndVec(cls):
        """"情報行列と情報ベクトルのクリア処理
        引数：
            なし
        返り値：
            なし

        """
        """
        length = Edge.__mCnt
        Edge.__mMatH = np.zeros((length, length))
        Edge.__mVecB = np.zeros((length, 1))

        Edge.__mMatH[0:3,0:3] += np.identity(3)*10000



#        if aObsCrnt[1] == aObsPrev[1]
#            self.__mLandMarkId = aObsCrnt[1]

#            self.id1, self.id2 = obs1.pose_id, obs2.pose_id          # どの2つの姿勢なのかを記録（それぞれ姿勢1、姿勢2と呼
        """

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
        self.__mTrjEst = TrajectoryEstimator()

        #---------- 制御周期 ----------
        self.__mDt = aDt
        self.__mTime = 0

        #---------- 姿勢 ----------
        self.__mPosesActu = [aPose]  # 姿勢（実際値）
        self.__mPosesGues = [aPose]  # 姿勢（推定値）
        #---------- 制御 ----------
        self.__mCtr = []
        #---------- 観測 ----------
        self.__mObsActu = []
        self.__mObsTrue = []
        self.__mHalfEdges = []

        #---------- 情報行列 ----------

        # 空の情報行列と情報ベクトルを作成
        self.__matH = np.zeros((3, 3))
        self.__vecB = np.zeros((3, 1))

        # 誤差楕円の信頼区間[%]
        self.__mConfidence_interval = 99.0
        self.__mEllipse = error_ellipse.ErrorEllipse(self.__mConfidence_interval)

    def __resizeInfoMatAndVec(self):
        """"情報行列と情報ベクトルのリサイズ処理
        引数：
            なし
        返り値：
            なし

        """

        length = len(self.__mPosesGues) * 3

        self.__matH = np.zeros((length, length))
        self.__vecB = np.zeros((length, 1))

        self.__matH[0:3,0:3] += np.identity(3)*10000


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
#        self.__resizeInfoMatAndVec()


        obsActuCrnt, obsTrueCrnt = self.__mScnSnsr.scan(poseActu)

        for obs in obsActuCrnt:
            self.__mHalfEdges.append(Observation(self.__mTime, obs[0], obs[1], len(self.__mPosesGues)-1))

        self.__mObsActu.append(obsActuCrnt)  # 観測結果
        self.__mObsTrue.append(obsTrueCrnt)  # 観測結果

        self.__mTime += 1


    def estimateOpticalTrajectory(self):


        lm_num = self.__mScnSnsr.getLandMarkNum()
        for i in range(lm_num):
            heobj_list = list(filter(lambda obj: obj.mLandMarkId == i, self.__mHalfEdges))
            pair = list(itertools.combinations(heobj_list,2))
            for p in pair:
                self.__mTrjEst.setPairObs(p[0], p[1], self.__mPosesGues)

        # 情報行列と情報ベクトル更新
        self.__mTrjEst.updateInfoMatAndVec()

        """
        pos_edges = []
        lm_num = self.__mScnSnsr.getLandMarkNum()
        time_list_all = []
        for i in range(lm_num):
            time_list = []
            heobj_list = list(filter(lambda obj: obj.mLandMarkId == i, self.__mHalfEdges))
            if len(heobj_list)>0:
                for heobj in heobj_list:
                    time_list.append(heobj.mTime)
                del time_list[time_list.index(min(time_list))]
            time_list_all.extend(time_list)

            pair = list(itertools.combinations(heobj_list,2))
            for p in pair:
                pos_edges.append(Edge(p[0], p[1], self.__mPosesGues)

        # 重複削除
        time_list_all = list(set(time_list_all))
        """

        """
        if len(obsActuCrnt) > 0 and len(self.__mObsActuPrev):
            for j in range(len(obsActuCrnt)):
                for k in range(len(self.__mObsActuPrev)):
                    if obsActuCrnt[j][0] == self.__mObsActuPrev[k][0]:
                        self.__Edges.append(Edge(self.__mTime, obsActuCrnt, self.__mObsActuPrev, poseGues, self.__mPosesGuesPrev))




        print("[軌跡推定]Ctr:{0}, PosesActu = {1}, PosesGues = {2}, Obs = {3}".format(len(self.__mCtr), len(self.__mPosesActu),
                                                                            len(self.__mPosesGues), len(self.__mObsActu)))

        obsPrev = []
        poseBfr = []
        for t, (obsCrnt, poseAft) in enumerate(zip(self.__mObsActu, self.__mPosesGues)):
            infoMat = []
            err = []
            if len(obsCrnt) > 0 and len(obsPrev) > 0:

                # ロボット推定姿勢によるエッジ(相対姿勢)算出
                relPoseRbt = self.__calcRelativePoseByRobotPose(poseAft, poseBfr)

                for j in range(len(obsCrnt)):
                    for k in range(len(obsPrev)):
                        if obsCrnt[j][0] == obsPrev[k][0]:
                            obsPoseAft = obsCrnt[j][1]
                            obsPoseBfr = obsPrev[k][1]

                            # 観測結果によるエッジ(相対姿勢)算出
                            lmCrntWorld = self.__tfRobot2LandMark(obsPoseAft)
                            lmPrevWorld = self.__tfRobot2LandMark(obsPoseBfr)
                            relPoseObs = self.__calcRelativePoseByObservation(lmCrntWorld, lmPrevWorld)

                            # 姿勢誤差算出
                            err = relPoseRbt - relPoseObs
                            print("error:ID<{0}>,  x = {1:.3f}[m], y = {2:.3f}[m], θ = {3:.3f}[deg]".format(obsCrnt[j][0], err[0, 0], err[1, 0], np.rad2deg(err[2, 0])))


                            # 計測座標系での情報行列算出
                            lmCovCrntM = self.__mScnSnsr.getLandMarkCovMatrixOnMeasurementSys(obsPoseAft)
                            lmCovCrntW = self.__mScnSnsr.tfMeasurement2World(lmCovCrntM, obsPoseAft[1], poseAft[2, 0])
                            lmCovPrevM = self.__mScnSnsr.getLandMarkCovMatrixOnMeasurementSys(obsPoseBfr)
                            lmCovPrevW = self.__mScnSnsr.tfMeasurement2World(lmCovPrevM, obsPoseBfr[1], poseBfr[2, 0])
                            infoMat = np.linalg.inv(lmCovCrntW + lmCovPrevW)

                            # ヤコビアン算出
                            theta = poseBfr[2, 0] + obsPoseBfr[1]
                            jacobMatPrev = np.array([[-1,  0,  obsPoseBfr[0] * np.sin(theta)],
                                                     [ 0, -1, -obsPoseBfr[0] * np.cos(theta)],
                                                     [ 0,  0, -1                             ]])
                            theta = poseAft[2, 0] + obsPoseAft[1]
                            jacobMatCrnt = np.array([[ 1,  0, -obsPoseAft[0] * np.sin(theta)],
                                                     [ 0,  1,  obsPoseAft[0] * np.cos(theta)],
                                                     [ 0,  0,  1                             ]])

                            pp = (t-1) * 3
                            pc = t * 3
                            # 情報行列更新
                            self.__matH[pp:pp + 3, pp:pp + 3] += jacobMatPrev.T @ infoMat @ jacobMatPrev
                            self.__matH[pp:pp + 3, pc:pc + 3] += jacobMatPrev.T @ infoMat @ jacobMatCrnt
                            self.__matH[pc:pc + 3, pp:pp + 3] += jacobMatCrnt.T @ infoMat @ jacobMatPrev
                            self.__matH[pc:pc + 3, pc:pc + 3] += jacobMatCrnt.T @ infoMat @ jacobMatCrnt

                            # 情報ベクトル更新
                            self.__vecB[pp:pp + 3, 0][:, np.newaxis] += jacobMatPrev.T @ infoMat @ err
                            self.__vecB[pc:pc + 3, 0][:, np.newaxis] += jacobMatCrnt.T @ infoMat @ err

            else:
                print("なし")

            obsPrev = obsCrnt
            poseBfr = poseAft

        if t > 3:
            delta = - np.linalg.inv(self.__matH) @ self.__vecB
        """









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


    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Robot")
    ax2.axis([-15, 15, -15, 15])
    ax2.grid()
    ax2.legend(fontsize = 10)

if __name__ == "__main__":


    frame_cnt = int(36 * 1000 / PERIOD_ms)

    fig = plt.figure(figsize = (18, 9))

    ani = animation.FuncAnimation(fig, graph_based_slam, frames = frame_cnt, fargs = (PERIOD_ms,), blit = False,
                                  interval = PERIOD_ms, repeat = False)

    # ani.save("Localization_by_pf.mp4", bitrate=6000)

    plt.show()

