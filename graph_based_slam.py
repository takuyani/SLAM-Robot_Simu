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
from copy import deepcopy


class Robot(object):
    """ロボットclass"""

    def __init__(self, x, dt):
        """"コンストラクタ
        引数：
            x：姿勢
             x[0, 0]：x座標[m]
             x[1, 0]：y座標[m]
             x[2, 0]：方角(deg)
            dt：演算周期[sec]
        """
        #---------- 制御周期 ----------
        self.__dt = dt

        #---------- 姿勢定義 ----------
        self.__x = np.array([[x[0, 0]],  # x座標[m]
                             [x[1, 0]],  # y座標[m]
                             [x[2, 0]]])  # 方角yaw[deg]

        dum = dt


    def getPose(self):
        """"姿勢取得処理
        引数：
            なし
        返り値：
            x：姿勢

        """
        return deepcopy(self.__x)


    def motionModel(self, v, w):
        """"動作処理
        引数：
            v：速度ν[m/s]
            w：角速度ω[rad/s]
        返り値：
        """
        a = v / w
        b = limit.limit_angle(w * self.__dt)
        yaw = self.__x[2, 0]
        yaw_add = limit.limit_angle(yaw + b)

        self.__x[0, 0] += a * (-np.sin(yaw) + np.sin(yaw_add))
        self.__x[1, 0] += a * (np.cos(yaw) - np.cos(yaw_add))
        self.__x[2, 0]  = yaw_add

def graph_based_slam():
    """"Graph-based SLAM処理
    引数：
        period_ms：更新周期[msec]
    返り値：
        なし
    """

    global gRobot

    #---------- ランドマーク ----------
    """
    LM(j) = [j番目LMのX座標, j番目LMのY座標]
    """
    LM = np.array([[ 5.0, 5.0],
                   [ 2.0, -3.0],
                   [-3.0, 4.0],
                   [-5.0, -1.0],
                   [ 0.0, 0.0]])

    return LM


time_s = 0
P1 = []

def animate(i, period_ms, aRobot):

    global time_s
    global P1

    RADIUS_m = 10.0  # 周回半径[m]

    OMEGA_rps = np.deg2rad(10.0)   # 角速度[rad/s]
    VEL_mps = RADIUS_m * OMEGA_rps  # 速度[m/s]

    col_x_true = 'red'
    time_s += period_ms / 1000

    aRobot.motionModel(VEL_mps, OMEGA_rps)
    x = aRobot.getPose()

    lm = graph_based_slam()

    plt.cla()

    ax1 = plt.subplot2grid((1, 1), (0, 0))

    # ランドマークの描写
    ax1.scatter(lm[:, 0], lm[:, 1], s = 100, c = "yellow", marker = "*", alpha = 0.5, linewidths = "2",
                edgecolors = "orange", label = 'Land Mark')

    # 状態x(真値)の描写
    P1.append(x[0:2, :])
    a, b = np.array(np.concatenate(P1, axis = 1))
    ax1.plot(a, b, c = col_x_true, linewidth = 1.0, linestyle = '-', label = 'Ground Truth')
    # 矢印（ベクトル）の始点
    X = x[0,0]
    Y = x[1,0]
    # 矢印（ベクトル）の成分
    U = np.cos(x[2,0])
    V = np.sin(x[2,0])
    # 矢印描写
    plt.quiver(X, Y, U, V, color = col_x_true, angles = 'xy', scale_units = 'xy', scale = 1)


    print('time:{0:.3f}[s] x = {1:.3f}[m], y = {2:.3f}[m], θ = {3:.3f}[deg]'.format(time_s, x[0,0], x[1,0], np.rad2deg(x[2,0])))

    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Graph-based SLAM')
    ax1.axis('equal', adjustable = 'box')
    ax1.grid()
    ax1.legend(fontsize = 10)



if __name__ == '__main__':

    period_ms = 100  # 更新周期[msec]
    frame_cnt = int(36 * 1000 / period_ms)

    # 描画
    fig = plt.figure(figsize = (12, 9))

    x_base = np.array([[10.0],  # x座標[m]
                       [ 0.0],  # y座標[m]
                       [np.deg2rad(90.0)]])  # 方角yaw[deg]

    oRobot = Robot(x_base, period_ms / 1000)

    ani = animation.FuncAnimation(fig, animate, frames = frame_cnt, fargs = (period_ms, oRobot), blit = False,
                                  interval = period_ms, repeat = False)

    # ani.save('Localization_by_pf.mp4', bitrate=6000)

    plt.show()

