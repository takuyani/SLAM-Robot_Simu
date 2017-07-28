#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================
# brief        座標変換
#
# author       Takuya Niibori
# attention    none
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt

BASE_ANG = np.pi / 2.0

def world2robot(origin, world):
    """ワールド座標系→ロボット座標系変換
    引数：
        origin：変換原点位置（ベクトル）
            origin[0, 0]：x座標
            origin[1, 0]：y座標
            origin[2, 0]：方角(rad)
        world：ワールド座標（ベクトル）
            world[n, 0]：x座標
            world[n, 1]：y座標
            n:要素数
    返り値：
        robot：ロボット座標（ベクトル）
            robot[n, 0]：x座標
            robot[n, 1]：y座標
            n:要素数
    """
    yaw = BASE_ANG - origin[2, 0]
    diff = world - origin.T[0, 0:2]
    rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw), np.cos(yaw) ]])
    robot = (rot @ diff.T).T
    return robot

def robot2world(origin, robot):
    """ローカル座標系→ワールド座標系変換
    引数：
        origin：変換原点位置（ベクトル）
            origin[0, 0]：x座標
            origin[1, 0]：y座標
            origin[2, 0]：方角(rad)
        robot：ロボット座標（ベクトル）
            robot[n, 0]：x座標
            robot[n, 1]：y座標
            n:要素数
    返り値：
        world：ワールド座標（ベクトル）
            world[n, 0]：x座標
            world[n, 1]：y座標
            n:要素数
    """
    yaw = origin[2, 0] - BASE_ANG
    rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw), np.cos(yaw) ]])
    world = (rot @ robot.T).T + origin.T[0, 0:2]
    return world

if __name__ == "__main__":
    origin = np.array([[1.0],
                       [3.0],
                       [np.deg2rad(-10.0)]])
    world = np.array([[ 4.0, 4.0],
                      [ 0.0, 0.0],
                      [ 3.0, -1.0],
                      [-3.0, -5.0]])

    cx = origin[0, 0]
    cy = origin[1, 0]
    scl = 10

    local = world2robot(origin, world)
    world2 = robot2world(origin, local)


    # 新規のウィンドウを描画
    fig = plt.figure(figsize=(12, 6))
    # サブプロットを追加
    ax1 = plt.subplot2grid((1, 3), (0, 0), aspect="equal", adjustable="box-forced")
    ax2 = plt.subplot2grid((1, 3), (0, 1), aspect="equal", adjustable="box-forced")
    ax3 = plt.subplot2grid((1, 3), (0, 2), aspect="equal", adjustable="box-forced")

    y = np.sin(origin[2])
    x = np.cos(origin[2])
    ax1.scatter(world.T[0], world.T[1], c="green", s=60, marker="o", alpha=0.5)
    ax1.scatter(origin[0], origin[1], c="red", s=60, marker="o", alpha=0.5)
    ax1.quiver(origin[0], origin[1], x, y, angles="xy", scale_units="xy", scale=1)
    # 線分の描写
    p_org = origin[0:2, 0].T
    for i in range(world.shape[0]):
        linex = np.array([p_org[0], world[i][0]])
        liney = np.array([p_org[1], world[i][1]])
        ax1.plot(linex, liney, "--", c="green")

    ax2.scatter(local.T[0], local.T[1], c="green", s=60, marker="o", alpha=0.5)
    ax2.scatter(0, 0, c="red", s=60, marker="o", alpha=0.5)
    ax2.quiver(0, 0, 0, 1, angles="xy", scale_units="xy", scale=1)
    # 線分の描写
    for i in range(world.shape[0]):
        linex = np.array([0, local[i][0]])
        liney = np.array([0, local[i][1]])
        ax2.plot(linex, liney, "--", c="green")

    ax3.scatter(world2.T[0], world2.T[1], c="green", s=60, marker="o", alpha=0.5)
    ax3.scatter(origin[0], origin[1], c="red", s=60, marker="o", alpha=0.5)
    ax3.quiver(origin[0], origin[1], x, y, angles="xy", scale_units="xy", scale=1)
    # 線分の描写
    p_org = origin[0:2, 0].T
    for i in range(world.shape[0]):
        linex = np.array([p_org[0], world2[i][0]])
        liney = np.array([p_org[1], world2[i][1]])
        ax3.plot(linex, liney, "--", c="green")

    ax1.grid()
    ax1.set_title("World", fontsize=12)
    ax1.axis([cx - scl, cx + scl, cy - scl, cy + scl])
    ax1.legend()

    ax2.grid()
    ax2.set_title("Robot", fontsize=12)
    ax2.axis([-scl, scl, -scl, scl])
    ax2.legend()

    ax3.grid()
    ax3.set_title("World2", fontsize=12)
    ax3.axis([cx - scl, cx + scl, cy - scl, cy + scl])
    ax3.legend()

    plt.show()