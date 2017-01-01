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

class Transform(object):
    '''座標変換'''

    def __init__(self):
        self.__BASE_ANG = np.pi / 2.0

    def world2local(self, origin, world):
        '''ワールド座標系→ローカル座標系変換
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
            local：ローカル座標（ベクトル）
             local[n, 0]：x座標
             local[n, 1]：y座標
             n:要素数

        '''
        yaw = self.__BASE_ANG - origin[2, 0]
        diff = world - origin.T[0, 0:2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw) ]])
        local = (rot @ diff.T).T
        return local

    def local2world(self, origin, local):
        '''ワールド座標系→ローカル座標系変換
        引数：
            origin：変換原点位置（ベクトル）
             origin[0, 0]：x座標
             origin[1, 0]：y座標
             origin[2, 0]：方角(rad)
            local：ローカル座標（ベクトル）
             local[n, 0]：x座標
             local[n, 1]：y座標
             n:要素数
        返り値：
            world：ワールド座標（ベクトル）
             world[n, 0]：x座標
             world[n, 1]：y座標
             n:要素数

        '''
        yaw = origin[2, 0] - self.__BASE_ANG
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw) ]])
        world = (rot @ local.T).T + origin.T[0, 0:2]
        return world



if __name__ == '__main__':
    origin = np.array([[1.0],
                       [3.0],
                       [np.deg2rad(-10.0)]])
    world = np.array([[ 4.0, 4.0],
                      [ 0.0, 0.0],
                      [-2.0, 5.0],
                      [ 3.0, -1.0],
                      [ 4.3, -5.7],
                      [-1.1, -4.5],
                      [ 6.0, 8.0],
                      [-1.0, -1.0]])

    tf = Transform()
    local = tf.world2local(origin, world)
    world2 = tf.local2world(origin, local)

    # 新規のウィンドウを描画
    fig = plt.figure()
    # サブプロットを追加
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    y = np.sin(origin[2])
    x = np.cos(origin[2])
    ax1.scatter(world.T[0], world.T[1], c='green', s=60, marker='o', alpha=0.5, label='World Observation')
    ax1.scatter(origin[0], origin[1], c='red', s=60, marker='o', alpha=0.5, label='World Position')
    ax1.quiver(origin[0], origin[1], x, y, angles='xy', scale_units='xy', scale=1)

    ax2.scatter(local.T[0], local.T[1], c='green', s=60, marker='o', alpha=0.5, label='Local Observation')
    ax2.scatter(0, 0, c='red', s=60, marker='o', alpha=0.5, label='Local Position')
    ax2.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1)

    ax3.scatter(world2.T[0], world2.T[1], c='green', s=60, marker='o', alpha=0.5, label='World2 Observation')
    ax3.scatter(origin[0], origin[1], c='red', s=60, marker='o', alpha=0.5, label='World2 Position')
    ax3.quiver(origin[0], origin[1], x, y, angles='xy', scale_units='xy', scale=1)

    ax1.axis('equal')
    ax1.grid()
    ax1.set_title('World', fontsize=12)
    ax1.legend()

    ax2.axis('equal')
    ax2.grid()
    ax2.set_title('Local', fontsize=12)
    ax2.legend()

    ax3.axis('equal')
    ax3.grid()
    ax3.set_title('World2', fontsize=12)
    ax3.legend()

    plt.show()