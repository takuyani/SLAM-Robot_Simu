#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================
# brief        パーティクルフィルタを用いた自己位置推定
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
from statsmodels.regression.tests.test_quantile_regression import idx
from numba.typing.npydecl import NdArange
from cmath import sin

class ParticleFilter(object):
    '''パーティクルフィルタ'''

    def __init__(self, period_ms):
        '''コンストラクタ
        引数：
            period_ms：更新周期[msec]
        返り値：
            なし

        '''
        self.__tf = transform.Transform()

        #---------- 定数定義 ----------
        self.__DT_s = period_ms / 1000  # 更新周期[sec]
        self.__NP = 10  # パーティクル数
        self.__NP_RECIP = 1 / self.__NP # パーティクル数の逆数
#        self.__NTH = self.__NP / 32.0    # リサンプリングを実施する有効パーティクル数
        self.__NTH = self.__NP + 1    # リサンプリングを実施する有効パーティクル数

        #---------- ランドマーク ----------
        self.__LM = np.array([[  5.0,  5.0],
                              [  0.0, 15.0],
                              [ -3.0,  4.0],
                              [-15.0,  0.0],
                              [  0.0,  0.0]])

        #---------- 状態空間モデルパラメータ定義 ----------
        self.__RADIUS_m = 10.0  # 周回半径[m]
        self.__YAW_RATE_rps = np.deg2rad(10.0)  # 角速度[rad/s]
        self.__VEL_mps = self.__RADIUS_m * self.__YAW_RATE_rps  # 速度[m/s]

        # 状態遷移行列A
        self.__A = np.array([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])

        # 制御行列B
        self.__B = np.array([[self.__VEL_mps, 0.0           , 0.0                ],
                             [0.0           , self.__VEL_mps, 0.0                ],
                             [0.0           , 0.0           , self.__YAW_RATE_rps]])

        # 観測行列C
        self.__C = np.array([[1.0, 0.0, 0.0],  # 観測行列
                             [0.0, 1.0, 0.0]])

        #---------- 雑音ベクトルの共分散行列定義 ----------
        # システム雑音
        cov_sys_px = 0.1  # 位置x[m]の標準偏差
        cov_sys_py = 0.1  # 位置y[m]の標準偏差
        cov_sys_yaw = 0.0  # 角度yaw[deg]の標準偏差
        self.__Q = np.diag([cov_sys_px, cov_sys_py, np.deg2rad(cov_sys_yaw)]) ** 2

        # 観測雑音
        cov_obs_px = 1.1  # 位置x[m]の標準偏差
        cov_obs_py = 1.1  # 位置y[m]の標準偏差
        self.__R = np.diag([cov_obs_px, cov_obs_py]) ** 2

        #---------- シミュレーションパラメータ ----------
        # システム雑音
        cov_sys_pxa  = cov_sys_px + 0.0  # 位置x[m]の標準偏差
        cov_sys_pya  = cov_sys_py + 0.0  # 位置y[m]の標準偏差
        cov_sys_yawa = cov_sys_yaw + 0.0 # 角度yaw[deg]の標準偏差
        self.__Q_act = np.diag([cov_sys_pxa, cov_sys_pya, np.deg2rad(cov_sys_yawa)]) ** 2

        # 観測雑音
        cov_obs_pxa = cov_obs_px + 0.0  # 位置x[m]の標準偏差
        cov_obs_pya = cov_obs_py + 0.0  # 位置y[m]の標準偏差
        self.__R_act = np.diag([cov_obs_pxa, cov_obs_pya]) ** 2

        #---------- 初期状態 ----------
        # 位置
        px0 = self.__RADIUS_m  # 位置x[m]
        py0 = 0.0  # 位置y[m]
        yaw0 = 90.0  # 角度yaw[deg]
        self.__x_true = np.array([[px0],
                                  [py0],
                                  [np.deg2rad(yaw0)]])
        self.__x_dr = self.__x_true
        self.__P = np.diag([0.01, 0.01, np.deg2rad(90.0)]) ** 2  # 誤差共分散行列

        self.__px = matlib.repmat(self.__x_true, 1, self.__NP) # パーティクル格納変数
        # 重み変数初期化
        self.__pw_ini = matlib.repmat(self.__NP_RECIP, 1, self.__NP)
        self.__pw = np.copy(self.__pw_ini)

    def main_pf(self):
        '''パーティクルフィルタメイン処理
        引数：
            なし
        返り値：
            x_true：状態(k)[真値]
            x_dr：状態(k)[デットレコニング]
            z：観測値z(k)
            x_hat_m：事前状態推定値x^m(k)

        '''
        # ---------- Ground Truth ----------
        self.__x_true = self.__f(self.__x_true)

        # ---------- Observation ----------
        z_l = self.__tf.world2local(self.__x_true, self.__LM)
        w = np.random.multivariate_normal([0.0, 0.0], self.__R_act, z_l.shape[0])
#        z_l += w
#        z = self.__observation(self.__x_true, w)

        # ---------- Dead Reckoning ----------
        # 状態方程式：x(k+1) = A * x(k) + B * u(k) + v [v~N(0,Q)]
        v = np.random.multivariate_normal([0.0, 0.0, 0.0], self.__Q_act, self.__NP).T
        self.__x_dr = self.__f(self.__x_dr) + v

        # ========== Particle Filter(PF) ==========
        # 予測
        px_est = self.__predict(self.__px)

        # 尤度の計算
        pw_new = self.__likelihood(px_est, self.__pw, z_l)

        # リサンプリング
        self.__px, self.__pw = self.__resampling(px_est, pw_new)

        max_val = np.max(pw_new)    # 重み最大値
        max_idx = np.argmax(pw_new) # 重み最大値のインデックス
        x_est = np.array(px_est[:,max_idx], ndmin=2).T  # 推定値x

        print('WghtMaxVal = {0}, WghtMaxIdx = {1}'.format(max_val, max_idx))

        return self.__LM, self.__x_true, x_est, px_est, pw_new[0], self.__Q

    def __predict(self, px):
        '''パーティクルフィルタメイン処理
        引数：
            なし
        返り値：
            x_true：状態(k)[真値]
            x_dr：状態(k)[デットレコニング]
            z：観測値z(k)
            x_hat_m：事前状態推定値x^m(k)

        '''
        v = np.random.multivariate_normal([0.0, 0.0, 0.0], self.__Q, self.__NP).T
        px_est = self.__f(px) + v

        return px_est

    def __likelihood(self, px_est, pw, z_l):
        sigma_xx = np.sqrt(self.__R[0][0])
        sigma_yy = np.sqrt(self.__R[1][1])
        sigma_xy = np.sqrt(self.__R[0][1])

        # 尤度の計算
        bn = np.zeros(self.__NP)
        for i in range(self.__NP):
            px = np.array([px_est[:,i]]).T
            pz_l = self.__tf.world2local(px, self.__LM)
            diff_pz = pz_l - z_l
            dx = diff_pz[:,0]
            dy = diff_pz[:,1]
            bnlm = mlab.bivariate_normal( dx, dy, sigma_xx, sigma_yy, 0.0, 0.0, sigma_xy )
            bn[i] = bnlm.prod()

        pw_update = pw * bn
        # 尤度の正規化
        pw_update = self.__normalize(pw_update)

        return pw_update

    def __resampling(self, px, pw):
        n_eff = np.reciprocal(pw @ pw.T)
        if n_eff < self.__NTH:
            pw_cum = np.cumsum(pw)
            base_id = np.arange(0.0, 1.0, self.__NP_RECIP)
            ofs = np.random.rand() * self.__NP_RECIP # 初回位置オフセット
            resample_id = base_id + ofs
            px_temp = np.copy(px)
            idx = 0
            for i in range(self.__NP):
                while resample_id[i] > pw_cum[idx]:
                    idx += 1
                px[:,i] = px_temp[:,idx]
            pw = np.copy(self.__pw_ini)  # 尤度の初期化

        return px, pw

    def __normalize(self, pw):
        pw_sum = np.sum(pw)
        nor_pw = pw / pw_sum
        nor_pw[np.isnan(nor_pw)] = 1 / self.__NP
        return nor_pw



    def __observation(self, x, w):
        '''観測値y(k)算出
            観測方程式：y(k) = C * x(k) + w [w~N(R,Q)]
        引数：
            x：状態x(k)
            w：観測雑音ベクトル[w~N(R,Q)]
        返り値：
            y：観測値y(k)
        '''
        x_l = np.array([[0.0],
                        [0.0],
                        [np.deg2rad(90.0)]])
        y_l = (self.__C @ x_l) + w
        y_w = self.__tf.local2world(x, y_l.T)
        return y_w.T

    def __f(self, x):
        '''状態x(k+1)算出
            状態方程式：x(k+1) = A * x(k) + B * u(k)
        引数：
            x：状態x(k)
        返り値：
            x_next：状態x(k+1)
        '''
        yaw = x[2, :]
        a = self.__DT_s * np.cos(yaw)
        b = self.__DT_s * np.sin(yaw)
        c = np.full_like(a, self.__DT_s)
        u = np.array([a,
                      b,
                      c])

        x_next = (self.__A @ x) + (self.__B @ u)

        for i in range(x_next.shape[1]):
            x_next[2, i] = self.__limit_angle(x_next[2, i])

        return x_next

    def __limit_angle(self, angle_in):
        '''角度範囲補正処理
            角度を-π～πの範囲に補正する
        引数：
            angle_in：補正前角度[rad]
        返り値：
            angle_out：補正後角度[rad]
        '''
        angle_out = np.absolute(angle_in)
        while angle_out > np.pi:
            angle_out -= np.pi * 2

        if angle_in < 0:
            angle_out *= -1

        return angle_out



P1 = []
T1 = []
T2 = []
T3 = []
P2 = []
P3 = []
P4 = []
time_s = 0

# 誤差楕円の信頼区間[%]
confidence_interval = 99.0

ee = error_ellipse.ErrorEllipse(confidence_interval)
tf = transform.Transform()

def animate(i, pf, period_ms):
    global P1, P2, P3, P4
    global time_s
    col_x_true = 'red'
#    col_x_dr = 'yellow'
    col_z = 'green'
    col_x_hat = 'blue'

    time_s += period_ms / 1000

    lm, x_true, x_est, px, pw, Q = pf.main_pf()

#    a = [
#       ['Tim',     55, 46 ],
#       ['Jack',    55, 70 ],
#       ['Mathhew', 23, 80 ],
#    ]
#
#    xt_l = tf.world2local(x_true, lm)
#    for i in range(px.shape[1]):
#        xest_l = tf.world2local(px[i], lm)
#        dpx = np.array([xest_l[0] - xt_l[0],
#                        xest_l[1] - xt_l[1]])
#        norm = np.linalg.norm(dpx, axis=0)
#
#    b = []
#
#    dpx = np.array([xest_l[0] - xt_l[0],
#                    xest_l[1] - xt_l[1]])
#    norm = np.linalg.norm(dpx, axis=0)
#    for i in range(px.shape[1]):
#        aa = i, norm[i], pw[i]
#        b.append(aa)

#    b = .append(px[0, :], px[1, :]

#    c = [ px[0, :], px[1, :] ]

#    for i in range(px.shape[1]):
#        tbl = [
#               ]

    plt.cla()

    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))

    # ランドマークの描写
    ax1.scatter(lm[:,0], lm[:,1], s=600, c="yellow", marker="*", alpha=0.5, linewidths="2", edgecolors="orange")
    ax2.scatter(lm[:,0], lm[:,1], s=600, c="yellow", marker="*", alpha=0.5, linewidths="2", edgecolors="orange")

    # 状態x(真値)の描写
    P1.append(x_true)
    a, b, c = np.array(np.concatenate(P1, axis=1))
    ax1.plot(a, b, c=col_x_true, linewidth=1.0, linestyle='-', label='Est')
    ax1.scatter(x_true[0], x_true[1], c=col_x_true, marker='o', alpha=0.5)
    ax1.quiver(x_true[0], x_true[1], np.cos(x_true[2]), np.sin(x_true[2]), color = 'red', width  = 0.003)
    ax2.plot(a, b, c=col_x_true, linewidth=1.0, linestyle='-', label='Est')
    ax2.scatter(x_true[0], x_true[1], c=col_x_true, marker='o', alpha=0.5)
    ax2.quiver(x_true[0], x_true[1], np.cos(x_true[2]), np.sin(x_true[2]), color = 'red', width  = 0.003)

    # 状態x(デットレコニング)の描写
#    P2.append(x_dr)
#    a, b, c = np.array(np.concatenate(P2, axis=1))
#    ax1.plot(a, b, c=col_x_dr, linewidth=1.0, linestyle='-', label='Dead Reckoning')
#    ax1.scatter(x_dr[0], x_dr[1], c=col_x_dr, marker='o', alpha=0.5)

    # 観測zの描写
#    P3.append(obs)
#    a, b = np.array(np.concatenate(P3, axis=1))
#    ax1.scatter(a, b, c=col_z, marker='o', alpha=0.5, label='Observation')

    # 状態x(推定値)の描写
#    P4.append(x_pre)
#    a, b, c = np.array(np.concatenate(P4, axis=1))
#    ax1.plot(a, b, c=col_x_hat, linewidth=1.0, linestyle='-', label='Predicted')
#    ax1.scatter(x_pre[0], x_pre[1], c=col_x_hat, marker='o', alpha=0.5)
    P4.append(x_est)
    a, b, c = np.array(np.concatenate(P4, axis=1))
    ax1.plot(a, b, c=col_x_hat, linewidth=1.0, linestyle='-', label='Est')
    ax1.scatter(px[0], px[1], c=col_x_hat, marker='o', alpha=0.5)
    ax1.quiver(px[0], px[1], np.cos(px[2]), np.sin(px[2]), width  = 0.003)
    ax2.plot(a, b, c=col_x_hat, linewidth=1.0, linestyle='-', label='Est')
    ax2.scatter(px[0], px[1], c=col_x_hat, marker='o', alpha=0.5)
    ax2.quiver(px[0], px[1], np.cos(px[2]), np.sin(px[2]), width  = 0.003)
#    for i in range(px.shape[1]):
#        l = '[' + str(i+1) + ']' + "%.3f" % pw[i]
#        ax2.annotate(l, xy = (px[0][i], px[1][i]), size = 15)

    # 誤差楕円生成
    Qxy = Q[0:2, 0:2]
    ee_l,ee_y, ee_ang_rad = ee.calc_error_ellipse(Qxy)
#    e = patches.Ellipse((x_pre[0, 0], x_pre[1, 0]), x, y, angle=np.rad2deg(ang_rad), linewidth=2, alpha=0.2,
#                         facecolor='yellow', edgecolor='black', label='Error Ellipse: %.2f[%%]' %
#                         confidence_interval)
#    print('time:{0:.3f}[s], x-cov:{1:.3f}[m], y-cov:{2:.3f}[m], xy-cov:{3:.3f}[m]'
#          .format(time_s, P[0, 0], P[1, 1], P[1, 0]))

#    ax1.add_patch(e)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Localization by PF')
    ax1.axis('equal')
    ax1.grid()
    ax1.legend(fontsize=10)

    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('Localization by PF')
    ee_l *= 1.5
    ax2.set_xlim(x_est[0][0] - ee_l, x_est[0][0] + ee_l)
    ax2.set_ylim(x_est[1][0] - ee_l, x_est[1][0] + ee_l)
    ax2.grid()
    ax2.legend(fontsize=10)


if __name__ == '__main__':

    period_ms = 100  # 更新周期[msec]
    frame_cnt = int(36 * 1000 / period_ms)

    # 描画
    fig = plt.figure(figsize=(12, 9))

    pf = ParticleFilter(period_ms)

    ani = animation.FuncAnimation(fig, animate, frames=frame_cnt, fargs=(pf, period_ms), blit=False,
                                  interval=period_ms, repeat=False)

#    ani.save('Localization_by_pf.mp4', bitrate=5000)

    plt.show()

