#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================
# brief        パーティクルフィルタを用いた自己位置推定
#
# author       Takuya Niibori
# attention    none
#==============================================================================

import numpy as np
from numpy import matlib as matlib
from matplotlib import animation, mlab
import matplotlib.pyplot as plt
from mylib import transform as tf
from mylib import error_ellipse
from mylib import limit

class ParticleFilter(object):
    '''パーティクルフィルタ'''

    def __init__(self, period_ms):
        '''コンストラクタ
        引数：
            period_ms：更新周期[msec]
        返り値：
            なし
        '''

        #---------- 定数定義 ----------
        self.__DT_s = period_ms / 1000  # 更新周期[sec]
        self.__NP = 1000  # パーティクル数
        self.__NP_RECIP = 1 / self.__NP  # パーティクル数の逆数
        self.__ESS_TH = self.__NP / 100.0  # 有効サンプルサイズ(ESS)閾値

        #---------- ランドマーク ----------
        '''
        LM(j) = [j番目LMのX座標, j番目LMのY座標]
        '''
        self.__LM = np.array([[ 5.0, 5.0],
                              [ 2.0, -3.0],
                              [-3.0, 4.0],
                              [-5.0, -1.0],
                              [ 0.0, 0.0]])

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

        #---------- 雑音ベクトルの共分散行列定義 ----------
        # システム雑音
        cov_sys_px = 0.03  # 位置x[m]の標準偏差
        cov_sys_py = 0.03  # 位置y[m]の標準偏差
        cov_sys_yaw = 2.0  # 角度yaw[deg]の標準偏差
        self.__Q = np.diag([cov_sys_px, cov_sys_py, np.deg2rad(cov_sys_yaw)]) ** 2

        # 観測雑音
        cov_obs_px = 0.3  # 位置x[m]の標準偏差
        cov_obs_py = 0.3  # 位置y[m]の標準偏差
        self.__R = np.diag([cov_obs_px, cov_obs_py]) ** 2

        #---------- 初期状態 ----------
        # 位置
        px0 = self.__RADIUS_m  # 位置x[m]
        py0 = 0.0  # 位置y[m]
        yaw0 = 90.0  # 角度yaw[deg]
        self.__x_true = np.array([[px0],
                                  [py0],
                                  [np.deg2rad(yaw0)]])
        # パーティクル格納変数
        self.__px = matlib.repmat(self.__x_true, 1, self.__NP)
        # 重み変数初期化
        self.__pw_ini = matlib.repmat(self.__NP_RECIP, 1, self.__NP)
        self.__pw = np.copy(self.__pw_ini)

    def main_pf(self):
        '''パーティクルフィルタメイン処理
        引数：
            なし
        返り値：
            LM：ランドマーク
            x_true：状態(k)[真値]
            x_est：状態(k)[推定値]
            self.__px：状態パーティクル(k)
            self.__Q：システム雑音
            max_idx：尤度最大値のインデックス
            max_val：尤度最大値
        '''
        # ---------- Ground Truth ----------
        self.__x_true = self.__f(self.__x_true)

        # ========== Particle Filter(PF) ==========
        # リサンプリング
        self.__px, self.__pw = self.__resampling(self.__px, self.__pw)

        # 予測
        self.__px = self.__predict(self.__px)

        # 観測
        z_l = self.__observation(self.__x_true)

        # 尤度の計算
        self.__pw = self.__likelihood(self.__px, self.__pw, z_l)

        max_val = np.max(self.__pw)  # 重み最大値
        max_idx = np.argmax(self.__pw)  # 重み最大値のインデックス
        x_est = np.array(self.__px[:, max_idx], ndmin=2).T  # 推定値x

        return self.__LM, self.__x_true, x_est, self.__px, self.__Q, max_idx, max_val

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
            x_next[2, i] = limit.limit_angle(x_next[2, i])

        return x_next

    def __observation(self, x_true):
        '''LandMark観測値算出
        引数：
            x：状態x(真値)
        返り値：
            z_l：ローカル座標系で観測されたLandMark
        '''
        z_l = tf.world2local(x_true, self.__LM)
        w = np.random.multivariate_normal([0.0, 0.0], self.__R, z_l.shape[0])
        z_l += w
        return z_l

    def __predict(self, px):
        '''予測処理
            全パーティクルの次状態(k+1)を計算する。
            x(k+1) = A * x(k) + B * u(k) + v [v~N(0,Q)])
        引数：
            px：パーティクルの現状態x(k)
        返り値：
            px_est：パーティクルの推定次状態x(k+1)
        '''
        v = np.random.multivariate_normal([0.0, 0.0, 0.0], self.__Q, self.__NP).T
        px_est = self.__f(px) + v

        return px_est

    def __likelihood(self, px_est, pw, z_l):
        '''尤度の計算
        引数：
            px_est：パーティクルの推定次状態x(k+1)
            pw：パーティクルの尤度
            z_l：ローカル座標系で観測されたLandMark
        返り値：
            pw_update：更新後のパーティクルの尤度
        '''
        sigma_xx = np.sqrt(self.__R[0][0])
        sigma_yy = np.sqrt(self.__R[1][1])
        sigma_xy = np.sqrt(self.__R[0][1])

        # 尤度の計算
        bn = np.zeros(self.__NP)
        for i in range(self.__NP):
            px = np.array([px_est[:, i]]).T
            pz_l = tf.world2local(px, self.__LM)
            diff_pz = pz_l - z_l
            dx = diff_pz[:, 0]
            dy = diff_pz[:, 1]
            bnlm = mlab.bivariate_normal(dx, dy, sigma_xx, sigma_yy, 0.0, 0.0, sigma_xy)
            bn[i] = bnlm.prod()

        pw_update = pw * bn
        # 尤度の正規化
        pw_update = self.__normalize(pw_update)

        return pw_update

    def __resampling(self, px, pw):
        '''リサンプリング処理
            有効サンプルサイズ(ESS)が閾値を下回った場合、リサンプリングが実行される。
        引数：
            px：パーティクルの状態
            pw：パーティクルの尤度
        返り値：
            px：次回演算用パーティクルの状態
            pw：次回演算用パーティクルの尤度
        '''
        ess = float(np.reciprocal(pw @ pw.T))
        if ess < self.__ESS_TH:
            pw_cum = np.cumsum(pw)
            base_id = np.arange(0.0, 1.0, self.__NP_RECIP)
            ofs = np.random.rand() * self.__NP_RECIP  # 初回位置オフセット
            resample_id = base_id + ofs
            px_temp = np.copy(px)
            idx = 0
            for i in range(self.__NP):
                while resample_id[i] > pw_cum[idx]:
                    idx += 1
                px[:, i] = px_temp[:, idx]
            pw = np.copy(self.__pw_ini)  # 尤度の初期化

        return px, pw

    def __normalize(self, pw):
        '''正規化処理
            全てのパーティクルの和が"1"となるように尤度を再計算する。
        引数：
            pw：パーティクルの尤度
        返り値：
            nor_pw：正規化後パーティクルの尤度
        '''
        pw_sum = np.sum(pw)
        nor_pw = pw / pw_sum
        nor_pw[np.isnan(nor_pw)] = self.__NP_RECIP
        return nor_pw


P1 = []
P2 = []
time_s = 0

# 誤差楕円の信頼区間[%]
confidence_interval = 99.0
ee = error_ellipse.ErrorEllipse(confidence_interval)

def animate(i, pf, period_ms):
    global P1, P2
    global time_s
    col_x_true = 'red'
    col_x_est = 'blue'

    time_s += period_ms / 1000

    lm, x_true, x_est, px, Q, w_idx, w_val = pf.main_pf()

    plt.cla()

    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))

    # ランドマークの描写
    ax1.scatter(lm[:, 0], lm[:, 1], s=100, c="yellow", marker="*", alpha=0.5, linewidths="2",
                edgecolors="orange", label='Land Mark')
    ax2.scatter(lm[:, 0], lm[:, 1], s=100, c="yellow", marker="*", alpha=0.5, linewidths="2",
                edgecolors="orange")

    # 線分の描写
    p_est = x_est[0:2, 0].T
    for i in range(lm.shape[0]):
        x = np.array([p_est[0], lm[i][0]])
        y = np.array([p_est[1], lm[i][1]])
        ax1.plot(x, y, '--', c='green')
        ax2.plot(x, y, '--', c='green')

    # 状態pxの描写
    ax1.scatter(px[0], px[1], c='cyan', marker='o', alpha=0.5)
    ax2.scatter(px[0], px[1], c='cyan', marker='o', alpha=0.5)
    ax2.quiver(px[0], px[1], np.cos(px[2]), np.sin(px[2]), color='cyan', units='inches', scale=6.0, width=0.01,
               headwidth=0.0, headlength=0.0, headaxislength=0.0)

    # 状態x(真値)の描写
    P1.append(x_true[0:2, :])
    a, b = np.array(np.concatenate(P1, axis=1))
    ax1.plot(a, b, c=col_x_true, linewidth=1.0, linestyle='-', label='Ground Truth')
    ax1.scatter(x_true[0], x_true[1], c=col_x_true, marker='o', alpha=0.5)
    ax2.plot(a, b, c=col_x_true, linewidth=1.0, linestyle='-')
    ax2.scatter(x_true[0], x_true[1], c=col_x_true, marker='o', alpha=0.5)
    ax2.quiver(x_true[0], x_true[1], np.cos(x_true[2]), np.sin(x_true[2]), color='red', units='inches', scale=6.0,
               width=0.01, headwidth=0.0, headlength=0.0, headaxislength=0.0)

    # 状態x(推定値)の描写
    P2.append(x_est[0:2, :])
    a, b = np.array(np.concatenate(P2, axis=1))
    ax1.plot(a, b, c=col_x_est, linewidth=1.0, linestyle='-', label='Estimation')
    ax1.scatter(x_est[0], x_est[1], c=col_x_est, marker='o', alpha=0.5)
    ax2.plot(a, b, c=col_x_est, linewidth=1.0, linestyle='-')
    ax2.scatter(x_est[0], x_est[1], c=col_x_est, marker='o', alpha=0.5)
    ax2.quiver(x_est[0], x_est[1], np.cos(x_est[2]), np.sin(x_est[2]), color=col_x_est, units='inches', scale=6.0,
               width=0.01, headwidth=0.0, headlength=0.0, headaxislength=0.0)

    # ラベル描写
    txt = 'Maximuim Likelihood Estimate:\n[Index]:{0}\n[Weight]:{1:.3f}'.format(w_idx, w_val)
    ax2.annotate(txt, xy=(x_est[0, 0], x_est[1, 0]), xycoords='data',
                xytext=(0.55, 0.9), textcoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc=(1.0, 0.7, 0.7)),
                arrowprops=dict(arrowstyle="->", color='black',
                                connectionstyle='arc3,rad=0'),
                )

    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Localization by PF')
    ax1.axis('equal', adjustable='box')
    ax1.grid()
    ax1.legend(fontsize=10)

    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('Zoom')
    ee_l = ee.calc_chi(confidence_interval, Q[0:2, 0:2]) * 3
    ax2.set_xlim(x_true[0][0] - ee_l, x_true[0][0] + ee_l)
    ax2.set_ylim(x_true[1][0] - ee_l, x_true[1][0] + ee_l)
    ax2.grid()
    ax2.legend(fontsize=10)

    print('time:{0:.3f}[s]'.format(time_s))


if __name__ == '__main__':

    period_ms = 100  # 更新周期[msec]
    frame_cnt = int(36 * 1000 / period_ms)

    # 描画
    fig = plt.figure(figsize=(18, 9))

    pf = ParticleFilter(period_ms)

    ani = animation.FuncAnimation(fig, animate, frames=frame_cnt, fargs=(pf, period_ms), blit=False,
                                  interval=period_ms, repeat=False)

    # ani.save('Localization_by_pf.mp4', bitrate=6000)

    plt.show()

