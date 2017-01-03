#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================
# brief        拡張カルマンフィルタを用いた自己位置推定
#
# author       Takuya Niibori
# attention    none
#==============================================================================

import numpy as np
import scipy as sp
from matplotlib import animation, patches
import matplotlib.pyplot as plt
from mylib import transform
from mylib import error_ellipse

class ExtendedKalmanFilter(object):
    '''拡張カルマンフィルタ'''

    def __init__(self, period_ms):
        '''コンストラクタ
        引数：
            period_ms：更新周期[msec]
        返り値：
            なし

        '''
        self.__tf = transform.Transform()

        #---------- 制御定数定義 ----------
        self.__DT_s = period_ms / 1000  # 更新周期[sec]
        self.__RADIUS_m = 10.0  # 周回半径[m]
        self.__YAW_RATE_rps = np.deg2rad(10.0)  # 角速度[rad/s]
        self.__VEL_mps = self.__RADIUS_m * self.__YAW_RATE_rps  # 速度[m/s]

        #---------- 観測定数定義 ----------
        self.__C = np.array([[1.0, 0.0, 0.0],  # 観測行列
                             [0.0, 1.0, 0.0]])

        #---------- 制御パラメータ初期化 ----------
        # 制御入力u(k)
        self.__u = np.array([[self.__VEL_mps],
                             [self.__YAW_RATE_rps]])

        #---------- ノイズ定義 ----------
        # システムノイズ
        cov_vel = 3.0  # 速度v[m/s]の標準偏差
        cov_yaw = 900.0  # 角速度ω[deg/s]の標準偏差
        self.__Q = np.diag([cov_vel, np.deg2rad(cov_yaw)]) ** 2

        # 観測ノイズ
        cov_px = 1.0  # 位置x[m]の標準偏差
        cov_py = 1.0  # 位置y[m]の標準偏差
        self.__R = np.diag([cov_px, cov_py]) ** 2

        #---------- シミュレーションパラメータ ----------
        cov_pxa = cov_px + 1.0  # 位置x[m]の標準偏差
        cov_pya = cov_py + 0.0  # 位置y[m]の標準偏差
        self.__R_act = np.diag([cov_pxa, cov_pya]) ** 2

        #---------- 初期状態 ----------
        # 位置
        px0 = self.__RADIUS_m  # 位置x[m]
        py0 = 0.0  # 位置y[m]
        yaw0 = 90.0  # 角度yaw[deg]
        self.__x_true = np.array([[px0],
                                  [py0],
                                  [np.deg2rad(yaw0)]])
        self.__x_dr = self.__x_true
        self.__x_hat = self.__x_true  # 状態推定値
        self.__P = np.diag([0.01, 0.01, np.deg2rad(90.0)]) ** 2  # 誤差共分散行列

    def main_ekf(self):
        '''拡張カルマンフィルタメイン処理
        引数：
            なし
        返り値：
            x_true：状態(k)[真値]
            x_dr：状態(k)[デットレコニング]
            z：観測値z(k)
            x_hat_m：事前状態推定値x^m(k)

        '''
        # ---------- Ground Truth ----------
        self.__x_true = self.__f(self.__x_true, self.__u)

        # ---------- Dead Reckoning ----------
        # 状態方程式：x(k+1) = A * x(k) + B * u(k) + v [v~N(0,Q)]
        v = np.random.multivariate_normal([0.0, 0.0], self.__Q, 1).T
        u_act = self.__u + v
        self.__x_dr = self.__f(self.__x_dr, u_act)

        # ========== Extended Kalman Filter(EKF) ==========
        # ---------- [Step1]Prediction ----------
        # 事前状態推定値
        x_hat_m = self.__f(self.__x_hat, self.__u)

        # 事前誤差共分散行列
        jF = self.__jacobF(self.__x_hat, self.__u)
        B = self.__calc_B(self.__x_hat)
        NoiseCov = B @ self.__Q @ B.T
        P_m = (jF @ self.__P @ jF.T) + NoiseCov

        # ---------- [Step2]Update/Filtering ----------
        z = self.__observation(self.__x_true, self.__C, self.__R_act)
        jH = self.__jacobH()
        e = z - (jH @ x_hat_m)
        G = self.__calc_kalman_gain(P_m, self.__C, self.__R)

        # 状態推定値
        self.__x_hat = x_hat_m + (G @ e)
        self.__x_hat[2, 0] = self.__limit_angle(self.__x_hat[2, 0])

        # 事後誤差共分散行列
        I = np.identity(self.__x_hat.shape[0])
        self.__P = (I - G @ self.__C) @ P_m

        return self.__x_true, self.__x_dr, z, x_hat_m, self.__P

    def __observation(self, x, C, R):
        '''観測値y(k)算出
            観測方程式：y(k) = C * x(k) + w [w~N(R,Q)]
        引数：
            x：状態x(k)
            C：観測行列
            R：観測雑音の共分散行列
        返り値：
            y：観測値y(k)
        '''
        w = np.random.multivariate_normal([0.0, 0.0], R, 1).T
        x_l = np.array([[0.0],
                        [0.0],
                        [np.deg2rad(90.0)]])
        y_l = (C @ x_l) + w
        y_w = self.__tf.local2world(x, y_l.T)
        return y_w.T

    def __calc_kalman_gain(self, P_m, C, R):
        '''カルマンゲイン算出
        引数：
            P_m: 事前誤差共分散行列Pm(k)
            C：観測行列
            R：観測雑音の共分散行列
        返り値：
            G：カルマンゲインG(k)
        '''
        S = (C @ P_m @ C.T) + R
        G = (P_m @ C.T) @ np.linalg.inv(S)
        return G

    def __f(self, x, u):
        '''状態x(k+1)算出
            状態方程式：x(k+1) = A * x(k) + B * u(k)
        引数：
            x：状態x(k)
            u：制御入力u(k)
        返り値：
            x_next：状態x(k+1)
        '''
        A = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        B = self.__calc_B(x)

        x_next = (A @ x) + (B @ u)
        x_next[2, 0] = self.__limit_angle(x_next[2, 0])

        return x_next

    def __calc_B(self, x):
        '''システムパラメータB(k)算出
        引数：
            x：状態x(k)
        返り値：
            B：システムパラメータB(k)
        '''
        yaw = x[2, 0]
        a = self.__DT_s * np.cos(yaw)
        b = self.__DT_s * np.sin(yaw)
        B = np.array([[a  , 0.0        ],
                      [b  , 0.0        ],
                      [0.0, self.__DT_s]])
        return B

    def __jacobF(self, xHat, u):
        '''動作モデルのヤコビアン
        引数：
            xHat：状態推定値x^(k)
            u：制御入力u(k)
            P：誤差共分散行列P(k)
        返り値：
            jF：動作モデルのヤコビアンjF(k)
        '''
        v = u[0, 0]
        yaw = xHat[2, 0]
        a = -self.__DT_s * v * np.sin(yaw)
        b = self.__DT_s * v * np.cos(yaw)
        jF = np.array([[0.0, 0.0, a  ],
                       [0.0, 0.0, b  ],
                       [0.0, 0.0, 0.0]])
        return jF

    def __jacobH(self):
        '''観測モデルのヤコビアン
        引数：
            なし
        返り値：
            jH：観測モデルのヤコビアンjH(k)
        '''
        jH = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])
        return jH

    def __limit_angle(self, angle_in):
        '''角度範囲補正処理
            角度を-π～πの範囲に補正する
        引数：
            angle_in：補正前角度[rad]
        返り値：
            angle_out：補正後角度[rad]
        '''
        angle_out = np.absolute(angle_in)
        if angle_out > np.pi:
            angle_out -= np.pi * 2

        if angle_in < 0:
            angle_out *= -1

        return angle_out



P1 = []
P2 = []
P3 = []
P4 = []
time_s = 0

# 誤差楕円の信頼区間[%]
confidence_interval = 99.0

ee = error_ellipse.ErrorEllipse(confidence_interval)

def animate(i, ekf, period_ms):
    global P1, P2, P3, P4
    global time_s
    col_x_true = 'red'
#    col_x_dr = 'yellow'
    col_z = 'green'
    col_x_hat = 'blue'

    time_s += period_ms / 1000

    x_true, x_dr, obs, x_pre, P = ekf.main_ekf()

    plt.cla()

    ax1 = plt.subplot2grid((1, 1), (0, 0))

    # 状態x(真値)の描写
    P1.append(x_true)
    a, b, c = np.array(np.concatenate(P1, axis=1))
    ax1.plot(a, b, c=col_x_true, linewidth=1.0, linestyle='-', label='Ground Truth')
    ax1.scatter(x_true[0], x_true[1], c=col_x_true, marker='o', alpha=0.5)

    # 状態x(デットレコニング)の描写
#    P2.append(x_dr)
#    a, b, c = np.array(np.concatenate(P2, axis=1))
#    ax1.plot(a, b, c=col_x_dr, linewidth=1.0, linestyle='-', label='Dead Reckoning')
#    ax1.scatter(x_dr[0], x_dr[1], c=col_x_dr, marker='o', alpha=0.5)

    # 観測zの描写
    P3.append(obs)
    a, b = np.array(np.concatenate(P3, axis=1))
    ax1.scatter(a, b, c=col_z, marker='o', alpha=0.5, label='Observation')

    # 状態x(推定値)の描写
    P4.append(x_pre)
    a, b, c = np.array(np.concatenate(P4, axis=1))
    ax1.plot(a, b, c=col_x_hat, linewidth=1.0, linestyle='-', label='Predicted')
    ax1.scatter(x_pre[0], x_pre[1], c=col_x_hat, marker='o', alpha=0.5)

    # 誤差楕円生成
    Pxy = P[0:2, 0:2]
    x, y, ang_rad = ee.calc_error_ellipse(Pxy)
    e = patches.Ellipse((x_pre[0, 0], x_pre[1, 0]), x, y, angle=np.rad2deg(ang_rad), linewidth=2, alpha=0.2,
                         facecolor='yellow', edgecolor='black', label='Error Ellipse: %.2f[%%]' %
                         confidence_interval)
    print('time:{0:.3f}[s], x-cov:{1:.3f}[m], y-cov:{2:.3f}[m], xy-cov:{3:.3f}[m]'
          .format(time_s, P[0, 0], P[1, 1], P[1, 0]))

    ax1.add_patch(e)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Localization by EKF')
    ax1.axis('equal')
    ax1.grid()
    ax1.legend(fontsize=10)


if __name__ == '__main__':

    period_ms = 100  # 更新周期[msec]
    frame_cnt = int(36 * 1000 / period_ms)

    # 描画
    fig = plt.figure(figsize=(12, 9))

    ekf = ExtendedKalmanFilter(period_ms)

    ani = animation.FuncAnimation(fig, animate, frames=frame_cnt, fargs=(ekf, period_ms), blit=False,
                                  interval=period_ms, repeat=False)

#    ani.save('Localization_by_ekf.mp4', bitrate=5000)

    plt.show()

