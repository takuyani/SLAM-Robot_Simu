#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================
# brief        リミット
#
# author       Takuya Niibori
# attention    none
#==============================================================================
import numpy as np

def limit_angle(angle_in):
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


if __name__ == '__main__':
    pass
