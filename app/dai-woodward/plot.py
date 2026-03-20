#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys, getopt
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import configparser

def do_plot(ini_filename, shock_tube_name, add_reference_data):

    config = configparser.ConfigParser()
    config.read(ini_filename)

    tEnd = config.getfloat('run', 'tEnd', fallback=0.2)
    level_min = config.getint('amr', 'level_min', fallback=0)
    level_max = config.getint('amr', 'level_max', fallback=0)

    config = configparser.ConfigParser()
    config.read(ini_filename)

    tEnd = config.getfloat('run', 'tEnd', fallback=0.2)
    gamma = config.getfloat('hydro', 'gamma0', fallback=1.4)
    prefix = config.get('shock-tube', 'name', fallback='')

    # load numerical solution
    pos = np.load(prefix+'_positions.npy')
    level = np.load(prefix+'_level.npy')

    rho = np.load(prefix+'_rho.npy')
    By = np.load(prefix+'_By.npy')

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(11,8))
    ax1.plot(pos, rho, 'g-', label='rho')
    ax2.plot(pos, By, 'b-', label='By')
    ax3.plot(pos, level, 'k-', label='AMR levels')

    if add_reference_data:
        pos_ref = np.load(prefix+'_ref_positions.npy')
        level_ref = np.load(prefix+'_ref_level.npy')
        rho_ref = np.load(prefix+'_ref_rho.npy')
        By_ref = np.load(prefix+'_ref_By.npy')
        ax1.plot(pos_ref, rho_ref, 'g--', label='rho ref')
        ax2.plot(pos_ref, By_ref, 'b--', label='By ref')
        ax3.plot(pos_ref, level_ref, 'k--', label='AMR levels ref')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.suptitle('{} at tEnd={}\ngamma={}, AMR levels {} to {}'.format(shock_tube_name, tEnd, gamma, level_min, level_max), fontsize=20)
    plt.show()

###############################################################################
if __name__ == "__main__":

    shock_tube_name="Dai-Woodward"
    parser = argparse.ArgumentParser(description='Display '+shock_tube_name+' plots.')
    parser.add_argument('--ini', type=str, default='test_mhd_dai_woodward_2d.ini', help='ini parameter file')
    parser.add_argument('--ref', default=False, action='store_true')
    args = parser.parse_args()

    do_plot(args.ini, shock_tube_name, args.ref)
