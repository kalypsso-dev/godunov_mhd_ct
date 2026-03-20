#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys, getopt, os
import argparse
import configparser
import subprocess

def run(ini_filename, level_min, level_max, block_size, is_3d, dry_run):
    print("Run test with\n  template={}\n  level_min={}\n  level_max={}\n  block_size={} 3d={}".format(ini_filename, level_min, level_max, block_size, is_3d))
    ini = configparser.RawConfigParser()
    ini.optionxform = lambda option : option
    ini.read(ini_filename)
    ini['amr']['level_min'] = "{}".format(level_min)
    ini['amr']['level_max'] = "{}".format(level_max)
    ini['amr']['bx'] = "{}".format(block_size)
    ini['amr']['by'] = "{}".format(block_size)
    if is_3d:
        ini['amr']['bz'] = "{}".format(block_size)


    prefix = os.path.splitext(ini_filename)[0]+'_{}_{}_{}'.format(level_min, level_max, block_size)

    ini['output']['outputPrefix'] = prefix

    ini_filename_out = prefix+'.ini'

    with open(ini_filename_out, 'w') as ini_out:
        ini.write(ini_out, space_around_delimiters=False)

    if not dry_run:
        with open(prefix+'.log', 'w') as log_out:
            print("Actually running test...")
            res = subprocess.run(["../solver_godunov_mhd_ct", "--ini", ini_filename_out], stdout=log_out, stderr=subprocess.STDOUT)
            if res.returncode != 0:
                print("Error running kalypsso, returned {}".format(res.returncode))

def run2d(ini_filename, large_run, dry_run):

    if not large_run:
        # fine resolution is 1024^2
        run(ini_filename, 3, 8,  4, False, dry_run)
        run(ini_filename, 3, 7,  8, False, dry_run)
        run(ini_filename, 3, 6, 16, False, dry_run)
        run(ini_filename, 3, 5, 32, False, dry_run)
        run(ini_filename, 3, 4, 64, False, dry_run)

    if large_run:
        # fine resolution is 4096^2
        run(ini_filename, 3, 10,   4, False, dry_run)
        run(ini_filename, 3,  9,   8, False, dry_run)
        run(ini_filename, 3,  8,  16, False, dry_run)
        run(ini_filename, 3,  7,  32, False, dry_run)
        run(ini_filename, 3,  6,  64, False, dry_run)
        run(ini_filename, 3,  5, 128, False, dry_run)
        run(ini_filename, 3,  4, 256, False, dry_run)
        run(ini_filename, 3,  3, 512, False, dry_run)

def run3d(ini_filename, large_run, dry_run):

    if not large_run:
        # fine resolution is 128^3
        run(ini_filename, 2, 5,  4, True, dry_run)
        run(ini_filename, 2, 4,  8, True, dry_run)
        run(ini_filename, 2, 3, 16, True, dry_run)

    if large_run:
        # fine resolution is 512^3
        run(ini_filename, 2, 7,  4, True, dry_run)
        run(ini_filename, 2, 6,  8, True, dry_run)
        run(ini_filename, 2, 5, 16, True, dry_run)
        #run(ini_filename, 2, 4, 32, True, dry_run)
        #run(ini_filename, 2, 3, 64, True, dry_run)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Block size study.')
    parser.add_argument('--ini', type=str, default='test_mhd_blast_2d.ini', help='kalypsso template ini parameter file')
    parser.add_argument('--dim', type=int, default=2, help='dimension')
    parser.add_argument('--large', action=argparse.BooleanOptionalAction, default=False, help='run larger test')
    parser.add_argument('--dry-run', action=argparse.BooleanOptionalAction, default=False, help='Generate ini file but not actually run simulations')
    args = parser.parse_args()

    if args.dim==3:
        run3d(args.ini, args.large, args.dry_run)
    else:
        run2d(args.ini, args.large, args.dry_run)
