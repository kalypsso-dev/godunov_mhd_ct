# Expansion shock tube

## Reference

A multi-state HLL approximate Riemann solver for ideal magnetohydrodynamics. Miyoshi and Kusano, Journal of Computational Physics, Volume 208, Issue 1, 1 September 2005, Pages 315-344.

https://doi.org/10.1016/j.jcp.2005.02.017

## Run

```shell
../solver_godunov_mhd_ct --ini ./test_mhd_expansion_I_2d.ini
# or
../solver_godunov_mhd_ct --ini ./test_mhd_expansion_II_2d.ini
```

## Plot

```shell
./plot_I.py
./plot_II.py
```
