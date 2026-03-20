# Brio-Wu shock tube 2d

## Reference

M. Brio and C.C. Wu. An upwind diﬀerencing scheme for the equations of ideal magnetohydrodynamics. J. Comput. Phys., 75(2):400–422, 1988.

https://doi.org/10.1016/0021-9991(88)90120-9

## Run

```shell
../solver_godunov_mhd_ct --ini ./test_mhd_brio_wu_I_2d.ini
../solver_godunov_mhd_ct --ini ./test_mhd_brio_wu_II_2d.ini
```

## Plot

```shell
./plot_brio_wu_I.py
./plot_brio_wu_II.py
```
