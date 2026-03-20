# Godunov\_mhd\_ct

This repository is part of [kalypsso-dev](https://github.com/kalypsso-dev) project and is not intended to be standalone, but rather used as a submodule of [kalypsso-app-pub](https://github.com/kalypsso-dev/kalypsso-app-pub).

## What is it ?

This repository contains a MHD solver implementation using the constraint transport method for solving the magnetic induction equation, while preserving the divergence-free property of the mangetic field.

The implementation is a direct adaptation from godunov\_hydro, using the same type of finite volume discretization, built upon the HLLD Riemann solver.

## Scientific references

- [A high order Godunov scheme with constrained transport and adaptive mesh refinement for astrophysical magnetohydrodynamics](https://doi.org/10.1051/0004-6361:20065371), Fromang et al., Astrophysics and Astronomy, 
Volume 457, number 2, (2006), Pages 371-384.


# Licenses

This project is released under [Apache-2.0 WITH LLVM-exception](https://github.com/kalypsso-dev/godunov_mhd_ct/LICENSES/Apache-2.0.txt).
