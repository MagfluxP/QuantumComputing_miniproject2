# Gao spin Hamiltonian used by the hybrid

The active magnetic input is Gao et al. (2024), Methods Eqs. 1-2 and
Supplementary Table S1. The optical model is supplied separately; see
[the current hybrid construction](hybrid_gao_optical.md).

For quantum S=1 operators on a chosen open seven-site triangular cluster:

H0 = sum_(i<j) Si.[Jn I + Jr_ij].Sj
   + B sum_NN (Si.Sj)^2 + (Azz/2) sum_i Sz_i^2.

There are 12 first-, 6 second- and 3 third-neighbor pairs. Primary values in meV
are J1=-5.03, J2=-0.32, J3=3.95, B=-0.89 and Azz=0.44. Nearest anisotropy
(xx,yy,zz,xy)=(-0.13,0.82,-0.69,-0.96); third-neighbor=(0.08,-0.02,-0.06,-0.045).
Other entries vanish in this transcription. C3 rotations generate bond tensors.

Source choices are unresolved, not fitted away:

- Main-text J2=+0.32 differs from the table's -0.32; both are reported.
- Reference axes follow the caption: y along a reference Ni-Ni bond, z out of
  plane. Surrounding supplementary text says cubic; clarification is needed.
- The literal single-ion factor 1/2 is retained. Pair terms occur once per
  unordered pair, equivalent to half a directed pair sum.

The source obtains magnons semiclassically in a larger periodic supercell.
Quantizing S=1 on this open cluster is our finite realization. It does not
establish an experimental single domain or reproduce bulk magnon energies.
The hybrid conditions a domain with an explicit weak chirality source and forms
thermal pair densities at 1.63 K. Spin diagonalization is classical.

`src/nii2_shg/gao_spin_model.py` defines H0; `hybrid_response.py` defines the
state and frame mapping. `outputs/gao_spin_h0.npz` is the 2187x2187 unconditioned
sparse matrix in eV. Base-three basis digits label +1,0,-1, with site 0 least
significant. `outputs/hybrid_shg_report.json` records the active state settings.

Source: [Gao et al.](https://doi.org/10.1038/s41586-024-07678-5).
The original supplement and source workbook are preserved without modification.
