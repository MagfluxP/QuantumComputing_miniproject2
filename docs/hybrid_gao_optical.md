# Current model: Gao magnetic density and retained optical operator

The notebook's Part 2 contains the single material calculation. Gao's spin
Hamiltonian determines correlated pair densities. Retained electronic excitation
energies and point-center dipoles determine an optical operator at 1030 nm:

beta_ijk(omega) = Tr[rho_Gao O_optical_ijk(omega)].

The magnetic and optical Hamiltonians have different roles. This is a declared
hybrid approximation, not an author-supplied optical Gao model or validated bulk
susceptibility. No Gao meV spin energy replaces an electronic optical denominator.

## Fixed magnetic state

Use the open seven-site S=1 Gao cluster, all first/second/third-neighbor terms,
biquadratic coupling and anisotropy. The primary bundle is Supplementary Table
S1, including J2=-0.32 meV. The main-text +0.32 meV is a sensitivity, not selected
for its fit. The table-caption reference axes and literal single-ion factor 1/2
remain explicitly stated source interpretations.

Condition a single domain with

H_selected = H_Gao - h_chi sum_NN sin(q.delta) n.(Si cross Sj).

T=1.63 K, h_chi=0.05 meV, q=0.138, normal tilt +55 degrees and one handedness
are fixed before comparison. The chirality source is time-reversal even and
inversion odd; it is an assumption, not a measured Gao coupling. Finite-cluster
source conditioning does not reproduce spontaneous bulk helical order.

The lowest 24 eigenstates supply a classical thermal trace. Its omitted-weight
bound is about 6.1e-31 at the primary settings. Partial traces give all 12 NN pair
densities. Four positions per orientation are averaged, not different domains.
Zero temperature equally mixes a degenerate ground manifold.

## Common frame and optical operation

The full spin basis is little-endian base three; optical pairs are ordered
|mi,mj>. Reverse sites when a bond opposes the directed canonical optical bond.
Rotate densities using the S=1 representation of R_b.T L.T, with lab x along q,
y the chosen C2 axis and z out of plane. The Gao +y reference bond aligns with
optical X. Sum the three complex orientation responses before squaring.

The retained 2Ni/2I electronic model has 20 spin-orbitals, 16 electrons and 4845
Fock states. Ni t2g shells are frozen. Fan's tabulated interactions/hoppings and
printed SOC matrix are retained. mu=-e sum_p r_p n_p uses ideal centers with
assumed Ni-I length 2.8 Angstrom. Internal Peierls/gauge consistency does not
establish optical matrix completeness.

Classical frequency-targeted reduction constructs common optical spaces through
512 states. A polar-overlap mapping embeds Gao pair densities in the electronic
nine-state spin manifold, flattened for frozen-spin optics. Gamma1=0.5 eV and
Gamma2=1 eV are assumed optical broadenings. Full matrix/basis conventions and
missing Wannier/t2g pathways remain model limitations.

The 16-state quantum demonstration loads normalized vec(rho), applies C,R1,C,R2
with fresh work ancillas, reads the dipole overlap and restores all norms.
Eighteen circuits cover six tensor components for three orientations; one real/
imaginary Hadamard witness is rerun in the notebook. This uses dense exact
resolvent blocks, not a full-space QSP implementation. The illustrative Part 1
contains the QSP phase-filter construction.

## Comparison and limitations

| Current hybrid | All-data angular residual | Blocked held-out error |
|---|---:|---:|
| Quantum algorithm, 16 optical states | 77.45% | 88.51% |
| Classical reference, 512 optical states | 64.69% | 77.33% |

The circuit agrees with the identical 16-state classical tensor to about 4.7e-11.
The 16-state tensor differs from the 512-state reference by about 100.2%, and
384-to-512 changes by 2.78%; neither establishes full-space convergence.
The train-only constant-per-channel baseline has 67.68% held-out error, better
than the current physical model.

All 73 source rows at 1.63 K are preserved. One shared nonnegative detector gain
and azimuth are fitted; no physical parameters or domain mixtures are fitted.
Six 30-degree blocks modulo 180 group channels, duplicates and opposite angles.
These reused data/folds provide diagnostics, not fresh independent validation.
Absolute amplitude and bulk propagation are not predicted.

The zero-selector signal vanishes; weak-source amplitude is nearly linear while
normalized shape changes little. Handedness, zero temperature, opposite tilt and
both J2 signs are uncertainty controls of this one model. They are not a search
for the most favorable primary result. No diagnostic establishes a unique cause
of disagreement. Missing optical pathways, source/boundary dependence, reduction,
source conventions and bulk optics remain separate uncertainties.

## Quantum advantage

No speedup is demonstrated. The matched benchmark evaluates the same 18
preloaded responses and separates preprocessing. Dense quantum statevector
simulation is slower than the direct classical recurrence on this CPU; these
are not hardware runtime measurements. The current normalized Hadamard overlap
is small, making naive shot readout expensive. The saved variance estimate is
not an optimized or fundamental hardware bound.

Potential benefit of fault-tolerant spectral filtering requires efficient
many-body oracles, affordable correlated-state preparation and feasible precision.
Those requirements are not met by classically constructing dense small-instance
gates and a Gao thermal density. Hilbert-register size alone does not prove an
end-to-end exponential advantage. See notebook Section 3 for measured costs and
explicit scope.
