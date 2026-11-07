# Sources, conventions and provenance

The current notebook has one material calculation: the Gao-state hybrid.
Primary reading copies/data remain locally in `data/references/raw/`, with URLs
and SHA256 hashes in `data/references/download_provenance.json`. Git excludes
downloaded PDFs/archives and manuscript copies; the experimental workbook and
provenance remain included. None is presented as project-created data.

- **Gao et al. (2024):** spin Hamiltonian, Table S1 and static Figure 2a source
  data. `gao2024_parameters.json` records values and source locations. The model
  uses Table S1 J2=-0.32 meV, with main-text +0.32 as sensitivity. Caption axes
  versus cubic wording and the single-ion prefactor remain explicit conventions.
  The dataset has 73 rows at the first temperature 1.63 K, including a repeated
  55-degree row. Sheet1 B/K channels retain their parallel/crossed assignments.
- **Fan et al. (2026):** tabulated interactions/hopping and the interpreted local
  electronic model retained for optical response. `fan2026_parameters.json`
  records the negative Delta convention and printed SOC/prose discrepancy.
  Printed static polarization kernels are not calibrated optical dipoles here.
- **Mou and Winter (2026):** effective optical spin-operator strategy. The full
  numerical electronic/optical/NTO matrix bundle was not located in the inspected
  public archive. Scalar Coulomb inputs cannot reconstruct it. Figure 4 fits
  Song's dataset rather than the Gao trace used here. The original acquisition
  manifest remains a missing-input audit for direct Mou reproduction, not a gate
  that claims the current hybrid is absent.
- **Kharazi et al.:** fault-tolerant nonlinear-response algorithm. Part 1 implements
  small-instance QSP windows and coherent products. Part 2's material resolvents
  are dense exact block encodings, not full-scale execution of that algorithm.

The fitted Gao Table S3 optical tensor is experimental calibration, not an
independent quantum prediction. It is preserved as reference metadata but is not
used to manufacture a material-model curve. No external runtime logs or quantum
hardware measurements are supplied, and none is inferred.

Sources:
[Gao](https://doi.org/10.1038/s41586-024-07678-5),
[Fan](https://doi.org/10.1038/s41524-026-02184-w),
[Mou and Winter](https://arxiv.org/html/2602.02872v1),
[Kharazi et al.](https://arxiv.org/html/2404.01454v1).
