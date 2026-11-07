# Preserved primary-source evidence

These files retain their original authorship and publisher provenance. They are
retained locally for scholarly verification, not presented as project-created
research data. Reading copies (PDFs, archives and extracted manuscripts) are
excluded from Git; published workbooks and provenance remain included. The
calculation needs `raw/gao2024_fig2a_static.xlsx`, not the reading copies. See `download_provenance.json` for public URLs and SHA256 hashes.

- `raw/fan2026.pdf`: Fan et al., doi:10.1038/s41524-026-02184-w.
  Table 2 and Methods Eqs. 9–14, PDF pages 6–7, were visually checked.
- `raw/gao2024_supplement.pdf`: Gao et al., doi:10.1038/s41586-024-07678-5.
  Table S1 (PDF p.9), Table S3 and polarization fits (PDF p.20) were inspected.
- `raw/gao2024_fig2.zip`: publisher's Figure 2 source bundle.
- `raw/gao2024_fig2a_static.xlsx`: unchanged extraction of
  `2023-12-23031B-f2/Figure2a_T-dependent_SHG.xlsx` from that bundle.
  Analysis uses Sheet1 A3:R76 (parallel/crossed channels, eight temperatures).
  The duplicate 55-degree row is retained; there is no 360-degree row.
  Temporary Excel lockfiles in the bundle are ignored. Other time-dependent
  workbooks in the ZIP were not analyzed in this revision.
- `raw/gao2024_fig4.xlsx`: publisher source download, preserved but not analyzed.

The calibrated chi tensor is an experimental fit. Reconstructing the source
angular pattern from it is not an independent material prediction. The notebook
and `docs/sources_and_conventions.md` distinguish these claims explicitly.

Additional H0 acquisition evidence:

- `raw/mou2026.pdf` and `raw/mou2026_source.tar`: public arXiv v1 downloads.
  Text/bibliography members are retained for checking exact printed conventions.
  The archive member list is recorded in the provenance JSON; no numerical matrix
  files are present. Coulomb parameter arithmetic was visually checked on PDF p.8.
- `raw/fan2026_supplement.pdf`: four-page publisher supplement; contains analytic
  reduced spin/cavity calculations and extended figures, not a numerical optical
  Hamiltonian bundle. Its simplified model is not substituted for the full model.
