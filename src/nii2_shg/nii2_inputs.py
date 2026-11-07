"""Audit material-input completeness without inventing a NiI2 Hamiltonian."""

import json
from pathlib import Path


class MissingNiI2Inputs(RuntimeError):
    """Required material-specific data were not supplied."""


def input_status(root: Path) -> dict:
    """Return missing paths and extracted scalar data from the project root.

    Files passing this audit still require basis, units, gauge, and physical
    validation. Presence of a file is not proof of reproducibility or accuracy.
    """
    folder = root / "data" / "nii2"
    manifest = json.loads((folder / "input_manifest.json").read_text())
    parameters = json.loads((folder / "literature_parameters.json").read_text())
    missing = [
        key
        for key, path in manifest["files"].items()
        if path is None or not (folder / path).is_file()
    ]
    return {
        "ready_for_validation": not missing,
        "missing": missing,
        "parameters": parameters,
        "manifest": manifest,
    }


def require_nii2_inputs(root: Path) -> dict:
    """Raise a descriptive error until all material/reference files are present."""
    status = input_status(root)
    if status["missing"]:
        raise MissingNiI2Inputs(
            "NiI2 response unavailable: missing " + ", ".join(status["missing"])
        )
    return status
