"""Read preserved Gao Figure 2a source data and reproduce a calibrated angular fit.

This is experimental-reference analysis, NOT a quantum or ab-initio prediction.
XLSX is read as XML without editing the source workbook or executing formulas.
"""

import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_static_source(root: Path) -> dict:
    """Return 8 temperature-dependent parallel/crossed polar traces from Sheet1.

    Source cells A4:R76 contain angle and detector signals in microvolts. Preserve
    the raw 0..355-degree samples, including duplicate 55 degrees; no smoothing, baseline subtraction or rescaling.
    """
    path = root / "data/references/raw/gao2024_fig2a_static.xlsx"
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            tree = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(node.itertext()) for node in tree.findall("s:si", NS)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        cells = {}
        for cell in sheet.findall(".//s:sheetData/s:row/s:c", NS):
            value = cell.find("s:v", NS)
            if value is None:
                continue
            text = value.text
            if cell.get("t") == "s":
                value = strings[int(text)]
            else:
                try:
                    value = float(text)
                except (ValueError, TypeError):
                    value = text
            cells[cell.get("r")] = value
    if (
        cells.get("A3") != "Angle(degree)"
        or "Parallel" not in cells.get("B2", "")
        or "Crossed" not in cells.get("K2", "")
    ):
        raise ValueError("Unexpected Figure 2a workbook layout")
    rows = sorted(
        int(re.search(r"\d+", key).group())
        for key in cells
        if re.fullmatch(r"A\d+", key) and isinstance(cells[key], float)
    )
    temperatures = [
        float(str(cells[f"{col}3"]).removesuffix("K")) for col in "BCDEFGHI"
    ]
    traces = []
    for temperature, par, cross in zip(
        temperatures, "BCDEFGHI", "KLMNOPQR", strict=True
    ):
        traces.append(
            {
                "temperature_k": temperature,
                "parallel_uv": [cells[f"{par}{row}"] for row in rows],
                "crossed_uv": [cells[f"{cross}{row}"] for row in rows],
            }
        )
    return {
        "angle_deg": [cells[f"A{row}"] for row in rows],
        "traces": traces,
        "source": "https://doi.org/10.1038/s41586-024-07678-5",
        "source_file": str(path.relative_to(root)),
        "sheet": "Sheet1",
        "range": "A3:R76",
        "units": "detector signal in microvolts",
        "channel_headers": {"parallel": cells["B2"], "crossed": cells["K2"]},
        "channel_header_cells": {"parallel": "B2", "crossed": "K2"},
        "first_temperature_signal_columns": {"parallel": "B", "crossed": "K"},
        "quality_note": "Published duplicate 55-degree row preserved; no 360-degree row.",
    }


def calibrated_patterns(
    angle_deg: np.ndarray, chi: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Contract the Table S3 tensor for a y-oriented C2 domain, Eqs. 22-23.

    Return |P/(epsilon0 E^2)|^2 in (pm/V)^2. Susceptibilities are assumed real
    following the published static fit. This does not predict detector volts.
    """
    phi = np.deg2rad(angle_deg)
    c, s = np.cos(phi), np.sin(phi)
    px = 2 * chi["xxy"] * c * s
    py = chi["yxx"] * c * c + chi["yyy"] * s * s
    return abs(px * c + py * s) ** 2, abs(-px * s + py * c) ** 2


def static_fit_comparison(root: Path) -> dict:
    """Fit one common nonnegative detector scale; no offsets or angle rotations.

    Since chi comes from the authors' experimental fit, agreement is a
    reconstruction check, not independent validation of a material prediction.
    No error bars are supplied in this source table; do not infer chi-square.
    """
    source = read_static_source(root)
    metadata = json.loads((root / "data/nii2/gao2024_parameters.json").read_text())
    chi = metadata["static_chi_pm_per_v"]
    model = calibrated_patterns(np.array(source["angle_deg"]), chi)
    observed = source["traces"][0]
    y = np.concatenate([observed["parallel_uv"], observed["crossed_uv"]])
    prediction = np.concatenate(model)
    scale = max(0, float(np.dot(y, prediction) / np.dot(prediction, prediction)))
    residual = scale * prediction - y
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "interpretation": "Reconstruction using published fitted chi, not a quantum prediction",
        "temperature_k": observed["temperature_k"],
        "chi_source": "Supplementary Table S3",
        "fitted_parameters": [
            "one common detector scale; zero offset; zero angle rotation"
        ],
        "detector_scale_uv_per_chi_squared": scale,
        "rmse_uv": rmse,
        "relative_l2_error": float(np.linalg.norm(residual) / np.linalg.norm(y)),
        "angle_deg": source["angle_deg"],
        "observed_parallel_uv": observed["parallel_uv"],
        "observed_crossed_uv": observed["crossed_uv"],
        "reconstructed_parallel_uv": (scale * model[0]).tolist(),
        "reconstructed_crossed_uv": (scale * model[1]).tolist(),
        "source_file": source["source_file"],
        "source_sheet": "Sheet1",
        "source_range": "A3:R76",
        "source_quality_note": source["quality_note"],
    }
