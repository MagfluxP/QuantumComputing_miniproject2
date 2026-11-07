"""Build the four-part notebook for the single current Gao-state hybrid."""

import argparse
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


md(r"""
# Quantum algorithm for second-harmonic generation
## Illustrative QSP and the Gao-state hybrid for NiI₂

A 1030 nm field generates second-harmonic light at 515 nm. This mini-project
implements quantum response circuits, tests them against classical calculations,
and asks whether the current material approximation reproduces measured SHG.

The notebook contains one illustrative model and one material model:

1. **Method:** implement the spectral-filter/QSP construction for illustrative
   optical levels with an independent density-matrix reference.
2. **Material calculation:** Gao's spin Hamiltonian and fixed magnetic parameters
   determine a correlated single-domain-conditioned density at 1.63 K. A retained
   electronic model supplies the optical response operator. Their explicit hybrid
   is the current supported material approximation.
3. **Comparison:** separate identical-input circuit accuracy, electronic reduction,
   experimental disagreement and computational costs; state the conditions for
   potential quantum advantage.
4. **Summary:** established results and unresolved physical/numerical limitations.

**Magnetic and optical Hamiltonians have different roles.** Gao supplies the
low-energy spin model; Fan's electronic energies and point-orbital dipoles remain
in the optical denominators. Calling every optical parameter a Gao input would
be incorrect. The resulting hybrid is not an author-validated optical NiI₂ model.

**Observable:** a complex microscopic coefficient β multiplying E₊² in the
2ω dipole, and its intensity proxy |β|². The experimental comparison fits one
shared detector gain and azimuth, so absolute bulk susceptibility is not tested.
A seven-site state-conditioning source and an incomplete optical basis remain
assumptions. No accurate bulk prediction or quantum speedup is established.

Sources: [Kharazi et al.](https://arxiv.org/html/2404.01454v1),
[Gao et al.](https://doi.org/10.1038/s41586-024-07678-5),
[Fan et al.](https://doi.org/10.1038/s41524-026-02184-w),
[Mou and Winter](https://arxiv.org/html/2602.02872v1).
""")
code(r"""
# Always load the source beside this notebook, including in a reused VS Code kernel.
import importlib
import inspect
import os
import sys
from pathlib import Path

_candidates=[]
if os.environ.get('NII2_PROJECT_ROOT'):
    _candidates.append(Path(os.environ['NII2_PROJECT_ROOT']).expanduser())
_notebook_path=globals().get('__vsc_ipynb_file__')
if _notebook_path:
    _notebook_parent=Path(_notebook_path).expanduser().resolve().parent
    _candidates.extend([_notebook_parent,*_notebook_parent.parents])
_candidates.extend([Path.cwd(),*Path.cwd().parents])
ROOT=next((p.resolve() for p in _candidates if (p/'pyproject.toml').is_file() and (p/'src/nii2_shg/model.py').is_file()),None)
if ROOT is None:
    raise RuntimeError('Open this notebook from miniproject2, or set NII2_PROJECT_ROOT to that folder before Run All.')
_project_source=ROOT/'src'
if str(_project_source) in sys.path:
    sys.path.remove(str(_project_source))
sys.path.insert(0,str(_project_source))
for _module_name in list(sys.modules):
    if _module_name=='nii2_shg' or _module_name.startswith('nii2_shg.'):
        del sys.modules[_module_name]
importlib.invalidate_caches()
from nii2_shg import material_response as _response_module
_required_response_keywords={'output_mu','initial_rho','second_gamma_factor'}
if not _required_response_keywords.issubset(inspect.signature(_response_module.classical_response).parameters):
    raise RuntimeError(f'The project code at {_response_module.__file__} is out of date; update miniproject2 before running this notebook.')
if Path(_response_module.__file__).resolve()!=(_project_source/'nii2_shg/material_response.py').resolve():
    raise RuntimeError('A different nii2_shg installation was loaded instead of this project source.')
classical_response = _response_module.classical_response
quantum_response = importlib.import_module('nii2_shg.response_circuit').quantum_response
print('Project source:',_project_source)
print('Classical response loaded from:',_response_module.__file__)

import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display, Image
from qiskit.quantum_info import Statevector
from nii2_shg.benchmarks import run_benchmark
from nii2_shg.circuits import dipole_block, overlap_circuit
from nii2_shg.model import OpticalModel, beta_from_products, classical_beta, photon_energy
from nii2_shg.qsp import design_window, qsp_filter, qsp_product, qsp_sequence, qsp_signal_circuit, scalar_qsp, window

OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
np.set_printoptions(precision=6,suppress=True)
plt.rcParams.update({'figure.dpi':120,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
model=OpticalModel(eta=1.)
fundamental_ev=photon_energy(1030)
gamma_toy_ev=.05
print(f'1030 nm fundamental: {fundamental_ev:.6f} eV; harmonic: 515 nm')
""")
md(r"""
# 1. Implement the spectral-filter method for illustrative SHG

This first part isolates algorithm correctness using known levels and dipoles.
It builds QSP filters and coherent optical products explicitly, then checks all
second-order terms against a classical reference. The levels are illustrative;
material-derived electronic parameters enter in Part 2. Numerical comparisons
use identical field, frequency, damping and normalization conventions.

## 1.1 Define the optical model and observable

The laser does not determine material levels. Use illustrative eigenenergies
$E_g=0$, $E_1=1.50$, $E_2=2.70$ eV, with an uncoupled padding state at 4 eV.
For one polarization, the active dipole is
$$
\mu=\begin{pmatrix}0&0.8&0.3s\\0.8&0&0.5\\0.3s&0.5&0\end{pmatrix}
\quad[e\,\text{Å}],
$$
where $s$ is dimensionless inversion breaking. The legacy model keyword `eta`
means this $s$ only. **It is unrelated to Mou's broadening eta in eV.**
At $s=0$, parity $(+,-,+)$ forbids the $g\leftrightarrow2$ transition.
These parameters are not fitted NiI₂ values.

Use $H(t)=H_0-\mu E(t)$ and $E(t)=E_+e^{-i\omega t}+\mathrm{c.c.}$.
Report $\beta$ multiplying $E_+^2$ in the dipole at $2\omega$, in
$(e\,\text{Å})^3/\mathrm{eV}^2$. The plotted $|\beta|^2$ is a microscopic
intensity proxy, not bulk $\chi^{(2)}$ in pm/V or detected optical power.
""")
code(r"""
print('H0 [eV]:\n', model.hamiltonian.real)
print('mu [e Angstrom]:\n', model.dipole.real)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for level, label in zip(model.energies[:3], ['ground', 'state 1', 'state 2']):
    ax.hlines(level, .2, .8, lw=3)
    ax.text(.85, level, f'{label}: {level:.2f} eV', va='center')
for height, xpos, label in [(fundamental_ev, 1.7, '1030 nm photon'), (2*fundamental_ev, 2.1, 'two photons')]:
    ax.annotate('', xy=(xpos, height), xytext=(xpos, 0), arrowprops={'arrowstyle':'->', 'color':'#bf7b26', 'lw':2})
    ax.text(xpos+.08, height/2, label, rotation=90, va='center')
ax.set(xlim=(0,2.5), ylim=(-.15,3.1), xticks=[], ylabel='Energy above ground [eV]', title='Illustrative optical states and laser energies')
fig.tight_layout(); fig.savefig(OUT/'levels.png', bbox_inches='tight'); plt.show()
""")
md(r"""
## 1.2 Dipole block encoding (paper II.3, III.1.1)

Expand $\mu=\sum c_jP_j$ in Pauli strings. PREP creates weights
$\sqrt{|c_j|/s_\mu}$, SELECT applies signed/complex Pauli gates, and PREP† combines
branches coherently. The accepted block equals $\mu/s_\mu$,
$s_\mu=\sum|c_j|$. No built-in spectroscopy method is called.
""")
code(r"""
dipole = dipole_block(model)
print('Ancillas:', dipole.ancillas, 'normalization:', dipole.scale)
print('Accepted-block error:', np.max(abs(dipole.accepted()*dipole.scale-model.dipole)))
display(dipole.circuit.draw(output='mpl'))
""")
md(r"""
## 1.3 Actual QSP phase sequences (paper II.4, III.1.2)

For a supplied window center $c$, rescale $\widetilde H=(H_0-cI)/R$ into $[-1,1]$.
Approximate an even erf-smoothed indicator by a bounded even Chebyshev polynomial
$p_d$. Fit phases in
$$
U_\Phi(x)=e^{i\phi_0Z}W(x)e^{i\phi_1Z}\cdots W(x)e^{i\phi_dZ},\qquad
W(x)=\begin{pmatrix}x&\sqrt{1-x^2}\\-\sqrt{1-x^2}&x\end{pmatrix}.
$$
The fit targets $\operatorname{Re}[U_{00}(x)]=p_d(x)$ on Chebyshev nodes and is
validated on independent grids. Extrema bound the target polynomial. The filter
uses an LCU of $U_\Phi$ and $U_\Phi^\dagger$ to select its Hermitian part.
The implementation explicitly builds the signal gates and phase rotations.
NumPy/SciPy support polynomial arithmetic and numerical optimization; no complete
QSP solver is called. This numerical fitter is for small degrees, not a general
high-degree synthesis guarantee.

**Scope:** windows are supplied around known toy energies; the paper's adaptive
spectral discovery, scalable chemistry oracles and amplitude estimation are not
implemented. A small diagonal square root constructs our signal oracle. The material calculation in Part 2 uses a separate small-instance resolvent circuit.
""")
code(r"""
design = design_window(32)
x = np.linspace(-1,1,1001)
fig, ax = plt.subplots(figsize=(7,3.3))
ax.plot(x, window(x), 'k--', label='Smoothed target')
ax.plot(x, design.polynomial(x), label='Degree-32 polynomial')
ax.plot(x[::30], scalar_qsp(x[::30], design.phases), '.', label='QSP phase sequence')
ax.set(xlabel='Rescaled energy relative to window center', ylabel='Accepted amplitude', title='Energy-window approximation')
ax.legend(); fig.tight_layout(); fig.savefig(OUT/'qsp_filter.png', bbox_inches='tight'); plt.show()
print('Phase synthesis grid error:', design.synthesis_error)
print('Polynomial vs smooth-window grid error:', design.window_error)
raw_qsp, _ = qsp_signal_circuit(model, 1.5, 4)
display(raw_qsp.draw(output='mpl'))
""")
md(r"""
## 1.4 Coherent dipole products and Hadamard readout (paper III.2)

Target $T_{21}=\langle g|\mu P_2\mu P_1\mu|g\rangle=0.12s$ for ideal filters.
Use approximate QSP windows and retain all normalization factors. Fresh ancilla
registers prevent failed branches from returning to the accepted block. All-zero
statevector amplitude yields the complex overlap; an actual Hadamard test checks
its real part. No intermediate measurements occur. Little endian ordering:
system q0,q1; work registers follow; optional Hadamard control is last.
""")
code(r"""
product, scale, coherent = qsp_product(model, degree=32)
exact_product = model.dipole[0,2]*model.dipole[2,1]*model.dipole[1,0]  # analytic transition product
print('QSP product:', product, 'Exact product:', exact_product)
print('Absolute filter-induced product error:', abs(product-exact_product))
print('Normalization:', scale, 'product-circuit qubits:', coherent.num_qubits)
hadamard = overlap_circuit(qsp_sequence(model, degree=32), 'real')
prob = Statevector.from_instruction(hadamard).probabilities([hadamard.num_qubits-1])
assert np.isclose(scale*(prob[0]-prob[1]), product.real, atol=1e-9)
print('Explicit Hadamard check:', scale*(prob[0]-prob[1]))
# For Hermitian mu/filters, the reversed overlap is its conjugate.
products = {(2,1): product, (1,2): product.conjugate()}
""")
md(r"""
## 1.5 Assemble full second-order response and validate independently

With $D_k(a,b)=k\hbar\omega-(E_a-E_b)+i\Gamma$, sum
$$
\beta=\sum_{mn}T_{mn}\left[
\frac1{D_2(m,g)D_1(n,g)}-\frac1{D_2(n,m)D_1(g,m)}
-\frac1{D_2(n,m)D_1(n,g)}+\frac1{D_2(g,n)D_1(g,m)}\right].
$$
All four commutator terms are retained. Only two ordered triples are nonzero for
this zero-diagonal model; exploiting that known sparsity is a toy assumption.
An independent reference evaluates density-matrix perturbation theory directly:
$\rho_1=-(\hbar\omega-L+i\Gamma)^{-1}[\mu,\rho_0]$,
$\rho_2=-(2\hbar\omega-L+i\Gamma)^{-1}[\mu,\rho_1]$,
$\beta=\mathrm{Tr}(\mu\rho_2)$ with $L\rho=[H_0,\rho]$.
""")
code(r"""
beta = beta_from_products(model, products, fundamental_ev, gamma_toy_ev)
reference = classical_beta(model, fundamental_ev, gamma_toy_ev)
print('1030 nm QSP-derived beta:', beta)
print('Independent reference:  ', reference)
print('Relative numerical error:', abs(beta-reference)/abs(reference))
wavelengths = np.linspace(700,1600,401)
quantum_curve = np.array([beta_from_products(model, products, photon_energy(w), gamma_toy_ev) for w in wavelengths])
classical_curve = np.array([classical_beta(model, photon_energy(w), gamma_toy_ev) for w in wavelengths])
fig, axes = plt.subplots(1,2,figsize=(10,3.5))
axes[0].plot(wavelengths, quantum_curve.real, label='Re beta, QSP products')
axes[0].plot(wavelengths, quantum_curve.imag, label='Im beta, QSP products')
axes[0].plot(wavelengths[::16], classical_curve.real[::16], 'k.', label='Classical reference Re beta')
axes[0].set(ylabel='beta [(e Angstrom)^3/eV^2]'); axes[0].legend(fontsize=8)
axes[1].plot(wavelengths, abs(quantum_curve)**2, label='QSP')
axes[1].plot(wavelengths, abs(classical_curve)**2, '--', label='Classical')
axes[1].set(ylabel='|beta|^2 [model units]'); axes[1].legend()
for ax in axes:
    ax.axvline(1030, color='gray', ls=':'); ax.set(xlabel='Fundamental wavelength [nm]')
fig.suptitle('Illustrative SHG: no NiI2 material prediction')
fig.tight_layout(); fig.savefig(OUT/'shg_response.png', bbox_inches='tight'); plt.show()
""")

from material_sections import append_material_sections

cells = append_material_sections(cells)
from submission_sections import add_submission_sections

cells = add_submission_sections(cells)
notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook.metadata.language_info = {"name": "python", "version": "3.12"}
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path)
args = parser.parse_args()
destination = args.output or ROOT / "notebooks/01_shg_walkthrough.ipynb"
nbf.write(notebook, destination)
subprocess.run(
    [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--no-cache",
        str(destination),
        "--fix",
        "--unsafe-fixes",
    ],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "ruff", "format", "--no-cache", str(destination)], check=True
)
print(destination)
