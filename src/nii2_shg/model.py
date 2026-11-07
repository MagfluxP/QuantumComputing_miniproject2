"""Three optical states; numerical parameters are illustrative, not NiI2 data."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complex128]


@dataclass(frozen=True)
class OpticalModel:
    """Effective model in eV and e Angstrom; index 3 is an uncoupled padding state."""

    eta: float = 1.0

    @property
    def inversion_breaking(self) -> float:
        """Dimensionless toy parameter; unrelated to Mou et al. broadening eta."""
        return self.eta

    @property
    def energies(self) -> NDArray[np.float64]:
        return np.array([0.0, 1.5, 2.7, 4.0])

    @property
    def hamiltonian(self) -> Array:
        return np.diag(self.energies).astype(complex)

    @property
    def dipole(self) -> Array:
        return np.array(
            [
                [0, 0.8, 0.3 * self.eta, 0],
                [0.8, 0, 0.5, 0],
                [0.3 * self.eta, 0.5, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=complex,
        )

    @property
    def scaled_hamiltonian(self) -> Array:
        """Affine scaling puts all four energies in [-1, 1]."""
        return (self.hamiltonian - 2.0 * np.eye(4)) / 2.0


def photon_energy(wavelength_nm: float) -> float:
    """Return photon energy in eV; wavelength must be positive."""
    if wavelength_nm <= 0:
        raise ValueError("Wavelength must be positive")
    return 1239.841984 / wavelength_nm


def classical_beta(
    model: OpticalModel, energy_ev: float, gamma_ev: float = 0.05
) -> complex:
    """Independent density-matrix second-order coefficient in (e A)^3/eV^2.

    H(t)=H0-mu E(t); E(t)=Eplus exp(-i wt)+c.c.; coefficient of Eplus^2.
    Positive uniform phenomenological damping is used in each resolvent.
    No density conversion, Fresnel factors, or propagation is included.
    """
    if gamma_ev <= 0:
        raise ValueError("Broadening must be positive")
    mu = model.dipole
    rho0 = np.zeros((4, 4), complex)
    rho0[0, 0] = 1
    gaps = model.energies[:, None] - model.energies[None, :]
    rho1 = -(mu @ rho0 - rho0 @ mu) / (energy_ev - gaps + 1j * gamma_ev)
    rho2 = -(mu @ rho1 - rho1 @ mu) / (2 * energy_ev - gaps + 1j * gamma_ev)
    return complex(np.trace(mu @ rho2))


def beta_from_products(
    model: OpticalModel,
    products: dict[tuple[int, int], complex],
    energy_ev: float,
    gamma_ev: float = 0.05,
) -> complex:
    """Assemble four commutator terms using T[m,n]=mu[g,m] mu[m,n] mu[n,g].

    This model has zero diagonal dipoles; only m,n in {1,2}, m!=n contribute.
    Same field convention as classical_beta, with an independent implementation.
    """
    if gamma_ev <= 0:
        raise ValueError("Broadening must be positive")
    energies = model.energies

    def den(order: int, a: int, b: int) -> complex:
        return order * energy_ev - (energies[a] - energies[b]) + 1j * gamma_ev

    result = 0j
    for (m, n), weight in products.items():
        result += weight * (
            1 / (den(2, m, 0) * den(1, n, 0))
            - 1 / (den(2, n, m) * den(1, 0, m))
            - 1 / (den(2, n, m) * den(1, n, 0))
            + 1 / (den(2, 0, n) * den(1, 0, m))
        )
    return complex(result)
