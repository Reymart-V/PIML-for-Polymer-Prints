"""Load the Abaqus interface temperature histories from Digambar's FEA run.

The CSV, written by ``gcode_fea/odb_to_piml.py``, is in long format with one row per
(interface, position, time)::

    wall_id, interface_id, position_mm, time_s, temp_C

- ``interface_id`` is the upper layer number, so interface 64 is the bond between
  layers 63 and 64.
- ``time_s`` restarts at 0 for every interface, at the moment its upper cell is
  activated (deposited).
- Rows from different interfaces are interleaved in the file. This module groups
  them and sorts each group by time.

Usage (from the repo root, where the notebooks run)::

    from fea_loader import load_fea_thermal, load_activation_times
    fea = load_fea_thermal()          # about 1 s for the 714,849-row file
    t, T = fea.get(64, 38.8762)       # time (s) and temperature (degC) arrays
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

FEA_RUN_DIR = (
    Path(__file__).resolve().parent
    / "Digambar Data"
    / "FFF_PLA_Wall_100mm_SYNTHETIC"
    / "gcode_run"
)
DEFAULT_THERMAL_CSV = FEA_RUN_DIR / "thermal_history_FEA.csv"
DEFAULT_INTERFACE_MAP = FEA_RUN_DIR / "interface_map.csv"

_POSITION_DECIMALS = 4  # positions are written as e.g. 38.8762


class FEAThermalHistory:
    """Interface temperature histories, grouped by (interface_id, position_mm)."""

    def __init__(
        self,
        interface_id: np.ndarray,
        position_mm: np.ndarray,
        time_s: np.ndarray,
        temp_C: np.ndarray,
        wall_id: str,
    ):
        order = np.lexsort((time_s, position_mm, interface_id))
        self.wall_id = wall_id
        self._iface = interface_id[order].astype(int)
        self._pos = np.round(position_mm[order], _POSITION_DECIMALS)
        self._t = time_s[order]
        self._T = temp_C[order]

        # Start/stop index of each (interface, position) group in the sorted arrays.
        new_group = np.r_[True, (np.diff(self._iface) != 0) | (np.diff(self._pos) != 0)]
        starts = np.flatnonzero(new_group)
        stops = np.r_[starts[1:], len(self._t)]
        self._slices = {
            (int(self._iface[a]), float(self._pos[a])): (a, b)
            for a, b in zip(starts, stops)
        }

        for (i, x), (a, b) in self._slices.items():
            if b - a < 2 or np.any(np.diff(self._t[a:b]) <= 0):
                raise ValueError(
                    f"History for interface {i}, x={x} mm needs at least 2 "
                    "strictly increasing times"
                )

    def __len__(self) -> int:
        return len(self._slices)

    def __repr__(self) -> str:
        return (
            f"FEAThermalHistory(wall_id={self.wall_id!r}, {len(self)} histories, "
            f"interfaces {self.interfaces.min()}-{self.interfaces.max()}, "
            f"{len(self._t):,} rows)"
        )

    @property
    def interfaces(self) -> np.ndarray:
        """Sorted interface ids present in the file."""
        return np.unique(self._iface)

    def positions(self, interface_id: int) -> np.ndarray:
        """Sorted positions (mm from the wall's left end) recorded for one interface."""
        xs = [x for (i, x) in self._slices if i == interface_id]
        if not xs:
            raise KeyError(
                f"No interface {interface_id}; available: "
                f"{self.interfaces.min()}-{self.interfaces.max()}"
            )
        return np.array(sorted(xs))

    def keys(self) -> list[tuple[int, float]]:
        """All (interface_id, position_mm) pairs."""
        return sorted(self._slices)

    def get(
        self, interface_id: int, position_mm: float, tol: float = 1e-3
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (time_s, temp_C) for one interface at one position.

        ``position_mm`` must match a recorded position to within ``tol`` mm.
        Use ``positions(interface_id)`` to list the valid values.
        """
        xs = self.positions(interface_id)
        j = int(np.argmin(np.abs(xs - position_mm)))
        if abs(xs[j] - position_mm) > tol:
            raise KeyError(
                f"Interface {interface_id} has no position within {tol} mm of "
                f"{position_mm}; available: {xs.tolist()}"
            )
        a, b = self._slices[(interface_id, float(xs[j]))]
        return self._t[a:b].copy(), self._T[a:b].copy()


def load_fea_thermal(
    path: str | Path = DEFAULT_THERMAL_CSV, wall_id: str | None = None
) -> FEAThermalHistory:
    """Read a long-format interface temperature CSV.

    If the file holds more than one wall, pass ``wall_id`` to choose one.
    """
    path = Path(path)
    walls = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=str, ndmin=1)
    numbers = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 3, 4), ndmin=2)
    found = np.unique(walls)
    if wall_id is None:
        if len(found) != 1:
            raise ValueError(
                f"{path.name} holds several walls {found.tolist()}; pass wall_id="
            )
        wall_id = str(found[0])
    keep = walls == wall_id
    if not keep.any():
        raise ValueError(
            f"No rows for wall_id={wall_id!r}; available: {found.tolist()}"
        )
    iface, pos, t, T = numbers[keep].T
    return FEAThermalHistory(iface, pos, t, T, wall_id)


def load_activation_times(
    path: str | Path = DEFAULT_INTERFACE_MAP,
) -> dict[tuple[int, float], float]:
    """Solver time (s) at which each interface's upper cell was activated.

    Keys are (interface_id, position_mm), matching ``FEAThermalHistory``. The
    time between a layer's arrival and the next layer's arrival at the same x is
    ``times[(i + 1, x)] - times[(i, x)]``.
    """
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {
            (
                int(r["interface_id"]),
                round(float(r["position_mm"]), _POSITION_DECIMALS),
            ): float(r["activation_solver_time_s"])
            for r in csv.DictReader(f)
        }
