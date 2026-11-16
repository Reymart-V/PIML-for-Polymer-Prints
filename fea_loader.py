"""Load the Abaqus interface temperature histories from Digambar's FEA run.

The CSVs, written by ``gcode_fea/odb_to_piml.py``, are in long format with one row per
(interface, position, time)::

    wall_id, interface_id, position_mm, time_s, temp_C

- ``interface_id`` is the upper layer number, so interface 64 is the bond between
  layers 63 and 64.
- ``time_s`` restarts at 0 for every interface, at the moment its upper cell is
  activated (deposited).
- Rows from different interfaces are interleaved in the file. This module groups
  them by interface and position, keeping each group in time order.

Two extractions of the same h = 15 FEA run exist:

- ``DEFAULT_THERMAL_CSV``: each history stops 25 s after deposition (0.7 M rows).
- ``LONG_THERMAL_CSV``: each history runs to 900 s after deposition, or to the end
  of the simulation (25 s after the last cell), whichever comes first
  (13.5 M rows, 0.8 GB).

The first load of a CSV parses it in chunks (about 30 s for the long file), then
caches the arrays next to it as ``<csv name>.cache.npz``. Later loads read the
cache in about a second. The cache is rebuilt automatically if the CSV changes.

Usage (from the repo root, where the notebooks run)::

    from fea_loader import LONG_THERMAL_CSV, load_activation_times, load_fea_thermal

    fea = load_fea_thermal()                       # 25 s histories
    fea_long = load_fea_thermal(LONG_THERMAL_CSV)  # 900 s histories
    t, T = fea.get(64, 38.8762)                    # time (s), temperature (degC)
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np

FEA_PACKAGE_DIR = (
    Path(__file__).resolve().parent / "Digambar Data" / "FFF_PLA_Wall_100mm_SYNTHETIC"
)
FEA_RUN_DIR = FEA_PACKAGE_DIR / "gcode_run"
DEFAULT_THERMAL_CSV = FEA_RUN_DIR / "thermal_history_FEA.csv"
LONG_THERMAL_CSV = FEA_PACKAGE_DIR / "thermal_history_FEA_long.csv"
DEFAULT_INTERFACE_MAP = FEA_RUN_DIR / "interface_map.csv"

_HEADER = ["wall_id", "interface_id", "position_mm", "time_s", "temp_C"]
_POSITION_DECIMALS = 4  # positions are written as e.g. 38.8762
_POS_SCALE = 10**_POSITION_DECIMALS  # positions are stored as integers of 1e-4 mm
_CACHE_VERSION = 1


class FEAThermalHistory:
    """Interface temperature histories, grouped by (interface_id, position_mm)."""

    def __init__(
        self,
        time_s: np.ndarray,
        temp_C: np.ndarray,
        group_iface: np.ndarray,
        group_pos: np.ndarray,
        group_start: np.ndarray,
        wall_id: str,
    ):
        """Arrays must already be grouped: group g is rows group_start[g]:group_start[g+1]."""
        self.wall_id = wall_id
        self._t = time_s
        self._T = temp_C
        stops = np.r_[group_start[1:], len(time_s)]
        self._slices = {
            (int(i), round(int(p) / _POS_SCALE, _POSITION_DECIMALS)): (int(a), int(b))
            for i, p, a, b in zip(group_iface, group_pos, group_start, stops)
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
            f"{self.n_rows:,} rows)"
        )

    @property
    def n_rows(self) -> int:
        return len(self._t)

    @property
    def interfaces(self) -> np.ndarray:
        """Sorted interface ids present in the file."""
        return np.unique([i for (i, _) in self._slices])

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
        """Return (time_s, temp_C) for one interface at one position, as float64 arrays.

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
        return self._t[a:b].astype(np.float64), self._T[a:b].astype(np.float64)


def _parse_csv(path: Path, wall_id: str | None, chunk_rows: int = 1_000_000):
    """Parse the CSV in chunks into compact arrays grouped by (interface, position)."""
    walls_found: set[str] = set()
    iface, pos, t, T = [], [], [], []
    with path.open(encoding="utf-8", newline="") as f:
        header = f.readline().strip().split(",")
        if header[:5] != _HEADER:
            raise ValueError(f"{path.name}: expected columns {_HEADER}, got {header}")
        while True:
            lines = list(itertools.islice(f, chunk_rows))
            if not lines:
                break
            prefix = lines[0][: lines[0].index(",") + 1]
            if all(ln.startswith(prefix) for ln in lines):  # usual case: one wall
                names = None
                walls_found.add(prefix[:-1])
            else:
                names = np.array([ln[: ln.index(",")] for ln in lines])
                walls_found.update(np.unique(names).tolist())
            num = np.loadtxt(lines, delimiter=",", usecols=(1, 2, 3, 4), ndmin=2)
            if wall_id is not None:
                if names is None:
                    keep = np.full(len(lines), prefix[:-1] == wall_id)
                else:
                    keep = names == wall_id
                num = num[keep]
            elif len(walls_found) > 1:
                raise ValueError(
                    f"{path.name} holds several walls {sorted(walls_found)}; "
                    "pass wall_id="
                )
            iface.append(num[:, 0].astype(np.int32))
            pos.append(np.rint(num[:, 1] * _POS_SCALE).astype(np.int64))
            t.append(num[:, 2])
            T.append(num[:, 3].astype(np.float32))
    if wall_id is None:
        wall_id = next(iter(walls_found))
    elif wall_id not in walls_found:
        raise ValueError(
            f"No rows for wall_id={wall_id!r}; available: {sorted(walls_found)}"
        )
    iface, pos, t, T = (np.concatenate(a) for a in (iface, pos, t, T))

    # Group by (interface, position). A stable sort keeps each group in file order,
    # which is time order; FEAThermalHistory checks that.
    key = iface.astype(np.int64) * 10**9 + pos
    order = np.argsort(key, kind="stable")
    key, t, T = key[order], t[order], T[order]
    group_start = np.flatnonzero(np.r_[True, np.diff(key) != 0])
    group_iface = (key[group_start] // 10**9).astype(np.int32)
    group_pos = key[group_start] % 10**9
    return t, T, group_iface, group_pos, group_start, wall_id


def load_fea_thermal(
    path: str | Path = DEFAULT_THERMAL_CSV,
    wall_id: str | None = None,
    use_cache: bool = True,
) -> FEAThermalHistory:
    """Read a long-format interface temperature CSV (see the module docstring).

    If the file holds more than one wall, pass ``wall_id`` to choose one.
    """
    path = Path(path)
    stat = path.stat()
    stamp = np.array([_CACHE_VERSION, stat.st_size, stat.st_mtime_ns], dtype=np.int64)
    suffix = "" if wall_id is None else f".{wall_id}"
    cache = path.with_name(f"{path.name}{suffix}.cache.npz")

    if use_cache and cache.exists():
        with np.load(cache, allow_pickle=False) as c:
            if np.array_equal(c["stamp"], stamp):
                return FEAThermalHistory(
                    c["t"],
                    c["T"],
                    c["group_iface"],
                    c["group_pos"],
                    c["group_start"],
                    str(c["wall_id"]),
                )

    t, T, g_iface, g_pos, g_start, wall = _parse_csv(path, wall_id)
    if use_cache:
        tmp = cache.with_name(cache.name + ".tmp.npz")
        np.savez(
            tmp,
            t=t,
            T=T,
            group_iface=g_iface,
            group_pos=g_pos,
            group_start=g_start,
            wall_id=np.array(wall),
            stamp=stamp,
        )
        tmp.replace(cache)
    return FEAThermalHistory(t, T, g_iface, g_pos, g_start, wall)


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
