# CLAUDE.md

## Context

Undergrad research in Dr. Austin Downey's ARTS Lab (USC). Collaborator:
Digambar Killedar (PhD student). Official project name: "PIML for FFF and LPBF."

Currently blocked on FEA thermal data from Digambar (requested since late
September). The FD solver exists to make progress without it.

Scope is FFF (polymer) only — LPBF is future work, per Dr. Downey.
Target quantity is interlayer bond strength via time above T_g.
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A physics-informed ML project on the thermal history of FFF (3D-printed) thin walls. The quantity of interest is
the temperature at the interface between a freshly deposited layer and the one below it, and in particular
**time above T_g** (glass transition), which controls interlayer bond strength. The project is two Jupyter
notebooks, with no package, tests or build.

- [fd_solver.ipynb](fd_solver.ipynb): an explicit FTCS finite-difference solver for 1D transient conduction
  through the build direction (z). It produces stand-in reference data until real FEA results (from Digambar)
  arrive.
- [pinn_thermal.ipynb](pinn_thermal.ipynb): a PyTorch PINN (MLP, tanh, (ẑ, t̂) → T̂) trained on the FD output
  with a data loss plus a weighted PDE-residual loss.

## Environment and running

- The Python 3.14 venv is in `.venv/` (the notebook kernel is `.venv (3.14.3)`) and has numpy 2.x,
  matplotlib, torch and ipykernel. There is no requirements file. `.venv-1/` is a duplicate venv that is not
  gitignored, so don't commit it.
- Notebooks are run cell by cell in VS Code. `nbconvert`/`nbclient` are **not** installed, so headless
  execution needs `.venv/Scripts/pip install nbconvert` first, then
  `.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace <nb>.ipynb`.
- Run order matters. `fd_solver.ipynb` writes `data/baseline.npz` and `data/sweep.npz` (both gitignored), and
  `pinn_thermal.ipynb` loads `data/baseline.npz`. Paths are relative, so run with the repo root as the working
  directory.

## Data contract between the notebooks

`data/baseline.npz` is the interface that the FEA data will eventually replace. The plan is that swapping in
FEA changes **only the loading cell** of the PINN notebook, so keep this key set stable:
`z (nz,)`, `t (n_t,)`, `T (n_t, nz)` in °C, plus scalars `k, rho, cp, alpha, H, LH, T_nozzle, T_prev, T_bed,
T_inf, h, Tg`.

`data/sweep.npz` holds `params (27, 3)` = [T_nozzle, layer_height, h], `t_above (27,)`,
`fields (27, n_t, nz)`, and a single shared `z`, `t`. The shared time grid is valid only because the FD time
step is `dt = safety·dz²/α`, which depends on `H`, `nz`, `k`, `rho` and `cp` and not on the swept parameters.
If the sweep ever varies any of those, the per-run grids differ, `np.array(fields)` breaks, and the saved `t`
is wrong.

## Physics and numerics conventions

- BCs: Dirichlet at the bed (z=0, `T_bed`) and convection at the top (z=H, `h`, `T_inf`), implemented with a
  ghost node, which is where the factor of 2 in the last-node update comes from. IC: `T_nozzle` for
  z ≥ H−LH, `T_prev` below, and the bed node at `T_bed`.
- The interface node is `argmin(|z − (H − LH)|)`, and both notebooks recompute it that way.
- The PINN works in non-dimensional variables: ẑ = z/H, t̂ = t/t_end, T̂ = (T − T_inf)/(T_nozzle − T_inf). The
  PDE becomes ∂T̂/∂t̂ = Fo·∂²T̂/∂ẑ² with Fo = α·t_end/H². The PINN loss has **no explicit IC/BC terms**, and the
  data loss is what pins them. `lambda_phys` is the main tuning knob.
- Material defaults are PLA placeholders, to be replaced with the lab's filament values.

## NumPy version

The venv has NumPy 2.x, where `np.trapz` no longer exists. Use `np.trapezoid`.

## Roadmap (from the notebooks)

Stage 1: the PINN reproduces the FD field (current). Stage 2: print settings (T_nozzle, layer_height) as extra
network inputs, trained on `sweep.npz`. Stage 3: inverse problem, making `h` a learnable `nn.Parameter` and
recovering it from data. Other planned steps are a data-free forward solve (physics + IC/BC losses only) and a
`lambda_phys` sensitivity sweep.
