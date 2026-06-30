# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate the PR baseline: TORAX + gyaradax-QL vs TORAX + QLKNN.

Runs the ITER hybrid predictor-corrector scenario (CHEASE equilibrium) under a
deliberately simplified, like-for-like setup -- only the ion-heat channel is
evolved, for *both* transport models -- and writes the artifacts behind the
PR's "early results": a headline panel, a profile overlay, a JSON summary, and
the raw arrays.

The gyaradax-QL model here is adiabatic-electrostatic, so only q_i (hence T_i)
is a genuine prediction; the plugin sets q_e = q_i and the particle flux to
zero. Evolving T_i alone for both models keeps the overlay honest.

Must run under the TORAX venv, which imports TORAX from its source tree:

    /system/user/publicwork/galletti/git/torax/.venv/bin/python \
        experiments/run_adiabatic_iterhybrid_predictor_corrector.py --device 0
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import os
import sys
import time

# This script lives in <torax_repo>/experiments/; the repo root is its parent.
# Putting it on sys.path lets `import torax` resolve from the source tree
# regardless of the current working directory.
TORAX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RHO_MATCH = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GYARADAX_LABEL, QLKNN_LABEL = "gyaradax-QL", "QLKNN"

# Clip / patch / domain keys copied from the stock qlknn transport block so both
# models see identical edge handling and the comparison isolates the turbulent
# core flux.
_INHERITED_TRANSPORT_KEYS = (
    "chi_min", "chi_max", "D_e_min",
    "apply_inner_patch", "apply_outer_patch",
    "chi_i_inner", "chi_e_inner", "chi_i_outer", "chi_e_outer",
    "D_e_inner", "D_e_outer", "V_e_inner", "V_e_outer",
    "rho_inner", "rho_outer",
)

# Representative ITG flux-tube for the standalone per-radius diagnostic.
_DIAG_FLUX_TUBE = dict(rlt=8.0, rln=2.5, q=2.0, shat=1.0, eps=0.18)
_GRAD_N_STEPS = 60  # short linear solve for the AD smoke test

log = logging.getLogger("baseline")


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Knobs for one baseline run. Grid defaults match the calibrated Cn head."""

    out_dir: str
    t_final_rel: float
    rho_match: tuple[float, ...]
    cn_calibration_path: str
    run_diagnostics: bool
    nvpar: int = 32
    nmu: int = 8
    ns: int = 16
    nkx: int = 43
    nky: int = 16
    ikxspace: int = 5
    n_steps_linear: int = 200

    @property
    def linear_grid(self) -> dict:
        return dict(
            nvpar=self.nvpar, nmu=self.nmu, ns=self.ns, nkx=self.nkx,
            nky=self.nky, ikxspace=self.ikxspace, n_steps_linear=self.n_steps_linear,
        )


@dataclasses.dataclass
class Comparison:
    """Both simulation outputs plus their wall-clock times."""

    gyaradax: object
    qlknn: object
    t_gyaradax: float
    t_qlknn: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="0", help="GPU index (CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--t-final", type=float, default=1.0,
                        help="seconds of T_i relaxation past t_initial")
    parser.add_argument("--rho-match", default=",".join(map(str, DEFAULT_RHO_MATCH)),
                        help="comma-separated rho_match radii")
    parser.add_argument("--cn", default="auto",
                        help="cn_calibration_path: 'auto', a registry name, "
                             "'none', or a path to a pickled head")
    parser.add_argument("--skip-diagnostics", action="store_true")
    return parser.parse_args()


def _configure_runtime(args: argparse.Namespace) -> None:
    """Pin the GPU and locate TORAX. Must run before JAX is imported."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if TORAX_ROOT not in sys.path:
        sys.path.insert(0, TORAX_ROOT)


# Runtime is configured at import time because JAX reads CUDA_VISIBLE_DEVICES on
# first import; the heavy imports below must follow it.
_ARGS = _parse_args()
_configure_runtime(_ARGS)

import jax  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import torax  # noqa: E402
from torax.examples import iterhybrid_predictor_corrector  # noqa: E402
from torax._src.transport_model import pydantic_model  # noqa: E402


def _evolve_ion_heat_only(config_dict: dict, t_final_rel: float) -> dict:
    """Restrict a TORAX config to a fixed-duration ion-heat-only relaxation."""
    numerics = config_dict["numerics"]
    t_initial = float(numerics.get("t_initial", 0.0))
    numerics.update(
        evolve_ion_heat=True, evolve_electron_heat=False,
        evolve_density=False, evolve_current=False,
        t_final=t_initial + t_final_rel,
    )
    return config_dict


def build_configs(exp: ExperimentConfig):
    """Build the gyaradax-QL and QLKNN ToraxConfigs for the same scenario."""
    base = iterhybrid_predictor_corrector.CONFIG
    inherited = {key: base["transport"][key] for key in _INHERITED_TRANSPORT_KEYS}

    gyaradax_dict = _evolve_ion_heat_only(copy.deepcopy(base), exp.t_final_rel)
    gyaradax_dict["transport"] = {
        "model_name": "gyaradax-ql",
        "backend": "jax",
        "rho_match": exp.rho_match,
        "cn_calibration_path": exp.cn_calibration_path,
        **exp.linear_grid,
        **inherited,
        "rho_min": inherited["rho_inner"],
        "rho_max": inherited["rho_outer"],
    }
    qlknn_dict = _evolve_ion_heat_only(copy.deepcopy(base), exp.t_final_rel)
    return (
        torax.ToraxConfig.from_dict(gyaradax_dict),
        torax.ToraxConfig.from_dict(qlknn_dict),
        gyaradax_dict,
    )


def run_timed(torax_config):
    """Run one simulation, returning (data_tree, wall_clock_seconds)."""
    start = time.perf_counter()
    data_tree, _ = torax.run_simulation(torax_config)
    return data_tree, time.perf_counter() - start


def _final_scalar(data_tree, name: str) -> float:
    return float(getattr(data_tree.scalars, name).values[-1])


def _final_profile(data_tree, name: str):
    return getattr(data_tree.profiles, name).values[-1]


def summarize(comparison, gyaradax_config, gyaradax_dict, exp):
    """Collect the headline numbers and run metadata into a JSON-able dict."""
    transport = gyaradax_config.transport.build_transport_model()

    def channel(name: str) -> dict:
        g = _final_scalar(comparison.gyaradax, name)
        q = _final_scalar(comparison.qlknn, name)
        return {"gyaradax_ql": g, "qlknn": q, "rel_delta_pct": 100.0 * (g - q) / q}

    return {
        "scenario": "iterhybrid_predictor_corrector (CHEASE, T_i-only, adiabatic-ES)",
        "rho_match": list(exp.rho_match),
        "t_final_s": float(gyaradax_dict["numerics"]["t_final"]),
        "linear_grid": exp.linear_grid,
        "cn_head": type(transport.cn_head).__name__,
        "cn_scalar": float(transport.cn_scalar),
        "Q_fusion": channel("Q_fusion"),
        "tau_E": channel("tau_E"),
        "wallclock_s": {
            "gyaradax_ql": comparison.t_gyaradax,
            "qlknn": comparison.t_qlknn,
        },
        "jax_devices": [str(d) for d in jax.devices()],
    }


def save_arrays(out_dir: str, comparison) -> None:
    """Persist the arrays behind the figures for downstream re-plotting."""
    g, q = comparison.gyaradax, comparison.qlknn
    arrays = {
        "rho_cell": g.profiles.rho_norm.values,
        "rho_face": g.profiles.rho_face_norm.values,
        "time_gyaradax": g.scalars.time.values,
        "time_qlknn": q.scalars.time.values,
        "Q_fusion_gyaradax": g.scalars.Q_fusion.values,
        "Q_fusion_qlknn": q.scalars.Q_fusion.values,
    }
    for key in ("T_i", "T_e", "n_e", "chi_turb_i"):
        arrays[f"{key}_gyaradax"] = _final_profile(g, key)
        arrays[f"{key}_qlknn"] = _final_profile(q, key)
    np.savez(os.path.join(out_dir, "profiles.npz"), **arrays)


def _overlay(ax, x, comparison, key, ylabel, title=None) -> None:
    """Overlay the QLKNN reference and the gyaradax-QL prediction for one field."""
    ax.plot(x, _final_profile(comparison.qlknn, key), "C0-", lw=2.2, label=QLKNN_LABEL)
    ax.plot(x, _final_profile(comparison.gyaradax, key), "C3--", lw=2.2, label=GYARADAX_LABEL)
    ax.set_xlabel(r"$\rho_{\mathrm{norm}}$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, fontsize=11)


def plot_profiles(out_dir: str, comparison) -> None:
    """Four-panel final-profile overlay (chi_i, T_i, T_e, n_e)."""
    rho_face = comparison.gyaradax.profiles.rho_face_norm.values
    rho_cell = comparison.gyaradax.profiles.rho_norm.values
    panels = [
        ("chi_turb_i", r"$\chi_i$ [m$^2$/s]", rho_face),
        ("T_i", r"$T_i$ [keV]", rho_cell),
        ("T_e", r"$T_e$ [keV]", rho_cell),
        ("n_e", r"$n_e$ [$10^{20}$ m$^{-3}$]", rho_cell),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (key, ylabel, x) in zip(axes.ravel(), panels):
        _overlay(ax, x, comparison, key, ylabel)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "profiles.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_headline(out_dir: str, comparison) -> None:
    """Headline row: ion temperature, ion heat diffusivity, fusion gain."""
    g, q = comparison.gyaradax, comparison.qlknn
    rho_cell = g.profiles.rho_norm.values
    rho_face = g.profiles.rho_face_norm.values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _overlay(axes[0], rho_cell, comparison, "T_i", r"$T_i$ [keV]", "Ion temperature")
    axes[0].legend(fontsize=10)
    _overlay(axes[1], rho_face, comparison, "chi_turb_i",
             r"$\chi_{\mathrm{turb},i}$ [m$^2$/s]", "Ion heat diffusivity")

    ax = axes[2]
    ax.plot(q.scalars.time.values, q.scalars.Q_fusion.values, "C0-o", lw=2.2, ms=5)
    ax.plot(g.scalars.time.values, g.scalars.Q_fusion.values, "C3--s", lw=2.2, ms=5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$Q_{\mathrm{fusion}}$")
    ax.set_title("Fusion gain", fontsize=11)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "headline.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_datatrees(out_dir: str, comparison) -> None:
    """Persist both runs as NetCDF so they can be re-plotted without re-running."""
    comparison.gyaradax.to_netcdf(os.path.join(out_dir, "gyaradax_ql.nc"))
    comparison.qlknn.to_netcdf(os.path.join(out_dir, "qlknn.nc"))


def plot_torax_native(out_dir: str, comparison) -> None:
    """Save TORAX's own multipanel comparison (overview + transport plot configs).

    `plot_run_from_data_tree` returns a plotly figure; HTML is always written,
    PNG only if `kaleido` is installed.
    """
    from torax.plotting.configs import default_plot_config, transport_plot_config

    data_trees = {GYARADAX_LABEL: comparison.gyaradax, QLKNN_LABEL: comparison.qlknn}
    panels = (("torax_overview", default_plot_config.PLOT_CONFIG),
              ("torax_transport", transport_plot_config.PLOT_CONFIG))
    for name, plot_config in panels:
        fig = torax.plot_run_from_data_tree(
            plot_config=plot_config, data_trees=data_trees, interactive=False,
            fig_title="ITER hybrid: gyaradax-QL vs QLKNN")
        fig.write_html(os.path.join(out_dir, f"{name}.html"))
        try:
            fig.write_image(os.path.join(out_dir, f"{name}.png"), scale=2)
        except Exception as exc:  # noqa: BLE001 - kaleido is an optional dependency
            log.info("%s.png skipped (%s: %s); HTML written",
                     name, type(exc).__name__, exc)


def plot_profile_evolution(out_dir: str, comparison) -> None:
    """2D (rho, time) heatmaps of the time-evolving channels for each model.

    Only T_i and chi_turb,i vary in time in the T_i-only setup (T_e / n_e are
    prescribed). This is the static analogue of TORAX's interactive time slider.
    """
    g, q = comparison.gyaradax, comparison.qlknn
    grids = {"cell": g.profiles.rho_norm.values, "face": g.profiles.rho_face_norm.values}
    channels = (("T_i", "cell", r"$T_i$ [keV]"),
                ("chi_turb_i", "face", r"$\chi_{\mathrm{turb},i}$ [m$^2$/s]"))
    models = ((QLKNN_LABEL, q), (GYARADAX_LABEL, g))

    fig, axes = plt.subplots(len(channels), len(models), figsize=(11, 7), squeeze=False)
    for row, (key, grid, clabel) in enumerate(channels):
        rho = grids[grid]
        fields = [(label, getattr(m.profiles, key)) for label, m in models]
        vmax = float(np.percentile(
            np.concatenate([f.values.ravel() for _, f in fields]), 98))
        for col, (label, field) in enumerate(fields):
            t = field["time"].values
            z = field.values
            if z.shape != (t.size, rho.size):
                z = z.T
            ax = axes[row, col]
            mesh = ax.pcolormesh(rho, t, z, shading="auto", cmap="magma", vmin=0, vmax=vmax)
            ax.set_xlabel(r"$\rho_{\mathrm{norm}}$")
            ax.set_ylabel("t [s]")
            ax.set_title(f"{label}: {clabel}", fontsize=10)
            fig.colorbar(mesh, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "profile_evolution.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def per_radius_diagnostic(gyaradax_config) -> dict:
    """Run one linear gyaradax solve at a representative ITG flux-tube.

    Exercises the plugin's per-radius path directly to report the per-ky growth
    rate spectrum and the calibrated QL heat flux for a known operating point.
    """
    import jax.numpy as jnp
    from gyaradax.integrals import calculate_fluxes, geom_tensors
    from gyaradax.params import GKParams
    from gyaradax.quasilinear.saturation import ql_flux_diagnostics
    from gyaradax.solver import default_state, gksolve, linear_precompute
    from torax._src.transport_model.gyaradax_based_transport_model import gyaradax_geometry_at
    from torax._src.transport_model.gyaradax_ql_transport_model import GyaradaxQLTransportModel

    transport = gyaradax_config.transport
    model = GyaradaxQLTransportModel.from_config(transport)
    params = dataclasses.replace(
        GKParams(), beta=jnp.asarray(0.0),
        adiabatic_electrons=True, non_linear=False, disable_per_ky_norm=True, dt=0.005,
        **{k: jnp.asarray(v) for k, v in _DIAG_FLUX_TUBE.items()})
    geom = gyaradax_geometry_at(q=params.q, shat=params.shat, eps=params.eps,
                                config=transport, topology=model.topology)
    df, (phi, _), state = gksolve(
        model._initial_df(), geom, params, default_state(nky=transport.nky),
        n_steps=transport.n_steps_linear, pre=linear_precompute(geom, params))
    df.block_until_ready()

    _, eflux_kxy, _ = calculate_fluxes(geom_tensors(geom), df, phi, reduce=False)
    weights = jnp.asarray(geom["ints"])
    phi2 = jnp.abs(phi) ** 2
    little_g = jnp.asarray(geom["little_g"])
    diag = ql_flux_diagnostics(
        growth_rate=state.last_growth_rate, phi2=phi2,
        phi2_kxy=jnp.sum(phi2 * weights[:, None, None], axis=0), flux_kxy=eflux_kxy,
        krho=jnp.asarray(geom["krho"]), kxrh=jnp.asarray(geom["kxrh"]),
        little_g=little_g if little_g.shape[0] == 3 else little_g.T, ds=jnp.mean(weights))
    return {
        "flux_tube": _DIAG_FLUX_TUBE,
        "gamma_ky": np.asarray(state.last_growth_rate).tolist(),
        "Q_QL": float(diag["Q_QL"]),
    }


def gradient_check(gyaradax_config) -> dict:
    """Differentiate q_i w.r.t. R/L_T through the pure-JAX plugin path."""
    import jax.numpy as jnp
    from gyaradax.params import GKParams
    from torax._src.transport_model.gyaradax_based_transport_model import gyaradax_geometry_at
    from torax._src.transport_model.gyaradax_ql_transport_model import (
        GyaradaxQLConfig, GyaradaxQLTransportModel)

    # early_stop=False -> the differentiated solve is a plain lax.scan.
    config = GyaradaxQLConfig(
        rho_match=gyaradax_config.transport.rho_match, backend="jax",
        n_steps_linear=_GRAD_N_STEPS, early_stop=False, cn_calibration_path="auto")
    model = GyaradaxQLTransportModel.from_config(config)
    fixed = {k: jnp.asarray(v) for k, v in _DIAG_FLUX_TUBE.items() if k != "rlt"}

    def qi_of_rlt(rlt):
        params = dataclasses.replace(
            GKParams(), rlt=rlt, beta=jnp.asarray(0.0), adiabatic_electrons=True,
            non_linear=False, disable_per_ky_norm=True, dt=0.005, **fixed)
        geom = gyaradax_geometry_at(q=params.q, shat=params.shat, eps=params.eps,
                                    config=config, topology=model.topology)
        qi, _, _ = model._gyaradax_ql_at_radius(params, geom)
        return qi

    derivative = float(jax.jit(jax.grad(qi_of_rlt))(jnp.asarray(_DIAG_FLUX_TUBE["rlt"])))
    return {"d_qi_d_rlt": derivative, "n_steps_linear": _GRAD_N_STEPS}


def run_diagnostics(gyaradax_config) -> dict:
    """Optional standalone checks; failures are recorded, not fatal."""
    results = {}
    for name, fn in (("per_radius", per_radius_diagnostic),
                     ("gradient", gradient_check)):
        try:
            results[name] = fn(gyaradax_config)
        except Exception as exc:  # noqa: BLE001 - diagnostics must not abort the run
            log.warning("%s diagnostic failed: %s: %s", name, type(exc).__name__, exc)
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    exp = ExperimentConfig(
        out_dir=_ARGS.out_dir,
        t_final_rel=_ARGS.t_final,
        rho_match=tuple(float(x) for x in _ARGS.rho_match.split(",")),
        cn_calibration_path=_ARGS.cn,
        run_diagnostics=not _ARGS.skip_diagnostics,
    )
    os.makedirs(exp.out_dir, exist_ok=True)

    log.info("JAX devices: %s", jax.devices())
    log.info("gyaradax-ql registered: %s",
             "GyaradaxQLConfig" in str(pydantic_model.CombinedCompatibleTransportModel))

    gyaradax_config, qlknn_config, gyaradax_dict = build_configs(exp)
    log.info("rho_match=%s  t_final=%.2fs  (T_i only, both models)",
             exp.rho_match, gyaradax_dict["numerics"]["t_final"])

    log.info("running gyaradax-QL (first JIT pass is a few minutes) ...")
    data_g, t_g = run_timed(gyaradax_config)
    log.info("gyaradax-QL done in %.1fs; running QLKNN reference ...", t_g)
    data_q, t_q = run_timed(qlknn_config)
    comparison = Comparison(data_g, data_q, t_g, t_q)

    summary = summarize(comparison, gyaradax_config, gyaradax_dict, exp)
    with open(os.path.join(exp.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    save_arrays(exp.out_dir, comparison)
    plot_profiles(exp.out_dir, comparison)
    plot_headline(exp.out_dir, comparison)
    save_datatrees(exp.out_dir, comparison)
    plot_torax_native(exp.out_dir, comparison)
    plot_profile_evolution(exp.out_dir, comparison)
    log.info("\n%s", json.dumps(summary, indent=2))

    if exp.run_diagnostics:
        log.info("running diagnostics (per-radius QL solve + AD smoke test) ...")
        diagnostics = run_diagnostics(gyaradax_config)
        with open(os.path.join(exp.out_dir, "diagnostics.json"), "w") as fh:
            json.dump(diagnostics, fh, indent=2)
        log.info("%s", json.dumps(diagnostics, indent=2))

    log.info("artifacts written to %s", exp.out_dir)


if __name__ == "__main__":
    main()
