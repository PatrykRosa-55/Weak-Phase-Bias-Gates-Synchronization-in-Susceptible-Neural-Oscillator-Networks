from __future__ import annotations

# =============================================================================
# 1. IMPORTS
# =============================================================================

import argparse
import concurrent.futures as cf
import hashlib
import math
import multiprocessing as mp
import os
import platform as platform_mod
import queue
import sys
import threading
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
import numpy as np
import pandas as pd
try:
    import scipy
    from scipy.stats import t as student_t
except Exception as exc:
    raise RuntimeError(
        "SciPy is required for exact t-based confidence intervals "
        "and for submission-ready reproducibility."
    ) from exc


# =============================================================================
# 2. MODEL CONSTANTS AND OUTPUT DEFINITIONS
# =============================================================================

SCRIPT_VERSION = "3.1.0-methods3-metrics-aligned"

TEMPLATE_FREQUENCIES_HZ = (7.83, 10.00, 14.30, 20.80, 27.30, 33.80)
CONDITIONS = ("sham", "transient_only", "driver_only", "transient_plus_driver")
K_VALUES = (0.35, 0.50, 0.65, 0.80, 0.95, 1.10, 1.25, 1.40, 1.55)
TRANSIENT_BOOST_VALUES = (0.00, 0.10, 0.20, 0.30, 0.45, 0.60)
PHASE_OFFSETS_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
DEFAULT_PHASE_SCAN_K_VALUES = (0.65, 0.80, 0.95)
DEFAULT_PHASE_SCAN_TRANSIENT_BOOSTS = (0.45,)
DEFAULT_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))
LOCK_COL = "lock_fraction_post"
STRONG_LOCK_COL = "strong_lock_fraction_post"
CSV_FILES = [
    "configuration_parameters.csv",
    "environment_manifest.csv",
    "main_results.csv",
    "main_summary.csv",
    "paired_contrasts.csv",
    "supplementary_results.csv",
    "supplementary_summary.csv",
    "trace_examples.csv",
]
FIGURE_FILES = [
    "figures/Fig8_phase_ignition_condition_summary.png",
    "figures/Fig8_phase_ignition_condition_summary.svg",
    "figures/Fig9_K_transient_ignition_map.png",
    "figures/Fig9_K_transient_ignition_map.svg",
    "figures/Fig10_representative_phase_ignition_traces.png",
    "figures/Fig10_representative_phase_ignition_traces.svg",
    "figures/FigS5_driver_locking_plv_map.png",
    "figures/FigS5_driver_locking_plv_map.svg",
]

FIGURE_DPI = 300
PANEL_WIDTH_IN = 6.0
PANEL_HEIGHT_IN = 5.0
TITLE_FONTSIZE = 12
PANEL_TITLE_FONTSIZE = 10
AXIS_LABEL_FONTSIZE = 11
TICK_LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 8
LINE_WIDTH = 1.8
CAPSIZE = 3.0
MATCHED_COLOR = "#1f77b4"
TRANSIENT_COLOR = "#f58518"
CONTROL_COLOR = "#b8b8b8"
COMBINED_COLOR = "#54a24b"


# =============================================================================
# 3. CONFIGURATION AND DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class Config:
    n_oscillators: int = 96
    n_replicates: int = 24
    workers: int = DEFAULT_WORKERS
    seed: int = 1234
    dt: float = 0.005
    duration_s: float = 12.0
    pre_s: float = 3.0
    post_s: float = 9.0
    natural_frequency_sd_hz: float = 0.75
    phase_diffusion_cycles_per_sqrt_s: float = 0.035
    driver_gain_cycles_per_s: float = 0.12
    transient_width_s: float = 0.20
    lock_threshold: float = 0.55
    strong_lock_threshold: float = 0.70
    minimum_sustained_lock_s: float = 0.10
    trace_sample_s: float = 0.025
    use_all_frequencies: bool = True
    main_phase_offset_deg: float = 0.0
    k_values: tuple[float, ...] = K_VALUES
    transient_boosts: tuple[float, ...] = TRANSIENT_BOOST_VALUES
    phase_offsets_deg: tuple[float, ...] = PHASE_OFFSETS_DEG
    phase_scan_k_values: tuple[float, ...] = DEFAULT_PHASE_SCAN_K_VALUES
    phase_scan_transient_boosts: tuple[float, ...] = DEFAULT_PHASE_SCAN_TRANSIENT_BOOSTS


@dataclass(frozen=True)
class Network:
    name: str
    frequency_hz: float


@dataclass(frozen=True)
class Trial:
    theta0: np.ndarray
    frequency_z: np.ndarray
    dW: np.ndarray


NETWORKS = tuple(Network(f"network_{f:.2f}Hz".replace(".", "_"), float(f)) for f in TEMPLATE_FREQUENCIES_HZ)


# =============================================================================
# 4. REPRODUCIBILITY, TIME GRID, AND VALIDATION
# =============================================================================

def stable_seed(*parts: object, base: int = 0) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return (int.from_bytes(digest, "little") + int(base)) % (2**32 - 1)


def n_steps_for(config: Config) -> int:
    return int(round(config.duration_s / config.dt))


def time_grid(config: Config) -> np.ndarray:
    return np.arange(n_steps_for(config), dtype=float) * config.dt


def is_integer_ratio(value: float, step: float) -> bool:
    ratio = value / step
    return math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9)


def validate_config(config: Config) -> None:
    if config.n_oscillators < 8:
        raise ValueError("n_oscillators must be >= 8")
    if config.n_replicates < 2:
        raise ValueError("n_replicates must be >= 2")
    if config.workers < 1:
        raise ValueError("workers must be >= 1")
    if config.dt <= 0 or config.duration_s <= 0:
        raise ValueError("dt and duration_s must be > 0")
    if not is_integer_ratio(config.duration_s, config.dt):
        raise ValueError("duration_s / dt must be an integer within numerical tolerance")
    if config.pre_s <= 0 or config.post_s <= 0:
        raise ValueError("pre_s and post_s must be > 0")
    if config.pre_s + config.post_s > config.duration_s + 1e-12:
        raise ValueError("pre_s + post_s must be <= duration_s")
    if not is_integer_ratio(config.pre_s, config.dt):
        raise ValueError("pre_s / dt must be an integer within numerical tolerance")
    if not is_integer_ratio(config.post_s, config.dt):
        raise ValueError("post_s / dt must be an integer within numerical tolerance")
    if config.natural_frequency_sd_hz <= 0:
        raise ValueError("natural_frequency_sd_hz must be > 0")
    if config.phase_diffusion_cycles_per_sqrt_s < 0:
        raise ValueError("phase_diffusion_cycles_per_sqrt_s must be >= 0")
    if config.driver_gain_cycles_per_s < 0:
        raise ValueError("driver_gain_cycles_per_s must be >= 0")
    if config.transient_width_s <= 0:
        raise ValueError("transient_width_s must be > 0")
    if not 0 <= config.lock_threshold <= 1:
        raise ValueError("lock_threshold must be in [0, 1]")
    if not 0 <= config.strong_lock_threshold <= 1:
        raise ValueError("strong_lock_threshold must be in [0, 1]")
    if config.strong_lock_threshold < config.lock_threshold:
        raise ValueError("strong_lock_threshold must be >= lock_threshold")
    if config.minimum_sustained_lock_s <= 0:
        raise ValueError("minimum_sustained_lock_s must be > 0")
    for label, values in {
        "k_values": config.k_values,
        "transient_boosts": config.transient_boosts,
        "phase_offsets_deg": config.phase_offsets_deg,
        "phase_scan_k_values": config.phase_scan_k_values,
        "phase_scan_transient_boosts": config.phase_scan_transient_boosts,
    }.items():
        if not values:
            raise ValueError(f"{label} cannot be empty")
    if any(k < 0 for k in config.k_values + config.phase_scan_k_values):
        raise ValueError("K values must be non-negative")
    if any(b < 0 for b in config.transient_boosts + config.phase_scan_transient_boosts):
        raise ValueError("transient boosts must be non-negative")


def default_output_dir() -> Path:
    return (Path(__file__).resolve().parent / "psmh_phase_ignition_outputs").resolve()


def make_output_dir(path: Path | None) -> Path:
    if path is None:
        path = default_output_dir()
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_progress_callback(progress: Callable[[str], None] | None) -> Callable[[str], None] | None:
    if progress is None:
        return None

    def wrapped(message: str) -> None:
        try:
            progress(message)
        except (OSError, ValueError):
            pass

    return wrapped


def console_progress(message: str) -> None:
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        pass


def select_networks(config: Config) -> tuple[Network, ...]:
    if config.use_all_frequencies:
        return NETWORKS
    return (next(n for n in NETWORKS if math.isclose(n.frequency_hz, 10.0)),)


# =============================================================================
# 5. STOCHASTIC TRIAL GENERATION
# =============================================================================

def make_trial(config: Config, network: Network, replicate: int) -> Trial:
    steps = n_steps_for(config)
    rng = np.random.default_rng(stable_seed(
        "trial", network.name, replicate, config.dt, config.duration_s,
        config.n_oscillators, config.natural_frequency_sd_hz, base=config.seed,
    ))
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=config.n_oscillators)
    frequency_z = rng.standard_normal(config.n_oscillators)
    dW = rng.normal(0.0, math.sqrt(config.dt), size=(steps, config.n_oscillators))
    return Trial(theta0=theta0, frequency_z=frequency_z, dW=dW)


# =============================================================================
# 6. PHASE METRICS AND TRANSIENT PROFILES
# =============================================================================

def order_parameters(theta: np.ndarray) -> tuple[complex, float, float, float]:
    z1 = np.mean(np.exp(1j * theta))
    r1 = float(abs(z1))
    r2 = float(abs(np.mean(np.exp(2j * theta))))
    r4 = float(abs(np.mean(np.exp(4j * theta))))
    return complex(z1), r1, r2, r4


def gaussian_pulse(t: np.ndarray, center_s: float, width_s: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - center_s) / width_s) ** 2)


def first_sustained_true_time(mask: np.ndarray, dt: float, minimum_duration_s: float) -> float:
    required = max(1, int(math.ceil(minimum_duration_s / dt)))
    run = 0
    for idx, value in enumerate(np.asarray(mask, dtype=bool)):
        if value:
            run += 1
            if run >= required:
                return float((idx - required + 1) * dt)
        else:
            run = 0
    return float("nan")


def longest_true_run(mask: np.ndarray) -> int:
    best = run = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def count_lock_episodes(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return 0
    starts = mask & np.concatenate([[True], ~mask[:-1]])
    return int(np.sum(starts))


def circular_lock_strength(z_trace: np.ndarray, phase: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    values = z_trace * np.exp(-1j * phase)
    if mask is not None:
        values = values[mask]
    values = values[np.isfinite(values.real) & np.isfinite(values.imag)]
    if len(values) == 0:
        return float("nan"), float("nan")
    z = np.mean(values)
    return float(abs(z)), float(np.angle(z))


# =============================================================================
# 7. KURAMOTO PHASE-IGNITION SIMULATION
# =============================================================================

def simulate_condition(
    config: Config,
    network: Network,
    trial: Trial,
    replicate: int,
    k_base: float,
    transient_boost: float,
    phase_offset_deg: float,
    condition: str,
    return_trace: bool = False,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    steps = n_steps_for(config)
    t = time_grid(config)
    theta = trial.theta0.copy()
    omega = 2.0 * np.pi * (network.frequency_hz + config.natural_frequency_sd_hz * trial.frequency_z)
    noise_rad = 2.0 * np.pi * config.phase_diffusion_cycles_per_sqrt_s
    driver_phase = 2.0 * np.pi * network.frequency_hz * t

    phase_offset_s = (float(phase_offset_deg) / 360.0) / network.frequency_hz
    transient_center_s = config.pre_s + 0.5 + phase_offset_s
    transient_active = condition in {"transient_only", "transient_plus_driver"}
    driver_active = condition in {"driver_only", "transient_plus_driver"}
    transient_amp = float(transient_boost) if transient_active else 0.0
    driver_gain = float(config.driver_gain_cycles_per_s) if driver_active else 0.0
    transient_profile = gaussian_pulse(t, transient_center_s, config.transient_width_s) if transient_active else np.zeros_like(t)

    r1 = np.zeros(steps)
    r2 = np.zeros(steps)
    r4 = np.zeros(steps)
    psi = np.zeros(steps)
    pci = np.zeros(steps)
    z_trace = np.zeros(steps, dtype=np.complex128)

    for step in range(steps):
        z1, rr1, rr2, rr4 = order_parameters(theta)
        z_trace[step] = z1
        r1[step], r2[step], r4[step], psi[step] = rr1, rr2, rr4, float(np.angle(z1))
        k_eff = float(k_base) + transient_amp * transient_profile[step]
        coupling = 2.0 * np.pi * k_eff * rr1 * np.sin(psi[step] - theta)
        driver = 0.0
        if driver_active and driver_gain > 0.0:
            driver = 2.0 * np.pi * driver_gain * np.sin(driver_phase[step] - theta)
        theta = (theta + (omega + coupling + driver) * config.dt + noise_rad * trial.dW[step]) % (2.0 * np.pi)
        pci[step] = rr1 * transient_profile[step]

    post_end_s = min(config.duration_s, config.pre_s + config.post_s)
    post_mask = (t >= config.pre_s) & (t < post_end_s)
    transient_mask = transient_profile > math.exp(-0.5 * 3.0**2) if transient_active else np.zeros(steps, dtype=bool)
    lock_mask = r1 >= config.lock_threshold
    strong_lock_mask = r1 >= config.strong_lock_threshold
    post_lock = lock_mask[post_mask]
    post_strong = strong_lock_mask[post_mask]
    post_t0 = float(t[np.where(post_mask)[0][0]]) if np.any(post_mask) else float("nan")
    t_sustained_post = first_sustained_true_time(post_lock, config.dt, config.minimum_sustained_lock_s)
    t_sustained_abs = post_t0 + t_sustained_post if np.isfinite(t_sustained_post) else float("nan")
    driver_lock_strength, driver_lock_lag = circular_lock_strength(z_trace, driver_phase, post_mask) if driver_active else (float("nan"), float("nan"))

    row: dict[str, object] = {
        "condition": condition,
        "network": network.name,
        "network_frequency_hz": network.frequency_hz,
        "replicate": int(replicate),
        "k_base_cycles_per_s": float(k_base),
        "transient_boost_cycles_per_s": float(transient_boost),
        "phase_offset_deg": float(phase_offset_deg),
        "driver_gain_cycles_per_s": float(config.driver_gain_cycles_per_s),
        "transient_center_s": float(transient_center_s),
        "mean_R1_post": float(np.mean(r1[post_mask])),
        "peak_R1_post": float(np.max(r1[post_mask])),
        "mean_R2_post": float(np.mean(r2[post_mask])),
        "mean_R4_post": float(np.mean(r4[post_mask])),
        LOCK_COL: float(np.mean(post_lock)),
        STRONG_LOCK_COL: float(np.mean(post_strong)),
        "time_to_sustained_lock_s": t_sustained_post,
        "time_to_sustained_lock_post_s": t_sustained_post,
        "time_of_first_sustained_lock_s": t_sustained_abs,
        "longest_lock_run_s": float(longest_true_run(post_lock) * config.dt),
        "lock_episode_count_post": count_lock_episodes(post_lock),
        "driver_locking_plv_post": driver_lock_strength,
        "driver_lock_strength_post": driver_lock_strength,
        "driver_lock_lag_rad": driver_lock_lag,
        "peak_PCI": float(np.max(pci)),
        "mean_PCI_transient_window": float(np.mean(pci[transient_mask])) if np.any(transient_mask) else 0.0,
        "peak_R1_transient_window": float(np.max(r1[transient_mask])) if np.any(transient_mask) else float("nan"),
    }

    trace = None
    if return_trace:
        every = max(1, int(round(config.trace_sample_s / config.dt)))
        idx = np.arange(0, steps, every)
        trace = pd.DataFrame({
            "time_s": t[idx],
            "condition": condition,
            "network": network.name,
            "replicate": replicate,
            "k_base_cycles_per_s": float(k_base),
            "transient_boost_cycles_per_s": float(transient_boost),
            "phase_offset_deg": float(phase_offset_deg),
            "R1": r1[idx],
            "R2": r2[idx],
            "R4": r4[idx],
            "PCI": pci[idx],
            "transient_profile": transient_profile[idx],
            "driver_phase_rad": driver_phase[idx],
            "population_phase_rad": psi[idx],
        })
    return row, trace


# =============================================================================
# 8. BLOCK WORKERS AND PARALLEL EXECUTION
# =============================================================================

def main_block_worker(job: tuple[Config, str, int]) -> list[dict[str, object]]:
    config, network_name, replicate = job
    network = next(n for n in NETWORKS if n.name == network_name)
    trial = make_trial(config, network, replicate)
    rows: list[dict[str, object]] = []
    for k in config.k_values:
        for boost in config.transient_boosts:
            for condition in CONDITIONS:
                row, _ = simulate_condition(config, network, trial, replicate, k, boost, config.main_phase_offset_deg, condition)
                row["analysis"] = "main_k_transient_scan"
                rows.append(row)
    return rows


def phase_scan_block_worker(job: tuple[Config, str, int]) -> list[dict[str, object]]:
    config, network_name, replicate = job
    network = next(n for n in NETWORKS if n.name == network_name)
    trial = make_trial(config, network, replicate)
    rows: list[dict[str, object]] = []
    for k in config.phase_scan_k_values:
        for boost in config.phase_scan_transient_boosts:
            for phase_offset in config.phase_offsets_deg:
                for condition in CONDITIONS:
                    row, _ = simulate_condition(config, network, trial, replicate, k, boost, phase_offset, condition)
                    row["analysis"] = "phase_offset_sensitivity"
                    rows.append(row)
    return rows


def run_blocks(
    config: Config,
    jobs: list[tuple[Config, str, int]],
    worker_func: Callable[[tuple[Config, str, int]], list[dict[str, object]]],
    progress: Callable[[str], None] | None = None,
    executor: cf.Executor | None = None,
) -> pd.DataFrame:
    if not jobs:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    if config.workers == 1 or executor is None:
        for i, job in enumerate(jobs, 1):
            rows.extend(worker_func(job))
            if progress and (i == len(jobs) or i % max(1, len(jobs) // 10) == 0):
                progress(f"Completed {i}/{len(jobs)} blocks")
    else:
        futures = [executor.submit(worker_func, job) for job in jobs]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            rows.extend(fut.result())
            if progress and (i == len(jobs) or i % max(1, len(jobs) // 10) == 0):
                progress(f"Completed {i}/{len(jobs)} blocks")
    return pd.DataFrame(rows)


# =============================================================================
# 9. PAIRED INTERACTION TERMS AND SUMMARIES
# =============================================================================

def ci_stats(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(len(arr))
    mean = float(np.mean(arr)) if n else float("nan")
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = float(sd / math.sqrt(n)) if n > 1 else 0.0
    if n > 1 and student_t is not None:
        ci = float(student_t.ppf(0.975, df=n - 1) * sem)
    elif n > 1:
        ci = float(1.96 * sem)
    else:
        ci = 0.0
    return {"mean": mean, "sd": sd, "sem": sem, "ci95_low": mean - ci, "ci95_high": mean + ci, "ci95_halfwidth": ci, "n_blocks": n}


def paired_interactions(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_R1_post",
        "peak_R1_post",
        "mean_R2_post",
        "mean_R4_post",
        LOCK_COL,
        STRONG_LOCK_COL,
        "time_to_sustained_lock_s",
        "time_to_sustained_lock_post_s",
        "time_of_first_sustained_lock_s",
        "longest_lock_run_s",
        "lock_episode_count_post",
        "driver_locking_plv_post",
        "peak_PCI",
        "mean_PCI_transient_window",
        "peak_R1_transient_window",
    ]
    id_cols = [
        "analysis", "network", "network_frequency_hz", "replicate", "driver_gain_cycles_per_s",
        "k_base_cycles_per_s", "transient_boost_cycles_per_s", "phase_offset_deg",
    ]
    rows: list[dict[str, object]] = []
    for keys, part in raw.groupby(id_cols, dropna=False):
        data = {cond: part[part["condition"] == cond].iloc[0] for cond in CONDITIONS if not part[part["condition"] == cond].empty}
        if set(data) != set(CONDITIONS):
            continue
        row = dict(zip(id_cols, keys if isinstance(keys, tuple) else (keys,)))
        for metric in metrics:
            sham = float(data["sham"].get(metric, float("nan")))
            transient = float(data["transient_only"].get(metric, float("nan")))
            driver = float(data["driver_only"].get(metric, float("nan")))
            both = float(data["transient_plus_driver"].get(metric, float("nan")))
            row[f"sham_{metric}"] = sham
            row[f"transient_only_{metric}"] = transient
            row[f"driver_only_{metric}"] = driver
            row[f"transient_plus_driver_{metric}"] = both
            if all(np.isfinite(v) for v in (sham, transient, driver, both)):
                row[f"synergy_{metric}"] = (both - sham) - (transient - sham) - (driver - sham)
                row[f"both_minus_best_single_{metric}"] = both - max(sham, transient, driver)
            else:
                row[f"synergy_{metric}"] = float("nan")
                row[f"both_minus_best_single_{metric}"] = float("nan")
            row[f"transient_increment_{metric}"] = transient - sham if np.isfinite(transient) and np.isfinite(sham) else float("nan")
            row[f"driver_increment_{metric}"] = driver - sham if np.isfinite(driver) and np.isfinite(sham) else float("nan")
            row[f"both_increment_{metric}"] = both - sham if np.isfinite(both) and np.isfinite(sham) else float("nan")
            row[f"both_minus_driver_only_{metric}"] = both - driver if np.isfinite(both) and np.isfinite(driver) else float("nan")
        row["driver_only_any_lock"] = bool(float(data["driver_only"].get(LOCK_COL, 0.0)) > 0.0)
        row["transient_only_any_lock"] = bool(float(data["transient_only"].get(LOCK_COL, 0.0)) > 0.0)
        row["both_any_lock"] = bool(float(data["transient_plus_driver"].get(LOCK_COL, 0.0)) > 0.0)
        row["strict_subthreshold_ignition_candidate"] = bool(
            float(data["driver_only"].get(LOCK_COL, 0.0)) == 0.0
            and float(data["transient_only"].get(LOCK_COL, 0.0)) == 0.0
            and float(data["transient_plus_driver"].get(LOCK_COL, 0.0)) > 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_interactions(interactions: pd.DataFrame, group_cols: list[str], by_frequency: bool = False) -> pd.DataFrame:
    if interactions.empty:
        return pd.DataFrame()
    numeric_cols = []
    for col in interactions.columns:
        if col in set(group_cols + ["network", "network_frequency_hz", "replicate", "analysis"]):
            continue
        if col.endswith("_deg") or col.endswith("_cycles_per_s"):
            continue
        if pd.api.types.is_bool_dtype(interactions[col]) or pd.api.types.is_numeric_dtype(interactions[col]):
            numeric_cols.append(col)
    if by_frequency:
        rep_group_cols = ["network", "network_frequency_hz"] + group_cols + ["replicate"]
    else:
        rep_group_cols = group_cols + ["replicate"]
    rep_rows: list[dict[str, object]] = []
    for keys, part in interactions.groupby(rep_group_cols, dropna=False):
        row = dict(zip(rep_group_cols, keys if isinstance(keys, tuple) else (keys,)))
        for col in numeric_cols:
            vals = part[col].astype(float).to_numpy()
            row[col] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        rep_rows.append(row)
    rep_level = pd.DataFrame(rep_rows)
    summary_group_cols = (["network", "network_frequency_hz"] if by_frequency else []) + group_cols
    rows: list[dict[str, object]] = []
    for keys, part in rep_level.groupby(summary_group_cols, dropna=False):
        row = dict(zip(summary_group_cols, keys if isinstance(keys, tuple) else (keys,)))
        for col in numeric_cols:
            for name, value in ci_stats(part[col]).items():
                row[f"{col}_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(summary_group_cols).reset_index(drop=True) if rows else pd.DataFrame()


def make_paired_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    keep = [
        "k_base_cycles_per_s", "transient_boost_cycles_per_s",
        "driver_increment_mean_R1_post_mean",
        "transient_increment_mean_R1_post_mean",
        "both_increment_mean_R1_post_mean",
        "both_minus_best_single_mean_R1_post_mean",
        "both_minus_best_single_mean_R2_post_mean",
        "both_minus_best_single_mean_R4_post_mean",
        "both_minus_best_single_lock_fraction_post_mean",
        "both_minus_best_single_time_to_sustained_lock_post_s_mean",
        "synergy_mean_R1_post_mean",
        "synergy_mean_R2_post_mean",
        "synergy_mean_R4_post_mean",
        "synergy_lock_fraction_post_mean",
        "synergy_time_to_sustained_lock_post_s_mean",
        "driver_only_driver_locking_plv_post_mean",
        "transient_plus_driver_driver_locking_plv_post_mean",
        "both_minus_driver_only_driver_locking_plv_post_mean",
        "strict_subthreshold_ignition_candidate_mean",
        "n_blocks",
    ]
    cols = [c for c in keep if c in summary.columns]
    return summary[cols].copy()


# =============================================================================
# 10. REPRESENTATIVE TRACES
# =============================================================================

def make_representative_traces(config: Config, networks: tuple[Network, ...], main_summary: pd.DataFrame) -> pd.DataFrame:
    if main_summary.empty:
        k = min(config.k_values, key=lambda x: abs(x - 0.80))
        boost = config.transient_boosts[-1]
    elif "strict_subthreshold_ignition_candidate_mean" in main_summary and np.isfinite(main_summary["strict_subthreshold_ignition_candidate_mean"]).any():
        candidates = main_summary.sort_values(
            ["strict_subthreshold_ignition_candidate_mean", "both_minus_best_single_lock_fraction_post_mean"],
            ascending=False,
        )
        best = candidates.iloc[0]
        if float(best.get("strict_subthreshold_ignition_candidate_mean", 0.0)) <= 0.0:
            best = main_summary.sort_values("both_minus_best_single_lock_fraction_post_mean", ascending=False).iloc[0]
        k = float(best["k_base_cycles_per_s"])
        boost = float(best["transient_boost_cycles_per_s"])
    else:
        best = main_summary.sort_values("both_minus_best_single_lock_fraction_post_mean", ascending=False).iloc[0]
        k = float(best["k_base_cycles_per_s"])
        boost = float(best["transient_boost_cycles_per_s"])
    network = next((n for n in networks if math.isclose(n.frequency_hz, 10.0)), networks[0])
    trial = make_trial(config, network, 0)
    frames: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        _, trace = simulate_condition(config, network, trial, 0, k, boost, config.main_phase_offset_deg, condition, return_trace=True)
        if trace is not None:
            frames.append(trace)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =============================================================================
# 11. OUTPUT TABLES AND FIGURES
# =============================================================================

def config_table(config: Config) -> pd.DataFrame:
    rows = []
    for key, value in asdict(config).items():
        if isinstance(value, tuple):
            value = ",".join(str(x) for x in value)
        rows.append({"parameter": key, "value": value})
    return pd.DataFrame(rows)


def write_environment_manifest(config: Config, output: Path) -> None:
    code_bytes = Path(__file__).read_bytes()
    pd.DataFrame([{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "script_name": Path(__file__).name,
        "python_version": sys.version.replace("\n", " "),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform_mod.platform(),
        "cpu_count": os.cpu_count(),
        "workers": config.workers,
        "code_sha256": hashlib.sha256(code_bytes).hexdigest(),
    }]).to_csv(output / "environment_manifest.csv", index=False)


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)


def heatmap_from_summary(summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return summary.pivot_table(index="k_base_cycles_per_s", columns="transient_boost_cycles_per_s", values=value_col).sort_index(ascending=True)


def plot_heatmap(ax, pivot: pd.DataFrame, title: str, colorbar_label: str):
    im = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    ax.set_xlabel("Transient coupling boost (cycles/s)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Baseline coupling, K (cycles/s)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, loc="left", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=8)
    return im


def make_figures(output: Path, main_summary: pd.DataFrame, phase_summary: pd.DataFrame, traces: pd.DataFrame) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not main_summary.empty:
        if "strict_subthreshold_ignition_candidate_mean" in main_summary:
            ranked = main_summary.sort_values(
                ["strict_subthreshold_ignition_candidate_mean", "both_minus_best_single_lock_fraction_post_mean"],
                ascending=False,
            )
            best = ranked.iloc[0]
            if float(best.get("strict_subthreshold_ignition_candidate_mean", 0.0)) <= 0.0:
                best = main_summary.sort_values("both_minus_best_single_lock_fraction_post_mean", ascending=False).iloc[0]
        else:
            best = main_summary.sort_values("both_minus_best_single_lock_fraction_post_mean", ascending=False).iloc[0]

        conditions = ["sham", "transient_only", "driver_only", "transient_plus_driver"]
        labels = ["sham", "transient", "driver", "transient + driver"]
        colors = [CONTROL_COLOR, TRANSIENT_COLOR, MATCHED_COLOR, COMBINED_COLOR]

        fig, axes = plt.subplots(1, 2, figsize=(PANEL_WIDTH_IN * 2, PANEL_HEIGHT_IN), constrained_layout=True)
        ax = axes[0]
        values = [best.get(f"{c}_mean_R1_post_mean", np.nan) for c in conditions]
        errors = [best.get(f"{c}_mean_R1_post_ci95_halfwidth", 0.0) for c in conditions]
        x = np.arange(len(conditions))
        ax.bar(x, values, color=colors, edgecolor="none")
        ax.errorbar(x, values, yerr=errors, fmt="none", color="black", capsize=CAPSIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(r"Post-event mean synchrony, $R_1$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title("A. Population synchrony", loc="left", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
        style_axes(ax)

        ax = axes[1]
        plv_conditions = ["driver_only", "transient_plus_driver"]
        plv_labels = ["driver", "transient + driver"]
        plv_values = [best.get(f"{c}_driver_locking_plv_post_mean", np.nan) for c in plv_conditions]
        plv_errors = [best.get(f"{c}_driver_locking_plv_post_ci95_halfwidth", 0.0) for c in plv_conditions]
        x2 = np.arange(len(plv_conditions))
        ax.bar(x2, plv_values, color=[MATCHED_COLOR, COMBINED_COLOR], edgecolor="none")
        ax.errorbar(x2, plv_values, yerr=plv_errors, fmt="none", color="black", capsize=CAPSIZE)
        ax.set_xticks(x2)
        ax.set_xticklabels(plv_labels, rotation=25, ha="right")
        ax.set_ylim(0.0, max(1.0, np.nanmax(plv_values) * 1.15 if np.isfinite(plv_values).any() else 1.0))
        ax.set_ylabel("Pooled driver-locking PLV", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title("B. Oscillator-driver phase locking", loc="left", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
        style_axes(ax)
        fig.suptitle("State-gated phase ignition", fontsize=TITLE_FONTSIZE, fontweight="bold")
        fig.savefig(figure_dir / "Fig8_phase_ignition_condition_summary.png", dpi=FIGURE_DPI)
        fig.savefig(figure_dir / "Fig8_phase_ignition_condition_summary.svg")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(PANEL_WIDTH_IN * 2, PANEL_HEIGHT_IN), constrained_layout=True)
        h1 = heatmap_from_summary(main_summary, "both_minus_best_single_lock_fraction_post_mean")
        h2 = heatmap_from_summary(main_summary, "strict_subthreshold_ignition_candidate_mean")
        im1 = plot_heatmap(axes[0], h1, "A. Gain beyond best single condition", "Lock-fraction gain")
        im2 = plot_heatmap(axes[1], h2, "B. Strict subthreshold ignition candidates", "Candidate fraction")
        fig.colorbar(im1, ax=axes[0], label="Lock-fraction gain")
        fig.colorbar(im2, ax=axes[1], label="Candidate fraction")
        fig.suptitle("Coupling- and transient-dependent ignition window", fontsize=TITLE_FONTSIZE, fontweight="bold")
        fig.savefig(figure_dir / "Fig9_K_transient_ignition_map.png", dpi=FIGURE_DPI)
        fig.savefig(figure_dir / "Fig9_K_transient_ignition_map.svg")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(PANEL_WIDTH_IN * 2, PANEL_HEIGHT_IN), constrained_layout=True)
        h3 = heatmap_from_summary(main_summary, "driver_only_driver_locking_plv_post_mean")
        h4 = heatmap_from_summary(main_summary, "transient_plus_driver_driver_locking_plv_post_mean")
        im3 = plot_heatmap(axes[0], h3, "A. Driver only", "Pooled driver-locking PLV")
        im4 = plot_heatmap(axes[1], h4, "B. Transient + driver", "Pooled driver-locking PLV")
        fig.colorbar(im3, ax=axes[0], label="Pooled driver-locking PLV")
        fig.colorbar(im4, ax=axes[1], label="Pooled driver-locking PLV")
        fig.suptitle("Oscillator-driver phase locking across ignition regimes", fontsize=TITLE_FONTSIZE, fontweight="bold")
        fig.savefig(figure_dir / "FigS5_driver_locking_plv_map.png", dpi=FIGURE_DPI)
        fig.savefig(figure_dir / "FigS5_driver_locking_plv_map.svg")
        plt.close(fig)

    if not traces.empty:
        fig, axes_obj = plt.subplots(
            2,
            2,
            figsize=(PANEL_WIDTH_IN * 2, PANEL_HEIGHT_IN * 2),
            sharex=True,
            constrained_layout=True,
        )
        axes = list(np.ravel(axes_obj))
        panel_labels = {
            "sham": "A. Sham",
            "transient_only": "B. Transient only",
            "driver_only": "C. Driver only",
            "transient_plus_driver": "D. Transient + driver",
        }
        for ax, condition in zip(axes, CONDITIONS):
            part = traces[traces["condition"] == condition]
            if not part.empty:
                ax.plot(part["time_s"], part["R1"], label=r"$R_1$", linewidth=LINE_WIDTH)
                ax.plot(part["time_s"], part["R2"], label=r"$R_2$", linewidth=LINE_WIDTH, alpha=0.75)
                ax.plot(part["time_s"], part["transient_profile"], label="transient", linewidth=LINE_WIDTH, alpha=0.5)
            ax.set_title(panel_labels.get(condition, condition.replace("_", " ")), loc="left", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
            ax.set_ylabel("Amplitude / synchrony", fontsize=AXIS_LABEL_FONTSIZE)
            ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right", frameon=False)
            style_axes(ax)
        for ax in axes[-2:]:
            ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
        fig.suptitle("Representative state-gated phase-ignition traces", fontsize=TITLE_FONTSIZE, fontweight="bold")
        fig.savefig(figure_dir / "Fig10_representative_phase_ignition_traces.png", dpi=FIGURE_DPI)
        fig.savefig(figure_dir / "Fig10_representative_phase_ignition_traces.svg")
        plt.close(fig)


# =============================================================================
# 12. ANALYSIS PIPELINE
# =============================================================================

def run_analysis(config: Config, output: Path, progress: Callable[[str], None] | None = None) -> dict[str, pd.DataFrame]:
    validate_config(config)
    output = make_output_dir(output)
    progress = safe_progress_callback(progress)
    crash_log = output / "crash_log.txt"
    if crash_log.exists():
        try:
            crash_log.unlink()
        except OSError:
            pass
    networks = select_networks(config)
    jobs = [(config, network.name, rep) for network in networks for rep in range(config.n_replicates)]
    try:
        if progress:
            progress("Running main K × transient scan")
        if config.workers > 1:
            with cf.ProcessPoolExecutor(max_workers=config.workers, mp_context=mp.get_context("spawn")) as executor:
                main_raw = run_blocks(config, jobs, main_block_worker, progress, executor)
                if progress:
                    progress("Running supplementary phase-offset scan")
                phase_raw = run_blocks(config, jobs, phase_scan_block_worker, progress, executor)
        else:
            main_raw = run_blocks(config, jobs, main_block_worker, progress, None)
            if progress:
                progress("Running supplementary phase-offset scan")
            phase_raw = run_blocks(config, jobs, phase_scan_block_worker, progress, None)

        if progress:
            progress("Computing paired interaction terms")
        main_interactions = paired_interactions(main_raw)
        phase_interactions = paired_interactions(phase_raw)
        main_group_cols = ["k_base_cycles_per_s", "transient_boost_cycles_per_s"]
        phase_group_cols = ["k_base_cycles_per_s", "transient_boost_cycles_per_s", "phase_offset_deg"]
        main_summary = summarize_interactions(main_interactions, main_group_cols, by_frequency=False)
        phase_summary = summarize_interactions(phase_interactions, phase_group_cols, by_frequency=False)
        paired_contrasts = make_paired_contrasts(main_summary)
        traces = make_representative_traces(config, networks, main_summary)

        config_table(config).to_csv(output / "configuration_parameters.csv", index=False)
        write_environment_manifest(config, output)
        main_interactions.to_csv(output / "main_results.csv", index=False)
        main_summary.to_csv(output / "main_summary.csv", index=False)
        paired_contrasts.to_csv(output / "paired_contrasts.csv", index=False)
        phase_interactions.to_csv(output / "supplementary_results.csv", index=False)
        phase_summary.to_csv(output / "supplementary_summary.csv", index=False)
        traces.to_csv(output / "trace_examples.csv", index=False)
        if progress:
            progress("Writing figures")
        make_figures(output, main_summary, phase_summary, traces)
        missing = [name for name in CSV_FILES + FIGURE_FILES if not (output / name).exists()]
        if missing:
            raise RuntimeError(f"Missing expected output files: {missing}")
        if progress:
            progress("Done")
        return {
            "main_raw": main_raw,
            "phase_raw": phase_raw,
            "main_results": main_interactions,
            "main_summary": main_summary,
            "paired_contrasts": paired_contrasts,
            "supplementary_results": phase_interactions,
            "supplementary_summary": phase_summary,
            "traces": traces,
        }
    except Exception:
        crash_log.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def quick_config(config: Config) -> Config:
    return replace(
        config,
        n_oscillators=min(config.n_oscillators, 24),
        n_replicates=2,
        workers=1,
        duration_s=1.20,
        pre_s=0.30,
        post_s=0.90,
        transient_width_s=0.05,
        trace_sample_s=0.025,
        use_all_frequencies=False,
        k_values=(0.65, 0.95),
        transient_boosts=(0.00, 0.45),
        phase_offsets_deg=(0.0, 180.0),
        phase_scan_k_values=(0.65,),
        phase_scan_transient_boosts=(0.45,),
    )


def remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            Path(root, name).unlink()
        for name in dirs:
            Path(root, name).rmdir()
    path.rmdir()


def run_self_checks() -> None:
    cfg = quick_config(Config())
    validate_config(cfg)
    phases = np.array([0.0] * 10 + [np.pi] * 10)
    _, r1, r2, _ = order_parameters(phases)
    assert r1 < 1e-12 and abs(r2 - 1.0) < 1e-12

    theta_samples = np.array([
        [0.1, 0.2, 0.4],
        [0.3, 0.4, 0.6],
        [0.5, 0.6, 0.8],
    ])
    driver_phase = np.array([0.0, 0.2, 0.4])
    z_trace = np.mean(np.exp(1j * theta_samples), axis=1)
    plv_via_z, _ = circular_lock_strength(z_trace, driver_phase)
    plv_direct = abs(np.mean(np.exp(1j * (theta_samples - driver_phase[:, None]))))
    assert abs(plv_via_z - plv_direct) < 1e-12

    out = Path("_self_check_psmh_phase_ignition_outputs")
    remove_tree(out)
    run_analysis(cfg, out, progress=None)
    missing = [name for name in CSV_FILES + FIGURE_FILES if not (out / name).exists() or (out / name).stat().st_size == 0]
    assert not missing, f"Missing expected self-check outputs: {missing}"
    main_results = pd.read_csv(out / "main_results.csv")
    main_summary = pd.read_csv(out / "main_summary.csv")
    required_main_results = [
        "sham_mean_R2_post",
        "transient_only_mean_R4_post",
        "driver_only_time_to_sustained_lock_post_s",
        "transient_plus_driver_time_to_sustained_lock_post_s",
    ]
    required_main_summary = [
        "driver_only_mean_R2_post_mean",
        "transient_plus_driver_mean_R4_post_mean",
        "both_increment_time_to_sustained_lock_post_s_mean",
    ]
    for col in required_main_results:
        assert col in main_results.columns, f"Missing column in main_results.csv: {col}"
    for col in required_main_summary:
        assert col in main_summary.columns, f"Missing column in main_summary.csv: {col}"
    mask = np.array([True] * int(math.ceil(cfg.minimum_sustained_lock_s / cfg.dt)) + [False] * 10)
    assert first_sustained_true_time(mask, cfg.dt, cfg.minimum_sustained_lock_s) == 0.0


# =============================================================================
# 13. COMMAND-LINE INTERFACE
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PSMH phase-ignition Kuramoto model")
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--output", default="")
    p.add_argument("--quick", action="store_true", help="Run a small smoke-test configuration.")
    p.add_argument("--single-frequency", action="store_true", help="Use only the 10 Hz template.")
    p.add_argument("--self-check", action="store_true", help="Run deterministic internal checks and a small output test.")
    return p


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


# =============================================================================
# 14. GUI
# =============================================================================

def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    root = tk.Tk()
    root.title("PSMH phase-ignition model")

    output_var = tk.StringVar(value=str(default_output_dir()))
    tk.Label(root, text="Output folder", anchor="w").grid(row=0, column=0, sticky="w", padx=6, pady=3)
    tk.Entry(root, textvariable=output_var, width=72).grid(row=0, column=1, columnspan=3, sticky="ew", padx=6, pady=3)
    tk.Button(root, text="Choose", command=lambda: output_var.set(filedialog.askdirectory() or output_var.get())).grid(row=0, column=4, padx=6, pady=3)

    settings = ttk.LabelFrame(root, text="Paper-core settings")
    settings.grid(row=1, column=0, columnspan=5, sticky="ew", padx=6, pady=6)
    fields: dict[str, tk.StringVar] = {}
    defaults = Config()

    def tuple_text(values: tuple[float, ...]) -> str:
        return ",".join(str(v) for v in values)

    setting_rows = [
        ("Oscillators", "n_oscillators", defaults.n_oscillators),
        ("Replicates", "n_replicates", defaults.n_replicates),
        ("Workers", "workers", defaults.workers),
        ("Seed", "seed", defaults.seed),
        ("dt (s)", "dt", defaults.dt),
        ("Duration (s)", "duration_s", defaults.duration_s),
        ("Pre window (s)", "pre_s", defaults.pre_s),
        ("Post window (s)", "post_s", defaults.post_s),
        ("Frequency SD (Hz)", "natural_frequency_sd_hz", defaults.natural_frequency_sd_hz),
        ("Phase diffusion", "phase_diffusion_cycles_per_sqrt_s", defaults.phase_diffusion_cycles_per_sqrt_s),
        ("Driver gain", "driver_gain_cycles_per_s", defaults.driver_gain_cycles_per_s),
        ("Transient width (s)", "transient_width_s", defaults.transient_width_s),
        ("Lock threshold", "lock_threshold", defaults.lock_threshold),
        ("Strong-lock threshold", "strong_lock_threshold", defaults.strong_lock_threshold),
        ("Minimum lock run (s)", "minimum_sustained_lock_s", defaults.minimum_sustained_lock_s),
        ("Trace sample (s)", "trace_sample_s", defaults.trace_sample_s),
        ("Main phase offset (deg)", "main_phase_offset_deg", defaults.main_phase_offset_deg),
        ("K values", "k_values", tuple_text(defaults.k_values)),
        ("Transient boosts", "transient_boosts", tuple_text(defaults.transient_boosts)),
        ("Phase offsets", "phase_offsets_deg", tuple_text(defaults.phase_offsets_deg)),
        ("Phase-scan K values", "phase_scan_k_values", tuple_text(defaults.phase_scan_k_values)),
        ("Phase-scan boosts", "phase_scan_transient_boosts", tuple_text(defaults.phase_scan_transient_boosts)),
    ]
    for idx, (label, key, value) in enumerate(setting_rows):
        row = idx // 2
        col = (idx % 2) * 2
        tk.Label(settings, text=label, anchor="w").grid(row=row, column=col, sticky="w", padx=6, pady=2)
        var = tk.StringVar(value=str(value))
        fields[key] = var
        tk.Entry(settings, textvariable=var, width=26).grid(row=row, column=col + 1, sticky="ew", padx=6, pady=2)

    settings.columnconfigure(1, weight=1)
    settings.columnconfigure(3, weight=1)

    status_var = tk.StringVar(value="Idle")
    tk.Label(root, textvariable=status_var, anchor="w").grid(row=2, column=0, columnspan=5, sticky="ew", padx=6, pady=(6, 0))
    progress_bar = ttk.Progressbar(root, mode="indeterminate")
    progress_bar.grid(row=3, column=0, columnspan=5, sticky="ew", padx=6, pady=(2, 6))
    run_button = tk.Button(root, text="Run")
    run_button.grid(row=4, column=0, columnspan=5, sticky="ew", padx=6, pady=(0, 6))
    log = scrolledtext.ScrolledText(root, width=96, height=18)
    log.grid(row=5, column=0, columnspan=5, sticky="nsew", padx=6, pady=6)
    q: queue.Queue[str] = queue.Queue()
    running_var = tk.BooleanVar(value=False)

    def poll_log() -> None:
        while True:
            try:
                msg = q.get_nowait()
            except queue.Empty:
                break
            if msg == "__DONE__":
                running_var.set(False)
                progress_bar.stop()
                run_button.configure(state="normal")
                status_var.set("Done")
                log.insert("end", "Done\n")
                log.see("end")
                continue
            if msg.startswith("__ERROR__"):
                running_var.set(False)
                progress_bar.stop()
                run_button.configure(state="normal")
                status_var.set("Error")
                log.insert("end", msg.replace("__ERROR__", "", 1) + "\n")
                log.see("end")
                messagebox.showerror("Error", "Analysis failed; see log.")
                continue
            status_var.set(msg)
            log.insert("end", msg + "\n")
            log.see("end")
        root.after(200, poll_log)

    def run_clicked() -> None:
        if running_var.get():
            messagebox.showinfo("Running", "Analysis is already running.")
            return
        try:
            cfg = Config(
                n_oscillators=int(fields["n_oscillators"].get()),
                n_replicates=int(fields["n_replicates"].get()),
                workers=int(fields["workers"].get()),
                seed=int(fields["seed"].get()),
                dt=float(fields["dt"].get()),
                duration_s=float(fields["duration_s"].get()),
                pre_s=float(fields["pre_s"].get()),
                post_s=float(fields["post_s"].get()),
                natural_frequency_sd_hz=float(fields["natural_frequency_sd_hz"].get()),
                phase_diffusion_cycles_per_sqrt_s=float(fields["phase_diffusion_cycles_per_sqrt_s"].get()),
                driver_gain_cycles_per_s=float(fields["driver_gain_cycles_per_s"].get()),
                transient_width_s=float(fields["transient_width_s"].get()),
                lock_threshold=float(fields["lock_threshold"].get()),
                strong_lock_threshold=float(fields["strong_lock_threshold"].get()),
                minimum_sustained_lock_s=float(fields["minimum_sustained_lock_s"].get()),
                trace_sample_s=float(fields["trace_sample_s"].get()),
                use_all_frequencies=defaults.use_all_frequencies,
                main_phase_offset_deg=float(fields["main_phase_offset_deg"].get()),
                k_values=parse_float_tuple(fields["k_values"].get()),
                transient_boosts=parse_float_tuple(fields["transient_boosts"].get()),
                phase_offsets_deg=parse_float_tuple(fields["phase_offsets_deg"].get()),
                phase_scan_k_values=parse_float_tuple(fields["phase_scan_k_values"].get()),
                phase_scan_transient_boosts=parse_float_tuple(fields["phase_scan_transient_boosts"].get()),
            )
            validate_config(cfg)
            output_path = Path(output_var.get()).expanduser().resolve()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        running_var.set(True)
        run_button.configure(state="disabled")
        progress_bar.start(10)
        status_var.set("Running")
        log.insert("end", f"Starting run with {cfg.workers} worker(s)\n")
        log.see("end")

        def worker() -> None:
            try:
                run_analysis(cfg, output_path, progress=lambda m: q.put(m))
                q.put("__DONE__")
            except Exception:
                q.put("__ERROR__" + traceback.format_exc())
        threading.Thread(target=worker, daemon=True).start()

    run_button.configure(command=run_clicked)
    root.columnconfigure(1, weight=1)
    root.columnconfigure(3, weight=1)
    root.rowconfigure(5, weight=1)
    poll_log()
    root.mainloop()


# =============================================================================
# 15. ENTRY POINT
# =============================================================================

def main() -> None:
    mp.freeze_support()
    parser = build_parser()
    args = parser.parse_args()
    if args.self_check:
        run_self_checks()
        print("Self-checks passed", flush=True)
        return
    if not args.no_gui:
        launch_gui()
        return
    cfg = Config()
    if args.quick:
        cfg = quick_config(cfg)
    if args.single_frequency:
        cfg = replace(cfg, use_all_frequencies=False)
    output = Path(args.output) if args.output else default_output_dir()
    run_analysis(cfg, output, progress=console_progress)


if __name__ == "__main__":
    main()
