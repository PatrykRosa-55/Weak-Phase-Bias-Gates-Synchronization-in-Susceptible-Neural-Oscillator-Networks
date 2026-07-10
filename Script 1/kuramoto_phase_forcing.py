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
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
try:
    import scipy
    from scipy.stats import t as student_t
except Exception as exc:  # pragma: no cover - import-time dependency guard
    raise RuntimeError(
        "SciPy is required for exact t-based confidence intervals "
        "and for submission-ready reproducibility."
    ) from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =============================================================================
# 2. MODEL SCOPE, EXPERIMENTAL CONSTANTS, AND OUTPUT DEFINITIONS
# =============================================================================

SCRIPT_VERSION = "1.3.1-audit-clean"

TEMPLATE_FREQUENCIES_HZ = [7.83, 10.00, 14.30, 20.80, 27.30, 33.80]
CONDITIONS = [
    "NO_STIMULUS",
    "CLEAN_MATCHED",
    "LOW_JITTER_MATCHED",
    "HIGH_JITTER_MATCHED",
    "NEAR_DETUNED",
    "FAR_DETUNED",
    "SHARED_IRREGULAR_FORCING",
    "INDEPENDENT_IRREGULAR_FORCING",
]
STIMULUS_CONDITIONS = [c for c in CONDITIONS if c != "NO_STIMULUS"]
JITTER_LEVELS = [0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75]
COUPLING_MULTIPLIERS = [1.00, 1.07, 1.14]
FORCING_MULTIPLIERS = [1.00, 1.09, 1.18]
K_VALUES = [0.35, 0.50, 0.65, 0.80, 0.95, 1.10, 1.25, 1.40, 1.55]
FORCING_VALUES = [0.00, 0.03, 0.06, 0.09, 0.12, 0.18, 0.24]
FREQUENCY_SD_VALUES = [0.30, 0.75, 0.90, 1.20, 1.50]
NEAR_DETUNING_HZ = 2.0
FAR_DETUNING_HZ = 6.0
LOW_JITTER_LEVEL = 0.10
HIGH_JITTER_LEVEL = 0.50
DEFAULT_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))
LOCK_COL = "lock_fraction_R_ge_0_55"
STRONG_LOCK_COL = "strong_lock_fraction_R_ge_0_70"
STIMULUS_PLV_COL = "stimulus_plv"
CONDITION_LABELS = {
    "CLEAN_MATCHED": "Clean matched",
    "LOW_JITTER_MATCHED": "Low jitter",
    "HIGH_JITTER_MATCHED": "High jitter",
    "NEAR_DETUNED": "Near detuned",
    "FAR_DETUNED": "Far detuned",
    "SHARED_IRREGULAR_FORCING": "Shared irregular",
    "INDEPENDENT_IRREGULAR_FORCING": "Independent irregular",
}
CSV_FILES = [
    "core_metrics.csv",
    "configuration_parameters.csv",
    "environment_manifest.csv",
    "main_results.csv",
    "main_summary.csv",
    "paired_contrasts.csv",
    "supplementary_results.csv",
    "supplementary_summary.csv",
    "by_frequency_summary.csv",
    "stimulus_locking_summary.csv",
    "stimulus_locking_by_frequency_summary.csv",
]
FIGURE_FILES = [
    "figures/Fig1_phase_forcing_and_parameter_scan.png",
    "figures/Fig1_phase_forcing_and_parameter_scan.svg",
    "figures/Fig2_coupling_strength_scan.png",
    "figures/Fig2_coupling_strength_scan.svg",
    "figures/Fig3_jitter_and_forcing_strength_scan.png",
    "figures/Fig3_jitter_and_forcing_strength_scan.svg",
    "figures/FigS1_numerical_frequency_dispersion.png",
    "figures/FigS1_numerical_frequency_dispersion.svg",
    "figures/FigS2_stimulus_phase_locking.png",
    "figures/FigS2_stimulus_phase_locking.svg",
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


# =============================================================================
# 3. CONFIGURATION AND DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class Config:
    n_oscillators: int = 96
    duration_s: float = 12.0
    dt: float = 0.005
    burn_in_s: float = 2.0
    n_replicates: int = 24
    scan_replicates: int = 24
    robustness_replicates: int = 24
    workers: int = DEFAULT_WORKERS
    seed: int = 1234
    base_k_cycles_per_s: float = 0.72
    forcing_strength_cycles_per_s: float = 0.12
    phase_diffusion_cycles_per_sqrt_s: float = 0.035
    natural_frequency_sd_hz: float = 0.75
    individual_forcing_log_sd: float = 0.0
    lock_threshold: float = 0.55
    strong_lock_threshold: float = 0.70


@dataclass(frozen=True)
class Network:
    name: str
    frequency_hz: float
    description: str


@dataclass(frozen=True)
class Trial:
    theta0: np.ndarray
    frequency_z: np.ndarray
    omega: np.ndarray
    susceptibility: np.ndarray
    dW: np.ndarray
    unit_frequency_path: np.ndarray | None
    unit_phase_path: np.ndarray | None
    shared_irregular_phase: np.ndarray | None
    independent_irregular_phase: np.ndarray | None


NETWORKS = [Network(f"network_{f:.2f}Hz".replace(".", "_"), f, f"{f:.2f} Hz frequency template") for f in TEMPLATE_FREQUENCIES_HZ]


# =============================================================================
# 4. REPRODUCIBILITY, TIME GRID, AND CONFIGURATION VALIDATION
# =============================================================================


def stable_seed(*parts: object, base: int = 0) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "little") + base) % (2**32 - 1)


def n_steps_for(config: Config) -> int:
    return int(round(config.duration_s / config.dt))


def time_grid(config: Config) -> np.ndarray:
    return np.arange(n_steps_for(config), dtype=float) * config.dt


def is_integer_ratio(value: float, step: float) -> bool:
    ratio = value / step
    return math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9)


def validate_config(config: Config) -> None:
    if config.n_oscillators < 2:
        raise ValueError("n_oscillators must be >= 2")
    if config.duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if config.dt <= 0:
        raise ValueError("dt must be > 0")
    if not is_integer_ratio(config.duration_s, config.dt):
        raise ValueError("duration_s / dt must be an integer within numerical tolerance")
    if not (0 <= config.burn_in_s < config.duration_s):
        raise ValueError("burn_in_s must satisfy 0 <= burn_in_s < duration_s")
    if not is_integer_ratio(config.burn_in_s, config.dt):
        raise ValueError("burn_in_s / dt must be an integer within numerical tolerance")
    if config.n_replicates < 2:
        raise ValueError("n_replicates must be >= 2")
    if config.scan_replicates < 2:
        raise ValueError("scan_replicates must be >= 2")
    if config.robustness_replicates < 2:
        raise ValueError("robustness_replicates must be >= 2")
    if config.workers < 1:
        raise ValueError("workers must be >= 1")
    if config.natural_frequency_sd_hz <= 0:
        raise ValueError("natural_frequency_sd_hz must be > 0")
    if config.individual_forcing_log_sd < 0:
        raise ValueError("individual_forcing_log_sd must be >= 0")
    if not (0.0 <= config.lock_threshold <= 1.0):
        raise ValueError("lock_threshold must be in [0, 1]")
    if not (0.0 <= config.strong_lock_threshold <= 1.0):
        raise ValueError("strong_lock_threshold must be in [0, 1]")
    if config.strong_lock_threshold < config.lock_threshold:
        raise ValueError("strong_lock_threshold must be >= lock_threshold")


def default_output_dir() -> Path:
    return (Path(__file__).resolve().parent / "kuramoto_phase_forcing_outputs").resolve()


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


# =============================================================================
# 5. PHASE-INPUT GENERATION AND STOCHASTIC TRIAL CONSTRUCTION
# =============================================================================


def smooth_uniform_by_time(rng: np.random.Generator, t: np.ndarray, update_hz: float, low: float, high: float) -> np.ndarray:
    duration_s = float(t[-1] + (t[1] - t[0] if len(t) > 1 else 0.0)) if len(t) else 0.0
    anchor_times = np.arange(0.0, duration_s + 1.0 / update_hz, 1.0 / update_hz)
    values = rng.uniform(low, high, size=len(anchor_times))
    return np.interp(t, anchor_times, values)


def phase_from_instant_frequency(freq_hz: np.ndarray, dt: float, phase_deviation: np.ndarray | None = None) -> np.ndarray:
    phase = np.zeros_like(freq_hz, dtype=float)
    if len(phase) > 1:
        phase[1:] = 2.0 * np.pi * np.cumsum(freq_hz[:-1]) * dt
    if phase_deviation is not None:
        phase += phase_deviation - phase_deviation[0]
    phase[0] = 0.0
    return phase


def make_irregular_phase(rng: np.random.Generator, t: np.ndarray, dt: float) -> np.ndarray:
    freq = smooth_uniform_by_time(rng, t, update_hz=10.0, low=3.0, high=40.0)
    phase_jitter = smooth_uniform_by_time(rng, t, update_hz=18.0, low=-np.pi, high=np.pi)
    return phase_from_instant_frequency(freq, dt, phase_jitter)


def make_trial(
    config: Config,
    network: Network,
    replicate: int,
    need_jitter: bool = True,
    need_irregular: bool = True,
    need_independent_irregular: bool = True,
) -> Trial:
    n_steps = n_steps_for(config)
    t = time_grid(config)
    rng = np.random.default_rng(stable_seed("trial", network.name, replicate, config.dt, config.duration_s, config.n_oscillators, base=config.seed))
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=config.n_oscillators)
    frequency_z = rng.normal(loc=0.0, scale=1.0, size=config.n_oscillators)
    freq_hz = network.frequency_hz + config.natural_frequency_sd_hz * frequency_z
    omega = 2.0 * np.pi * freq_hz
    if config.individual_forcing_log_sd == 0.0:
        susceptibility = np.ones(config.n_oscillators, dtype=float)
    else:
        susceptibility = rng.lognormal(mean=0.0, sigma=config.individual_forcing_log_sd, size=config.n_oscillators)
        susceptibility = susceptibility / np.mean(susceptibility)
    dW = rng.normal(loc=0.0, scale=np.sqrt(config.dt), size=(n_steps, config.n_oscillators))
    unit_frequency_path = smooth_uniform_by_time(rng, t, update_hz=5.0, low=-1.0, high=1.0) if need_jitter else None
    unit_phase_path = smooth_uniform_by_time(rng, t, update_hz=8.0, low=-1.0, high=1.0) if need_jitter else None
    shared_irregular_phase = make_irregular_phase(rng, t, config.dt) if need_irregular else None
    independent_irregular_phase = (
        np.column_stack([make_irregular_phase(rng, t, config.dt) for _ in range(config.n_oscillators)])
        if need_independent_irregular
        else None
    )
    return Trial(theta0, frequency_z, omega, susceptibility, dW, unit_frequency_path, unit_phase_path, shared_irregular_phase, independent_irregular_phase)


# =============================================================================
# 6. PAIRED NUMERICAL CONTROLS AND NESTED TRIAL UTILITIES
# =============================================================================


def make_paired_trials(config: Config, network: Network, replicate: int, dt_coarse: float, dt_fine: float) -> tuple[Trial, Trial]:
    if not math.isclose(dt_coarse, 2.0 * dt_fine):
        raise ValueError("This paired comparison expects dt_coarse = 2 * dt_fine")
    fine_config = replace(config, dt=dt_fine)
    coarse_config = replace(config, dt=dt_coarse)
    n_fine = n_steps_for(fine_config)
    n_coarse = n_steps_for(coarse_config)
    if n_fine != 2 * n_coarse:
        raise ValueError("Fine grid must contain exactly two steps per coarse step")
    rng = np.random.default_rng(stable_seed("paired_dt", network.name, replicate, config.duration_s, config.n_oscillators, config.natural_frequency_sd_hz, base=config.seed))
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=config.n_oscillators)
    frequency_z = rng.normal(loc=0.0, scale=1.0, size=config.n_oscillators)
    freq_hz = network.frequency_hz + config.natural_frequency_sd_hz * frequency_z
    omega = 2.0 * np.pi * freq_hz
    if config.individual_forcing_log_sd == 0.0:
        susceptibility = np.ones(config.n_oscillators, dtype=float)
    else:
        susceptibility = rng.lognormal(mean=0.0, sigma=config.individual_forcing_log_sd, size=config.n_oscillators)
        susceptibility = susceptibility / np.mean(susceptibility)
    dW_fine = rng.normal(loc=0.0, scale=np.sqrt(dt_fine), size=(n_fine, config.n_oscillators))
    dW_coarse = dW_fine.reshape(n_coarse, 2, config.n_oscillators).sum(axis=1)
    assert np.allclose(dW_coarse, dW_fine.reshape(n_coarse, 2, config.n_oscillators).sum(axis=1))
    fine = Trial(theta0, frequency_z, omega, susceptibility, dW_fine, None, None, None, None)
    coarse = Trial(theta0, frequency_z, omega, susceptibility, dW_coarse, None, None, None, None)
    return coarse, fine


def slice_trial_time(trial: Trial, n_steps: int) -> Trial:
    return Trial(
        theta0=trial.theta0,
        frequency_z=trial.frequency_z,
        omega=trial.omega,
        susceptibility=trial.susceptibility,
        dW=trial.dW[:n_steps].copy(),
        unit_frequency_path=None if trial.unit_frequency_path is None else trial.unit_frequency_path[:n_steps].copy(),
        unit_phase_path=None if trial.unit_phase_path is None else trial.unit_phase_path[:n_steps].copy(),
        shared_irregular_phase=None if trial.shared_irregular_phase is None else trial.shared_irregular_phase[:n_steps].copy(),
        independent_irregular_phase=None if trial.independent_irregular_phase is None else trial.independent_irregular_phase[:n_steps].copy(),
    )


def slice_trial_oscillators(trial: Trial, n_oscillators: int) -> Trial:
    return Trial(
        theta0=trial.theta0[:n_oscillators].copy(),
        frequency_z=trial.frequency_z[:n_oscillators].copy(),
        omega=trial.omega[:n_oscillators].copy(),
        susceptibility=trial.susceptibility[:n_oscillators].copy(),
        dW=trial.dW[:, :n_oscillators].copy(),
        unit_frequency_path=trial.unit_frequency_path,
        unit_phase_path=trial.unit_phase_path,
        shared_irregular_phase=trial.shared_irregular_phase,
        independent_irregular_phase=None if trial.independent_irregular_phase is None else trial.independent_irregular_phase[:, :n_oscillators].copy(),
    )


# =============================================================================
# 7. KURAMOTO DYNAMICS AND PHASE-FORCING CONDITIONS
# =============================================================================


def analysis_window(config: Config) -> slice:
    burn = int(math.ceil(config.burn_in_s / config.dt))
    return slice(burn, n_steps_for(config))


def order_parameter(theta: np.ndarray) -> tuple[float, float]:
    z = np.mean(np.exp(1j * theta))
    return float(abs(z)), float(np.angle(z))


def forcing_phase(config: Config, network: Network, trial: Trial, condition: str, frequency_hz: float | None = None, jitter_level: float | None = None) -> tuple[np.ndarray, bool, float, float]:
    t = time_grid(config)
    f0 = network.frequency_hz if frequency_hz is None else float(frequency_hz)
    if condition == "NO_STIMULUS":
        phase = np.zeros_like(t)
        return phase, False, 0.0, 0.0
    if condition == "CLEAN_MATCHED":
        phase = 2.0 * np.pi * f0 * t
        phase[0] = 0.0
        return phase, False, config.forcing_strength_cycles_per_s, 0.0
    if condition in {"LOW_JITTER_MATCHED", "HIGH_JITTER_MATCHED"} or jitter_level is not None:
        if jitter_level is not None:
            level = float(jitter_level)
        elif condition == "LOW_JITTER_MATCHED":
            level = LOW_JITTER_LEVEL
        elif condition == "HIGH_JITTER_MATCHED":
            level = HIGH_JITTER_LEVEL
        else:
            raise ValueError("A jitter level is required")
        if trial.unit_frequency_path is None or trial.unit_phase_path is None:
            raise ValueError("Jitter paths were not generated for this trial")
        frequency_deviation = level * f0 * trial.unit_frequency_path
        phase_deviation = level * np.pi * trial.unit_phase_path
        inst_freq = np.maximum(0.05, f0 + frequency_deviation)
        phase = phase_from_instant_frequency(inst_freq, config.dt, phase_deviation)
        return phase, False, config.forcing_strength_cycles_per_s, 0.0
    if condition == "SHARED_IRREGULAR_FORCING":
        if trial.shared_irregular_phase is None:
            raise ValueError("Shared irregular path was not generated for this trial")
        return trial.shared_irregular_phase, False, config.forcing_strength_cycles_per_s, 0.0
    if condition == "INDEPENDENT_IRREGULAR_FORCING":
        if trial.independent_irregular_phase is None:
            raise ValueError("Independent irregular paths were not generated for this trial")
        return trial.independent_irregular_phase, True, config.forcing_strength_cycles_per_s, 0.0
    raise ValueError(f"Unsupported direct forcing condition: {condition}")


# Integrates the noisy forced Kuramoto system and returns synchrony metrics.
def simulate_phase_input(
    config: Config,
    network: Network,
    trial: Trial,
    condition: str,
    phase_input: np.ndarray,
    matrix_input: bool,
    effective_forcing_cycles_per_s: float,
    coupling_multiplier: float = 1.0,
    forcing_multiplier: float = 1.0,
    base_k_cycles_per_s: float | None = None,
    forcing_strength_cycles_per_s: float | None = None,
    return_trace: bool = False,
) -> dict[str, object]:
    n_steps = n_steps_for(config)
    theta = trial.theta0.copy()
    k_base = config.base_k_cycles_per_s if base_k_cycles_per_s is None else float(base_k_cycles_per_s)
    f_base = effective_forcing_cycles_per_s if forcing_strength_cycles_per_s is None else float(forcing_strength_cycles_per_s)
    if condition == "NO_STIMULUS":
        f_base = 0.0
    k_eff = 2.0 * np.pi * k_base * coupling_multiplier
    s_eff = 2.0 * np.pi * f_base * forcing_multiplier
    measure_stimulus_locking = condition != "NO_STIMULUS" and s_eff > 0.0
    window = analysis_window(config)
    r_trace = np.zeros(n_steps)
    stimulus_complex_sum = 0.0 + 0.0j
    stimulus_sample_count = 0
    for step in range(n_steps):
        r, psi = order_parameter(theta)
        r_trace[step] = r
        if measure_stimulus_locking and step >= window.start and step < window.stop:
            if matrix_input:
                phase_difference = theta - phase_input[step]
            else:
                phase_difference = theta - float(phase_input[step])
            stimulus_complex_sum += np.sum(np.exp(1j * phase_difference))
            stimulus_sample_count += int(np.size(phase_difference))
        coupling = k_eff * r * np.sin(psi - theta)
        if matrix_input:
            stimulus = s_eff * trial.susceptibility * np.sin(phase_input[step] - theta)
        else:
            stimulus = s_eff * trial.susceptibility * np.sin(float(phase_input[step]) - theta)
        deterministic_term = trial.omega + coupling + stimulus
        theta = (
            theta
            + deterministic_term * config.dt
            + 2.0 * np.pi * config.phase_diffusion_cycles_per_sqrt_s * trial.dW[step]
        ) % (2.0 * np.pi)
    r_eval = r_trace[window]
    result = {
        "mean_R": float(np.mean(r_eval)),
        "max_R": float(np.max(r_eval)),
        LOCK_COL: float(np.mean(r_eval >= config.lock_threshold)),
        STRONG_LOCK_COL: float(np.mean(r_eval >= config.strong_lock_threshold)),
        STIMULUS_PLV_COL: float(abs(stimulus_complex_sum) / stimulus_sample_count) if stimulus_sample_count else float("nan"),
    }
    if return_trace:
        result["R_trace"] = r_trace
    return result

# Builds and evaluates one experimental forcing condition.
def simulate_condition(
    config: Config,
    network: Network,
    trial: Trial,
    condition: str,
    replicate: int,
    coupling_multiplier: float = 1.0,
    forcing_multiplier: float = 1.0,
    base_k_cycles_per_s: float | None = None,
    forcing_strength_cycles_per_s: float | None = None,
    jitter_level: float | None = None,
    return_trace: bool = False,
) -> dict[str, object]:
    if condition in {"NEAR_DETUNED", "FAR_DETUNED"}:
        detuning = NEAR_DETUNING_HZ if condition == "NEAR_DETUNED" else FAR_DETUNING_HZ
        metrics = []
        traces = []
        for sign in (-1.0, 1.0):
            phase, matrix, eff, _ = forcing_phase(config, network, trial, "CLEAN_MATCHED", frequency_hz=network.frequency_hz + sign * detuning)
            m = simulate_phase_input(config, network, trial, condition, phase, matrix, eff, coupling_multiplier, forcing_multiplier, base_k_cycles_per_s, forcing_strength_cycles_per_s, return_trace)
            if return_trace:
                traces.append(m.pop("R_trace"))
            metrics.append(m)
        out = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}
        if return_trace:
            out["R_trace"] = np.mean(np.vstack(traces), axis=0)
        detuning_hz = detuning
    else:
        phase, matrix, eff, detuning_hz = forcing_phase(config, network, trial, condition, jitter_level=jitter_level)
        out = simulate_phase_input(config, network, trial, condition, phase, matrix, eff, coupling_multiplier, forcing_multiplier, base_k_cycles_per_s, forcing_strength_cycles_per_s, return_trace)
    out.update({
        "network": network.name,
        "network_frequency_hz": network.frequency_hz,
        "replicate": int(replicate),
        "condition": condition,
        "detuning_hz": float(detuning_hz),
        "coupling_multiplier": float(coupling_multiplier),
        "forcing_multiplier": float(forcing_multiplier),
        "base_k_cycles_per_s": float(config.base_k_cycles_per_s if base_k_cycles_per_s is None else base_k_cycles_per_s),
        "forcing_strength_cycles_per_s": float(config.forcing_strength_cycles_per_s if forcing_strength_cycles_per_s is None else forcing_strength_cycles_per_s),
        "effective_k_cycles_per_s": float(config.base_k_cycles_per_s if base_k_cycles_per_s is None else base_k_cycles_per_s) * coupling_multiplier,
        "effective_forcing_cycles_per_s": 0.0 if condition == "NO_STIMULUS" else float(config.forcing_strength_cycles_per_s if forcing_strength_cycles_per_s is None else forcing_strength_cycles_per_s) * forcing_multiplier,
        "duration_s": config.duration_s,
        "dt": config.dt,
        "burn_in_s": config.burn_in_s,
        "n_oscillators": config.n_oscillators,
    })
    return out


# =============================================================================
# 8. PAIRED STATISTICS AND REPLICATE-LEVEL SUMMARIES
# =============================================================================


def ci_stats(values: Iterable[float]) -> dict[str, float | int | str]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(len(arr))

    if n == 0:
        return {
            "mean": float("nan"),
            "sd": float("nan"),
            "sem": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "ci95_halfwidth": float("nan"),
            "n_blocks": 0,
            "ci_method": "undefined_n_0",
        }

    mean = float(np.mean(arr))

    if n == 1:
        return {
            "mean": mean,
            "sd": 0.0,
            "sem": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
            "ci95_halfwidth": 0.0,
            "n_blocks": 1,
            "ci_method": "undefined_n_lt_2",
        }

    sd = float(np.std(arr, ddof=1))
    sem = float(sd / math.sqrt(n))
    ci = float(student_t.ppf(0.975, df=n - 1) * sem)
    return {
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
        "ci95_halfwidth": ci,
        "n_blocks": n,
        "ci_method": "student_t",
    }


def summarize_delta(df: pd.DataFrame, groups: list[str], value_col: str = "delta_mean_R") -> pd.DataFrame:
    replicate_groups = groups + ["replicate"]
    replicate_level = df.groupby(replicate_groups, dropna=False)[value_col].mean().reset_index()
    rows = []
    for keys, part in replicate_level.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        stats = ci_stats(part[value_col])
        row.update({f"{value_col}_{k}": v for k, v in stats.items() if k != "n_blocks"})
        row["n_blocks"] = stats["n_blocks"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def summarize_delta_by_frequency(df: pd.DataFrame, groups: list[str], value_col: str = "delta_mean_R") -> pd.DataFrame:
    freq_groups = groups + ["network", "network_frequency_hz"]
    rows = []
    for keys, part in df.groupby(freq_groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(freq_groups, keys))
        stats = ci_stats(part[value_col])
        row.update({f"{value_col}_{k}": v for k, v in stats.items() if k != "n_blocks"})
        row["n_blocks"] = stats["n_blocks"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(freq_groups).reset_index(drop=True)


def summarize_observed(df: pd.DataFrame, groups: list[str], value_cols: list[str]) -> pd.DataFrame:
    replicate_groups = groups + ["replicate"]
    replicate_level = df.groupby(replicate_groups, dropna=False)[value_cols].mean().reset_index()
    rows = []
    for keys, part in replicate_level.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        for value_col in value_cols:
            stats = ci_stats(part[value_col])
            row.update({f"{value_col}_{k}": v for k, v in stats.items() if k != "n_blocks"})
        row["n_blocks"] = int(len(part))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def summarize_observed_by_frequency(df: pd.DataFrame, groups: list[str], value_cols: list[str]) -> pd.DataFrame:
    freq_groups = groups + ["network", "network_frequency_hz"]
    rows = []
    for keys, part in df.groupby(freq_groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(freq_groups, keys))
        for value_col in value_cols:
            stats = ci_stats(part[value_col])
            row.update({f"{value_col}_{k}": v for k, v in stats.items() if k != "n_blocks"})
        row["n_blocks"] = int(len(part))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(freq_groups).reset_index(drop=True)


def summarize_core_conditions(delta: pd.DataFrame) -> pd.DataFrame:
    replicate_level = delta.groupby(["condition", "replicate"], dropna=False).agg(
        delta_mean_R=("delta_mean_R", "mean"),
        **{f"delta_{LOCK_COL}": (f"delta_{LOCK_COL}", "mean"), f"delta_{STRONG_LOCK_COL}": (f"delta_{STRONG_LOCK_COL}", "mean")},
    ).reset_index()
    rows = []
    for condition, part in replicate_level.groupby("condition", dropna=False):
        row = {"condition": condition}
        row.update({f"delta_mean_R_{k}": v for k, v in ci_stats(part["delta_mean_R"]).items() if k != "n_blocks"})
        row["n_blocks"] = int(len(part))
        for col in [f"delta_{LOCK_COL}", f"delta_{STRONG_LOCK_COL}"]:
            row[f"{col}_mean"] = float(part[col].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


def summarize_core_conditions_by_frequency(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = ["condition", "network", "network_frequency_hz"]
    for keys, part in delta.groupby(groups, dropna=False):
        row = dict(zip(groups, keys))
        row.update({f"delta_mean_R_{k}": v for k, v in ci_stats(part["delta_mean_R"]).items() if k != "n_blocks"})
        row["n_blocks"] = int(len(part))
        for col in [f"delta_{LOCK_COL}", f"delta_{STRONG_LOCK_COL}"]:
            row[f"{col}_mean"] = float(part[col].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def core_paired_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["network", "network_frequency_hz", "replicate"]
    base_cols = index_cols + ["mean_R", "max_R", LOCK_COL, STRONG_LOCK_COL]
    base = metrics[metrics["condition"] == "NO_STIMULUS"][base_cols].rename(columns={
        "mean_R": "base_mean_R",
        "max_R": "base_max_R",
        LOCK_COL: f"base_{LOCK_COL}",
        STRONG_LOCK_COL: f"base_{STRONG_LOCK_COL}",
    })
    out = metrics[metrics["condition"] != "NO_STIMULUS"].merge(base, on=index_cols, how="left")
    out["delta_mean_R"] = out["mean_R"] - out["base_mean_R"]
    out["delta_max_R"] = out["max_R"] - out["base_max_R"]
    out[f"delta_{LOCK_COL}"] = out[LOCK_COL] - out[f"base_{LOCK_COL}"]
    out[f"delta_{STRONG_LOCK_COL}"] = out[STRONG_LOCK_COL] - out[f"base_{STRONG_LOCK_COL}"]
    return out.sort_values(["condition", "network", "replicate"]).reset_index(drop=True)


def paired_condition_contrasts(delta: pd.DataFrame) -> pd.DataFrame:
    piv = delta.pivot_table(index=["network", "replicate"], columns="condition", values="delta_mean_R")
    comparisons = [
        ("CLEAN_MATCHED_minus_LOW_JITTER_MATCHED", "CLEAN_MATCHED", "LOW_JITTER_MATCHED"),
        ("CLEAN_MATCHED_minus_HIGH_JITTER_MATCHED", "CLEAN_MATCHED", "HIGH_JITTER_MATCHED"),
        ("CLEAN_MATCHED_minus_NEAR_DETUNED", "CLEAN_MATCHED", "NEAR_DETUNED"),
        ("CLEAN_MATCHED_minus_FAR_DETUNED", "CLEAN_MATCHED", "FAR_DETUNED"),
        ("CLEAN_MATCHED_minus_SHARED_IRREGULAR_FORCING", "CLEAN_MATCHED", "SHARED_IRREGULAR_FORCING"),
        ("CLEAN_MATCHED_minus_INDEPENDENT_IRREGULAR_FORCING", "CLEAN_MATCHED", "INDEPENDENT_IRREGULAR_FORCING"),
        ("LOW_JITTER_MATCHED_minus_HIGH_JITTER_MATCHED", "LOW_JITTER_MATCHED", "HIGH_JITTER_MATCHED"),
    ]
    rows = []
    for name, a, b in comparisons:
        per_network = (piv[a] - piv[b]).dropna().rename("difference").reset_index()
        per_replicate = per_network.groupby("replicate", dropna=False)["difference"].mean()
        stats = ci_stats(per_replicate)
        sd = float(np.std(per_replicate, ddof=1)) if len(per_replicate) > 1 else 0.0
        mean = float(np.mean(per_replicate)) if len(per_replicate) else float("nan")
        rows.append({"contrast": name, **stats, "cohen_dz": float(mean / sd) if sd > 0 else float("nan")})
    return pd.DataFrame(rows)


# =============================================================================
# 9. PARALLEL EXECUTION
# =============================================================================


def run_blocks(
    config: Config,
    jobs: list[tuple],
    worker_func: Callable,
    progress: Callable[[str], None] | None = None,
    executor: cf.Executor | None = None,
    allow_fallback: bool = False,
) -> list:
    if not jobs:
        return []

    def sequential() -> list:
        rows = []
        for i, job in enumerate(jobs, 1):
            rows.extend(worker_func(job))
            if progress and (i == len(jobs) or i % max(1, len(jobs) // 10) == 0):
                progress(f"Completed {i}/{len(jobs)} blocks")
        return rows

    if config.workers == 1 or executor is None:
        if config.workers > 1 and executor is None and not allow_fallback:
            raise ValueError("An executor is required when workers > 1 and fallback is disabled")
        return sequential()

    rows = []
    futures = [executor.submit(worker_func, job) for job in jobs]
    for i, fut in enumerate(cf.as_completed(futures), 1):
        rows.extend(fut.result())
        if progress and (i == len(jobs) or i % max(1, len(jobs) // 10) == 0):
            progress(f"Completed {i}/{len(jobs)} blocks")
    return rows


# =============================================================================
# 10. BLOCK-LEVEL SIMULATION WORKERS
# =============================================================================


# Evaluates all primary forcing conditions for one network-replicate block.
def core_block_worker(job: tuple[Config, str, int]) -> list[dict[str, object]]:
    config, network_name, replicate = job
    network = next(n for n in NETWORKS if n.name == network_name)
    trial = make_trial(config, network, replicate, need_jitter=True, need_irregular=True, need_independent_irregular=True)
    return [simulate_condition(config, network, trial, condition, replicate) for condition in CONDITIONS]


# Evaluates one parameter scan using shared random inputs within a block.
def scan_pair_block_worker(job: tuple[Config, str, int, str, tuple[float, ...]]) -> list[dict[str, object]]:
    config, network_name, replicate, scan_name, values = job
    network = next(n for n in NETWORKS if n.name == network_name)
    if scan_name not in {"coupling_strength", "forcing_strength", "jitter"}:
        raise ValueError(scan_name)
    trial = make_trial(
        config,
        network,
        replicate,
        need_jitter=(scan_name == "jitter"),
        need_irregular=False,
        need_independent_irregular=False,
    )
    rows = []
    for value in values:
        kwargs = {}
        if scan_name == "coupling_strength":
            kwargs["base_k_cycles_per_s"] = value
        elif scan_name == "forcing_strength":
            kwargs["forcing_strength_cycles_per_s"] = value
        elif scan_name == "jitter":
            kwargs["jitter_level"] = value
        base = simulate_condition(config, network, trial, "NO_STIMULUS", replicate, **{k: v for k, v in kwargs.items() if k != "jitter_level"})
        if scan_name == "jitter":
            stim_condition = "LOW_JITTER_MATCHED"
            kwargs["jitter_level"] = float(value)
        else:
            stim_condition = "CLEAN_MATCHED"
        stim = simulate_condition(config, network, trial, stim_condition, replicate, **kwargs)
        row = {
            "network": network.name,
            "network_frequency_hz": network.frequency_hz,
            "replicate": replicate,
            "base_mean_R": base["mean_R"],
            "stimulated_mean_R": stim["mean_R"],
            "delta_mean_R": stim["mean_R"] - base["mean_R"],
            f"delta_{LOCK_COL}": stim[LOCK_COL] - base[LOCK_COL],
        }
        if scan_name == "coupling_strength":
            row["coupling_strength_K_cycles_per_s"] = value
        elif scan_name == "forcing_strength":
            row["forcing_strength_cycles_per_s"] = value
        else:
            row["jitter_level"] = value
        rows.append(row)
    return rows


# Evaluates the coupling-by-forcing factorial design.
def factorial_block_worker(job: tuple[Config, str, int]) -> list[dict[str, object]]:
    config, network_name, replicate = job
    network = next(n for n in NETWORKS if n.name == network_name)
    trial = make_trial(config, network, replicate, need_jitter=False, need_irregular=False, need_independent_irregular=False)
    rows = []
    for cm in COUPLING_MULTIPLIERS:
        for fm in FORCING_MULTIPLIERS:
            base = simulate_condition(config, network, trial, "NO_STIMULUS", replicate, coupling_multiplier=cm, forcing_multiplier=fm)
            stim = simulate_condition(config, network, trial, "CLEAN_MATCHED", replicate, coupling_multiplier=cm, forcing_multiplier=fm)
            rows.append({
                "coupling_multiplier": cm,
                "forcing_multiplier": fm,
                "network": network.name,
                "network_frequency_hz": network.frequency_hz,
                "replicate": replicate,
                "base_mean_R": base["mean_R"],
                "stimulated_mean_R": stim["mean_R"],
                "delta_mean_R": stim["mean_R"] - base["mean_R"],
            })
    return rows


# Evaluates natural-frequency dispersion using shared random realizations.
def frequency_dispersion_block_worker(job: tuple[Config, str, int]) -> list[dict[str, object]]:
    base_config, network_name, replicate = job
    network = next(n for n in NETWORKS if n.name == network_name)
    rows = []
    for frequency_sd in FREQUENCY_SD_VALUES:
        config = replace(base_config, natural_frequency_sd_hz=frequency_sd)
        trial = make_trial(config, network, replicate, need_jitter=True, need_irregular=True, need_independent_irregular=True)
        base = simulate_condition(config, network, trial, "NO_STIMULUS", replicate)
        for condition in STIMULUS_CONDITIONS:
            stim = simulate_condition(config, network, trial, condition, replicate)
            rows.append({
                "natural_frequency_sd_hz": frequency_sd,
                "condition": condition,
                "network": network.name,
                "network_frequency_hz": network.frequency_hz,
                "replicate": replicate,
                "base_mean_R": base["mean_R"],
                "stimulated_mean_R": stim["mean_R"],
                "delta_mean_R": stim["mean_R"] - base["mean_R"],
            })
    return rows


# Evaluates duration and population-size robustness using a nested master trial.
def additional_robustness_block_worker(job: tuple[Config, str, int]) -> list[dict[str, object]]:
    base_config, network_name, replicate = job
    network = next(n for n in NETWORKS if n.name == network_name)
    n_master = max(192, base_config.n_oscillators)
    duration_master = max(24.0, base_config.duration_s)
    master_config = replace(base_config, n_oscillators=n_master, duration_s=duration_master)
    master_trial = make_trial(
        master_config,
        network,
        replicate,
        need_jitter=False,
        need_irregular=False,
        need_independent_irregular=False,
    )
    scenario_configs_trials = []

    baseline_config = replace(base_config, duration_s=base_config.duration_s, n_oscillators=base_config.n_oscillators)
    baseline_trial = slice_trial_oscillators(slice_trial_time(master_trial, n_steps_for(baseline_config)), base_config.n_oscillators)
    scenario_configs_trials.append(("baseline", baseline_config, baseline_trial))

    longer_config = replace(base_config, duration_s=duration_master, n_oscillators=base_config.n_oscillators)
    longer_trial = slice_trial_oscillators(master_trial, base_config.n_oscillators)
    scenario_configs_trials.append(("longer_duration_24s", longer_config, longer_trial))

    larger_config = replace(base_config, duration_s=base_config.duration_s, n_oscillators=n_master)
    larger_trial = slice_trial_time(master_trial, n_steps_for(larger_config))
    scenario_configs_trials.append(("larger_population_N192", larger_config, larger_trial))

    rows = []
    for scenario, config, trial in scenario_configs_trials:
        base = simulate_condition(config, network, trial, "NO_STIMULUS", replicate)
        stim = simulate_condition(config, network, trial, "CLEAN_MATCHED", replicate)
        rows.append({
            "scenario": scenario,
            "network": network.name,
            "network_frequency_hz": network.frequency_hz,
            "replicate": replicate,
            "n_oscillators": config.n_oscillators,
            "duration_s": config.duration_s,
            "base_mean_R": base["mean_R"],
            "stimulated_mean_R": stim["mean_R"],
            "delta_mean_R": stim["mean_R"] - base["mean_R"],
        })
    return rows


# Compares coarse and fine integration steps on a shared Brownian path.
def paired_time_step_block_worker(job: tuple[Config, str, int]) -> list[dict[str, object]]:
    config, network_name, replicate = job
    network = next(n for n in NETWORKS if n.name == network_name)
    dt_coarse = config.dt
    dt_fine = config.dt / 2.0
    coarse_config = replace(config, dt=dt_coarse)
    fine_config = replace(config, dt=dt_fine)
    coarse_trial, fine_trial = make_paired_trials(config, network, replicate, dt_coarse, dt_fine)
    c_base = simulate_condition(coarse_config, network, coarse_trial, "NO_STIMULUS", replicate, return_trace=True)
    c_stim = simulate_condition(coarse_config, network, coarse_trial, "CLEAN_MATCHED", replicate, return_trace=True)
    f_base = simulate_condition(fine_config, network, fine_trial, "NO_STIMULUS", replicate, return_trace=True)
    f_stim = simulate_condition(fine_config, network, fine_trial, "CLEAN_MATCHED", replicate, return_trace=True)
    coarse_delta = c_stim["mean_R"] - c_base["mean_R"]
    fine_delta = f_stim["mean_R"] - f_base["mean_R"]
    fine_trace_on_coarse = np.asarray(f_stim["R_trace"])[::2]
    coarse_trace = np.asarray(c_stim["R_trace"])
    burn = analysis_window(coarse_config).start
    n = min(len(fine_trace_on_coarse), len(coarse_trace))
    rmse = float(np.sqrt(np.mean((fine_trace_on_coarse[burn:n] - coarse_trace[burn:n]) ** 2)))
    return [{
        "network": network.name,
        "network_frequency_hz": network.frequency_hz,
        "replicate": replicate,
        "coarse_dt_s": dt_coarse,
        "fine_dt_s": dt_fine,
        "coarse_delta_mean_R": coarse_delta,
        "fine_delta_mean_R": fine_delta,
        "fine_minus_coarse_delta_mean_R": fine_delta - coarse_delta,
        "absolute_delta_difference": abs(fine_delta - coarse_delta),
        "stimulated_R_trace_rmse": rmse,
    }]


# =============================================================================
# 11. EXPERIMENT AND PARAMETER-SCAN ORCHESTRATION
# =============================================================================


def run_core_experiment(config: Config, progress=None, executor: cf.Executor | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jobs = [(config, n.name, r) for n in NETWORKS for r in range(config.n_replicates)]
    rows = run_blocks(config, jobs, core_block_worker, progress, executor=executor)
    metrics = pd.DataFrame(rows).sort_values(["network", "replicate", "condition"]).reset_index(drop=True)
    deltas = core_paired_deltas(metrics)
    summary = summarize_core_conditions(deltas)
    by_frequency = summarize_core_conditions_by_frequency(deltas)
    stimulus_metrics = metrics[metrics["condition"] != "NO_STIMULUS"].copy()
    stimulus_locking_summary = summarize_observed(
        stimulus_metrics,
        ["condition"],
        [STIMULUS_PLV_COL],
    )
    stimulus_locking_by_frequency = summarize_observed_by_frequency(
        stimulus_metrics,
        ["condition"],
        [STIMULUS_PLV_COL],
    )
    contrasts = paired_condition_contrasts(deltas)
    return metrics, deltas, summary, by_frequency, stimulus_locking_summary, stimulus_locking_by_frequency, contrasts

def run_coupling_forcing_factorial_scan(config: Config, progress=None, executor: cf.Executor | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jobs = [(config, n.name, r) for n in NETWORKS for r in range(config.scan_replicates)]
    df = pd.DataFrame(run_blocks(config, jobs, factorial_block_worker, progress, executor=executor)).sort_values(["coupling_multiplier", "forcing_multiplier", "network", "replicate"]).reset_index(drop=True)
    summary = summarize_delta(df, ["coupling_multiplier", "forcing_multiplier"])
    by_frequency = summarize_delta_by_frequency(df, ["coupling_multiplier", "forcing_multiplier"])
    return df, summary, by_frequency


def run_simple_scan(config: Config, scan_name: str, values: list[float], progress=None, executor: cf.Executor | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jobs = [(config, n.name, r, scan_name, tuple(values)) for n in NETWORKS for r in range(config.scan_replicates)]
    df = pd.DataFrame(run_blocks(config, jobs, scan_pair_block_worker, progress, executor=executor))
    key = {"coupling_strength": "coupling_strength_K_cycles_per_s", "forcing_strength": "forcing_strength_cycles_per_s", "jitter": "jitter_level"}[scan_name]
    df = df.sort_values([key, "network", "replicate"]).reset_index(drop=True)
    return df, summarize_delta(df, [key]), summarize_delta_by_frequency(df, [key])


def run_frequency_dispersion_scan(config: Config, progress=None, executor: cf.Executor | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jobs = [(config, n.name, r) for n in NETWORKS for r in range(config.robustness_replicates)]
    df = pd.DataFrame(run_blocks(config, jobs, frequency_dispersion_block_worker, progress, executor=executor)).sort_values(["natural_frequency_sd_hz", "condition", "network", "replicate"]).reset_index(drop=True)
    return df, summarize_delta(df, ["natural_frequency_sd_hz", "condition"]), summarize_delta_by_frequency(df, ["natural_frequency_sd_hz", "condition"])


def run_paired_time_step_convergence(config: Config, progress=None, executor: cf.Executor | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = [(config, n.name, r) for n in NETWORKS for r in range(config.robustness_replicates)]
    df = pd.DataFrame(run_blocks(config, jobs, paired_time_step_block_worker, progress, executor=executor)).sort_values(["network", "replicate"]).reset_index(drop=True)
    rows = []
    for col in ["coarse_delta_mean_R", "fine_delta_mean_R", "fine_minus_coarse_delta_mean_R", "absolute_delta_difference", "stimulated_R_trace_rmse"]:
        replicate_values = df.groupby("replicate", dropna=False)[col].mean()
        row = {"metric": col}
        row.update(ci_stats(replicate_values))
        if col == "coarse_delta_mean_R":
            row["dt_s"] = float(df["coarse_dt_s"].iloc[0])
        elif col == "fine_delta_mean_R":
            row["dt_s"] = float(df["fine_dt_s"].iloc[0])
        else:
            row["dt_s"] = np.nan
        rows.append(row)
    return df, pd.DataFrame(rows)


def run_additional_robustness(config: Config, progress=None, executor: cf.Executor | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = [(config, n.name, r) for n in NETWORKS for r in range(config.robustness_replicates)]
    df = pd.DataFrame(run_blocks(config, jobs, additional_robustness_block_worker, progress, executor=executor)).sort_values(["scenario", "network", "replicate"]).reset_index(drop=True)
    baseline = df[df["scenario"] == "baseline"][["network", "replicate", "delta_mean_R"]].rename(columns={"delta_mean_R": "baseline_delta_mean_R"})
    df = df.merge(baseline, on=["network", "replicate"], how="left")
    df["delta_mean_R_minus_baseline"] = df["delta_mean_R"] - df["baseline_delta_mean_R"]
    df = df.sort_values(["scenario", "network", "replicate"]).reset_index(drop=True)
    summary = summarize_delta(df, ["scenario"])
    diff_summary = summarize_delta(df, ["scenario"], value_col="delta_mean_R_minus_baseline")
    return df, summary.merge(diff_summary, on=["scenario", "n_blocks"], how="left")


# =============================================================================
# 12. REPRODUCIBILITY TABLES AND METADATA
# =============================================================================



def write_definitions(config: Config, output: Path) -> None:
    pd.DataFrame([vars(config)]).T.reset_index().rename(columns={"index": "parameter", 0: "value"}).to_csv(output / "configuration_parameters.csv", index=False)


def write_environment_manifest(config: Config, output: Path) -> None:
    code_bytes = Path(__file__).read_bytes()
    pd.DataFrame([{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "script_name": Path(__file__).name,
        "python_version": sys.version.replace("\n", " "),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "platform": platform_mod.platform(),
        "cpu_count": os.cpu_count(),
        "workers": config.workers,
        "code_sha256": hashlib.sha256(code_bytes).hexdigest(),
    }]).to_csv(output / "environment_manifest.csv", index=False)


# =============================================================================
# 13. PUBLICATION FIGURES
# =============================================================================


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)


# Generates the main and supplementary publication figures from summary tables.
def make_figures(output: Path, core_summary: pd.DataFrame, factorial_summary: pd.DataFrame, coupling_summary: pd.DataFrame, jitter_summary: pd.DataFrame, forcing_summary: pd.DataFrame, freq_summary: pd.DataFrame, dt_summary: pd.DataFrame, stimulus_locking_summary: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(PANEL_WIDTH_IN * 2, PANEL_HEIGHT_IN), constrained_layout=True)
    core_plot = core_summary.set_index("condition").loc[STIMULUS_CONDITIONS].reset_index()
    x = np.arange(len(core_plot))
    axes[0].bar(x, core_plot["delta_mean_R_mean"], color="#4C78A8")
    axes[0].errorbar(x, core_plot["delta_mean_R_mean"], yerr=core_plot["delta_mean_R_ci95_halfwidth"], fmt="none", color="black", capsize=CAPSIZE)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([CONDITION_LABELS[c] for c in core_plot["condition"]], rotation=45, ha="right")
    axes[0].set_ylabel(r"Mean synchrony change, $\Delta \bar{R}$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].set_title("A. Phase-forcing condition comparison", loc="left", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
    style_axes(axes[0])
    heat = factorial_summary.pivot(index="coupling_multiplier", columns="forcing_multiplier", values="delta_mean_R_mean").sort_index(ascending=True)
    im = axes[1].imshow(heat.values, origin="lower", cmap="viridis", aspect="auto")
    axes[1].set_xticks(np.arange(len(heat.columns)))
    axes[1].set_xticklabels([f"{v:.2f}" for v in heat.columns])
    axes[1].set_yticks(np.arange(len(heat.index)))
    axes[1].set_yticklabels([f"{v:.2f}" for v in heat.index])
    axes[1].set_xlabel("Forcing multiplier", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_ylabel("Coupling multiplier", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_title("B. Coupling-forcing multiplier scan", loc="left", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            axes[1].text(j, i, f"{heat.values[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=axes[1], label=r"Mean synchrony change, $\Delta \bar{R}$")
    fig.suptitle("Frequency-matched phase forcing and parameter sensitivity", fontsize=TITLE_FONTSIZE, fontweight="bold")
    fig.savefig(figure_dir / "Fig1_phase_forcing_and_parameter_scan.png", dpi=FIGURE_DPI)
    fig.savefig(figure_dir / "Fig1_phase_forcing_and_parameter_scan.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(PANEL_WIDTH_IN, PANEL_HEIGHT_IN), constrained_layout=True)
    ax.errorbar(coupling_summary["coupling_strength_K_cycles_per_s"], coupling_summary["delta_mean_R_mean"], yerr=coupling_summary["delta_mean_R_ci95_halfwidth"], marker="o", linewidth=LINE_WIDTH, color="#F58518", capsize=CAPSIZE)
    ax.set_xlabel("Coupling strength, K (cycles/s)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Mean synchrony change, $\Delta \bar{R}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Coupling-dependent phase-forcing effect", fontsize=TITLE_FONTSIZE, fontweight="bold")
    style_axes(ax)
    fig.savefig(figure_dir / "Fig2_coupling_strength_scan.png", dpi=FIGURE_DPI)
    fig.savefig(figure_dir / "Fig2_coupling_strength_scan.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(PANEL_WIDTH_IN * 2, PANEL_HEIGHT_IN), constrained_layout=True)
    axes[0].errorbar(jitter_summary["jitter_level"], jitter_summary["delta_mean_R_mean"], yerr=jitter_summary["delta_mean_R_ci95_halfwidth"], marker="o", linewidth=LINE_WIDTH, capsize=CAPSIZE)
    axes[0].set_xlabel("Driver phase jitter", fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].set_ylabel(r"Mean synchrony change, $\Delta \bar{R}$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].set_title("A. Phase-jitter scan", loc="left", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
    style_axes(axes[0])
    axes[1].errorbar(forcing_summary["forcing_strength_cycles_per_s"], forcing_summary["delta_mean_R_mean"], yerr=forcing_summary["delta_mean_R_ci95_halfwidth"], marker="o", linewidth=LINE_WIDTH, capsize=CAPSIZE, color="#54A24B")
    axes[1].set_xlabel("Forcing strength (cycles/s)", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_ylabel(r"Mean synchrony change, $\Delta \bar{R}$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_title("B. Forcing-strength scan", loc="left", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
    style_axes(axes[1])
    fig.suptitle("Coherence and forcing-strength dependence", fontsize=TITLE_FONTSIZE, fontweight="bold")
    fig.savefig(figure_dir / "Fig3_jitter_and_forcing_strength_scan.png", dpi=FIGURE_DPI)
    fig.savefig(figure_dir / "Fig3_jitter_and_forcing_strength_scan.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(PANEL_WIDTH_IN * 2, PANEL_HEIGHT_IN), constrained_layout=True)
    for condition, part in freq_summary.groupby("condition"):
        axes[0].errorbar(
            part["natural_frequency_sd_hz"],
            part["delta_mean_R_mean"],
            yerr=part["delta_mean_R_ci95_halfwidth"],
            marker="o",
            linewidth=LINE_WIDTH,
            capsize=2,
            label=CONDITION_LABELS.get(condition, condition),
        )
    axes[0].set_xlabel("Natural-frequency SD (Hz)", fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].set_ylabel(r"Mean synchrony change, $\Delta \bar{R}$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].set_title("A. Natural-frequency dispersion scan", loc="left", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
    axes[0].legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    style_axes(axes[0])
    dt_plot = dt_summary.set_index("metric").loc[["coarse_delta_mean_R", "fine_delta_mean_R"]]
    labels = [f"dt = {value:.6g} s" for value in dt_plot["dt_s"]]
    axes[1].bar(np.arange(len(labels)), dt_plot["mean"], color="#B279A2")
    axes[1].errorbar(np.arange(len(labels)), dt_plot["mean"], yerr=dt_plot["ci95_halfwidth"], fmt="none", color="black", capsize=CAPSIZE)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel(r"Mean synchrony change, $\Delta \bar{R}$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_title("B. Paired time-step convergence", loc="left", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
    style_axes(axes[1])
    fig.suptitle("Numerical and frequency-dispersion robustness", fontsize=TITLE_FONTSIZE, fontweight="bold")
    fig.savefig(figure_dir / "FigS1_numerical_frequency_dispersion.png", dpi=FIGURE_DPI)
    fig.savefig(figure_dir / "FigS1_numerical_frequency_dispersion.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(PANEL_WIDTH_IN, PANEL_HEIGHT_IN), constrained_layout=True)
    stim_plot = stimulus_locking_summary.set_index("condition").loc[STIMULUS_CONDITIONS].reset_index()
    x = np.arange(len(stim_plot))
    labels = [CONDITION_LABELS[c] for c in stim_plot["condition"]]
    ax.bar(x, stim_plot[f"{STIMULUS_PLV_COL}_mean"], color="#4C78A8")
    ax.errorbar(x, stim_plot[f"{STIMULUS_PLV_COL}_mean"], yerr=stim_plot[f"{STIMULUS_PLV_COL}_ci95_halfwidth"], fmt="none", color="black", capsize=CAPSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(r"Stimulus-locking PLV", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Stimulus-phase locking across forcing conditions", fontsize=TITLE_FONTSIZE, fontweight="bold")
    style_axes(ax)
    fig.savefig(figure_dir / "FigS2_stimulus_phase_locking.png", dpi=FIGURE_DPI)
    fig.savefig(figure_dir / "FigS2_stimulus_phase_locking.svg")
    plt.close(fig)


# =============================================================================
# 14. INTERNAL VALIDATION AND SELF-CHECKS
# =============================================================================


def assert_summary_n_blocks(summary: pd.DataFrame, expected: int, label: str) -> None:
    if "n_blocks" not in summary.columns:
        raise AssertionError(f"{label} summary does not contain n_blocks")
    bad = summary[summary["n_blocks"] != expected]
    if not bad.empty:
        raise AssertionError(f"{label} summary has unexpected n_blocks; expected {expected}")


def assert_no_nan_in_summary_means(summary: pd.DataFrame, label: str) -> None:
    mean_columns = [col for col in summary.columns if col == "mean" or col.endswith("_mean")]
    if mean_columns and summary[mean_columns].isna().any().any():
        raise AssertionError(f"NaN detected in {label} summary means")


def run_small_parallel_check(config: Config) -> None:
    small = replace(config, n_oscillators=8, duration_s=0.04, dt=0.005, burn_in_s=0.01, n_replicates=2, scan_replicates=2, robustness_replicates=2)
    jobs = [(small, NETWORKS[0].name, 0), (small, NETWORKS[1].name, 1)]
    one = pd.DataFrame(run_blocks(replace(small, workers=1), jobs, core_block_worker, allow_fallback=False)).sort_values(["network", "replicate", "condition"]).reset_index(drop=True)
    with cf.ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as executor:
        two = pd.DataFrame(run_blocks(replace(small, workers=2), jobs, core_block_worker, executor=executor, allow_fallback=False)).sort_values(["network", "replicate", "condition"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(one, two, check_dtype=False, atol=0.0, rtol=0.0)


def run_self_checks(config: Config | None = None) -> None:
    config = Config() if config is None else config
    validate_config(config)
    test_config = replace(config, n_oscillators=max(2, min(config.n_oscillators, 8)), duration_s=0.04, dt=0.005, burn_in_s=0.01, n_replicates=max(2, config.n_replicates), scan_replicates=max(2, config.scan_replicates), robustness_replicates=max(2, config.robustness_replicates), workers=1)
    network = NETWORKS[0]
    trial = make_trial(test_config, network, 0)
    constant_freq = np.full(5, network.frequency_hz)
    constant_phase = phase_from_instant_frequency(constant_freq, test_config.dt)
    assert np.allclose(constant_phase, 2.0 * np.pi * network.frequency_hz * np.arange(5) * test_config.dt)
    no_phase, _, eff, det = forcing_phase(test_config, network, trial, "NO_STIMULUS")
    assert eff == 0.0 and det == 0.0 and np.allclose(no_phase, 0.0)
    for condition in ["CLEAN_MATCHED", "LOW_JITTER_MATCHED", "HIGH_JITTER_MATCHED", "SHARED_IRREGULAR_FORCING", "INDEPENDENT_IRREGULAR_FORCING"]:
        phase, matrix, _, _ = forcing_phase(test_config, network, trial, condition)
        assert np.allclose(phase[0], 0.0) if matrix else math.isclose(float(phase[0]), 0.0, abs_tol=1e-12)
    phase_005, _, _, _ = forcing_phase(test_config, network, trial, "LOW_JITTER_MATCHED", jitter_level=0.05)
    phase_075, _, _, _ = forcing_phase(test_config, network, trial, "LOW_JITTER_MATCHED", jitter_level=0.75)
    assert not np.allclose(phase_005, phase_075)
    assert np.allclose((0.05 * network.frequency_hz * trial.unit_frequency_path) / (0.05 * network.frequency_hz), trial.unit_frequency_path)
    assert np.allclose((0.75 * network.frequency_hz * trial.unit_frequency_path) / (0.75 * network.frequency_hz), trial.unit_frequency_path)
    assert NEAR_DETUNING_HZ == 2.0
    assert FAR_DETUNING_HZ == 6.0
    paths = []
    for level in JITTER_LEVELS:
        frequency_deviation = level * network.frequency_hz * trial.unit_frequency_path
        scaled_back = frequency_deviation / (level * network.frequency_hz) if level > 0 else trial.unit_frequency_path
        paths.append(scaled_back)
    for path in paths[1:]:
        assert np.allclose(path, paths[0])
    coarse, fine = make_paired_trials(test_config, network, 0, test_config.dt, test_config.dt / 2.0)
    assert np.allclose(coarse.dW, fine.dW.reshape(coarse.dW.shape[0], 2, test_config.n_oscillators).sum(axis=1))
    assert test_config.burn_in_s < test_config.duration_s
    forbidden_columns = {"R" + "_auc", "delta_R" + "_auc", "time_to" + "_lock_s"}
    sample = simulate_condition(test_config, network, trial, "CLEAN_MATCHED", 0)
    assert forbidden_columns.isdisjoint(sample.keys())
    assert 0.0 <= float(sample[STIMULUS_PLV_COL]) <= 1.0
    no_stim_sample = simulate_condition(test_config, network, trial, "NO_STIMULUS", 0)
    assert math.isnan(float(no_stim_sample[STIMULUS_PLV_COL]))
    trial_sd_030 = make_trial(replace(test_config, natural_frequency_sd_hz=0.30), network, 1)
    trial_sd_150 = make_trial(replace(test_config, natural_frequency_sd_hz=1.50), network, 1)
    assert np.allclose(trial_sd_030.theta0, trial_sd_150.theta0)
    assert np.allclose(trial_sd_030.frequency_z, trial_sd_150.frequency_z)
    assert np.allclose(trial_sd_030.dW, trial_sd_150.dW)
    expected_omega_difference = 2.0 * np.pi * (1.50 - 0.30) * trial_sd_030.frequency_z
    assert np.allclose(trial_sd_150.omega - trial_sd_030.omega, expected_omega_difference)
    master_config = replace(test_config, n_oscillators=192, duration_s=0.08)
    master_trial = make_trial(master_config, network, 2, need_jitter=True, need_irregular=True, need_independent_irregular=True)
    trial_n96 = slice_trial_oscillators(master_trial, 96)
    assert np.allclose(trial_n96.theta0, master_trial.theta0[:96])
    assert np.allclose(trial_n96.frequency_z, master_trial.frequency_z[:96])
    assert np.allclose(trial_n96.dW, master_trial.dW[:, :96])
    assert np.allclose(trial_n96.susceptibility, master_trial.susceptibility[:96])
    assert np.allclose(trial_n96.unit_frequency_path, master_trial.unit_frequency_path)
    assert np.allclose(trial_n96.shared_irregular_phase, master_trial.shared_irregular_phase)
    assert np.allclose(trial_n96.independent_irregular_phase, master_trial.independent_irregular_phase[:, :96])
    run_small_parallel_check(test_config)



def tagged_frame(df: pd.DataFrame, analysis: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "analysis", analysis)
    return out


def write_compact_results(output: Path, tables: dict[str, pd.DataFrame]) -> None:
    tables["core_metrics"].to_csv(output / "core_metrics.csv", index=False)
    main_results = pd.concat([
        tagged_frame(tables["core_paired_deltas"], "condition_comparison"),
        tagged_frame(tables["coupling_forcing_factorial_scan"], "coupling_forcing_factorial"),
        tagged_frame(tables["coupling_strength_scan"], "coupling_strength_scan"),
        tagged_frame(tables["jitter_scan"], "jitter_scan"),
        tagged_frame(tables["forcing_strength_scan"], "forcing_strength_scan"),
    ], ignore_index=True, sort=False)
    main_summary = pd.concat([
        tagged_frame(tables["core_condition_summary"], "condition_comparison"),
        tagged_frame(tables["coupling_forcing_factorial_summary"], "coupling_forcing_factorial"),
        tagged_frame(tables["coupling_strength_summary"], "coupling_strength_scan"),
        tagged_frame(tables["jitter_summary"], "jitter_scan"),
        tagged_frame(tables["forcing_strength_summary"], "forcing_strength_scan"),
    ], ignore_index=True, sort=False)
    supplementary_results = pd.concat([
        tagged_frame(tables["frequency_dispersion_scan"], "frequency_dispersion_scan"),
        tagged_frame(tables["paired_time_step_convergence"], "paired_time_step_convergence"),
        tagged_frame(tables["additional_robustness"], "duration_and_population_robustness"),
    ], ignore_index=True, sort=False)
    supplementary_summary = pd.concat([
        tagged_frame(tables["frequency_dispersion_summary"], "frequency_dispersion_scan"),
        tagged_frame(tables["paired_time_step_convergence_summary"], "paired_time_step_convergence"),
        tagged_frame(tables["additional_robustness_summary"], "duration_and_population_robustness"),
    ], ignore_index=True, sort=False)
    by_frequency_summary = pd.concat([
        tagged_frame(tables["core_condition_by_frequency_summary"], "condition_comparison"),
        tagged_frame(tables["coupling_forcing_factorial_by_frequency_summary"], "coupling_forcing_factorial"),
        tagged_frame(tables["coupling_strength_by_frequency_summary"], "coupling_strength_scan"),
        tagged_frame(tables["jitter_by_frequency_summary"], "jitter_scan"),
        tagged_frame(tables["forcing_strength_by_frequency_summary"], "forcing_strength_scan"),
        tagged_frame(tables["frequency_dispersion_by_frequency_summary"], "frequency_dispersion_scan"),
    ], ignore_index=True, sort=False)
    main_results.to_csv(output / "main_results.csv", index=False)
    main_summary.to_csv(output / "main_summary.csv", index=False)
    tables["paired_condition_contrasts"].to_csv(output / "paired_contrasts.csv", index=False)
    supplementary_results.to_csv(output / "supplementary_results.csv", index=False)
    supplementary_summary.to_csv(output / "supplementary_summary.csv", index=False)
    by_frequency_summary.to_csv(output / "by_frequency_summary.csv", index=False)
    tables["stimulus_locking_summary"].to_csv(output / "stimulus_locking_summary.csv", index=False)
    tables["stimulus_locking_by_frequency_summary"].to_csv(output / "stimulus_locking_by_frequency_summary.csv", index=False)

# =============================================================================
# 15. ANALYSIS PIPELINE AND OUTPUT EXPORT
# =============================================================================


def run_pipeline(config: Config, output: Path, progress: Callable[[str], None] | None, executor: cf.Executor | None) -> dict[str, pd.DataFrame]:
    write_definitions(config, output)
    write_environment_manifest(config, output)

    if progress:
        progress("Running core experiment")
    core_metrics, core_delta, core_summary, core_by_frequency, stimulus_locking_summary, stimulus_locking_by_frequency, contrasts = run_core_experiment(config, progress, executor=executor)
    assert_summary_n_blocks(core_summary, config.n_replicates, "core condition")
    assert_summary_n_blocks(core_by_frequency, config.n_replicates, "core condition by frequency")
    assert_summary_n_blocks(stimulus_locking_summary, config.n_replicates, "stimulus locking")
    assert_summary_n_blocks(stimulus_locking_by_frequency, config.n_replicates, "stimulus locking by frequency")
    assert_no_nan_in_summary_means(core_summary, "core condition")
    assert_no_nan_in_summary_means(stimulus_locking_summary, "stimulus locking")

    if progress:
        progress("Running coupling × forcing multiplier scan")
    factorial, factorial_summary, factorial_by_frequency = run_coupling_forcing_factorial_scan(config, progress, executor=executor)
    assert_summary_n_blocks(factorial_summary, config.scan_replicates, "coupling-forcing factorial")
    assert_summary_n_blocks(factorial_by_frequency, config.scan_replicates, "coupling-forcing factorial by frequency")
    assert_no_nan_in_summary_means(factorial_summary, "coupling-forcing factorial")

    if progress:
        progress("Running coupling-strength scan")
    coupling, coupling_summary, coupling_by_frequency = run_simple_scan(config, "coupling_strength", K_VALUES, progress, executor=executor)
    assert_summary_n_blocks(coupling_summary, config.scan_replicates, "coupling strength")
    assert_summary_n_blocks(coupling_by_frequency, config.scan_replicates, "coupling strength by frequency")
    assert_no_nan_in_summary_means(coupling_summary, "coupling strength")

    if progress:
        progress("Running jitter scan")
    jitter, jitter_summary, jitter_by_frequency = run_simple_scan(config, "jitter", JITTER_LEVELS, progress, executor=executor)
    assert_summary_n_blocks(jitter_summary, config.scan_replicates, "jitter")
    assert_summary_n_blocks(jitter_by_frequency, config.scan_replicates, "jitter by frequency")
    assert_no_nan_in_summary_means(jitter_summary, "jitter")

    if progress:
        progress("Running forcing-strength scan")
    forcing, forcing_summary, forcing_by_frequency = run_simple_scan(config, "forcing_strength", FORCING_VALUES, progress, executor=executor)
    assert_summary_n_blocks(forcing_summary, config.scan_replicates, "forcing strength")
    assert_summary_n_blocks(forcing_by_frequency, config.scan_replicates, "forcing strength by frequency")
    assert_no_nan_in_summary_means(forcing_summary, "forcing strength")

    if progress:
        progress("Running frequency-dispersion scan")
    freq, freq_summary, freq_by_frequency = run_frequency_dispersion_scan(config, progress, executor=executor)
    assert_summary_n_blocks(freq_summary, config.robustness_replicates, "frequency dispersion")
    assert_summary_n_blocks(freq_by_frequency, config.robustness_replicates, "frequency dispersion by frequency")
    assert_no_nan_in_summary_means(freq_summary, "frequency dispersion")

    if progress:
        progress("Running paired time-step convergence")
    dt_df, dt_summary = run_paired_time_step_convergence(config, progress, executor=executor)
    assert_summary_n_blocks(dt_summary, config.robustness_replicates, "paired time step")
    assert_no_nan_in_summary_means(dt_summary, "paired time step")

    if progress:
        progress("Running additional robustness checks")
    add, add_summary = run_additional_robustness(config, progress, executor=executor)
    assert_summary_n_blocks(add_summary, config.robustness_replicates, "additional robustness")
    assert_no_nan_in_summary_means(add_summary, "additional robustness")

    tables = {
        "core_metrics": core_metrics,
        "core_paired_deltas": core_delta,
        "core_condition_summary": core_summary,
        "core_condition_by_frequency_summary": core_by_frequency,
        "stimulus_locking_summary": stimulus_locking_summary,
        "stimulus_locking_by_frequency_summary": stimulus_locking_by_frequency,
        "paired_condition_contrasts": contrasts,
        "coupling_forcing_factorial_scan": factorial,
        "coupling_forcing_factorial_summary": factorial_summary,
        "coupling_forcing_factorial_by_frequency_summary": factorial_by_frequency,
        "coupling_strength_scan": coupling,
        "coupling_strength_summary": coupling_summary,
        "coupling_strength_by_frequency_summary": coupling_by_frequency,
        "jitter_scan": jitter,
        "jitter_summary": jitter_summary,
        "jitter_by_frequency_summary": jitter_by_frequency,
        "forcing_strength_scan": forcing,
        "forcing_strength_summary": forcing_summary,
        "forcing_strength_by_frequency_summary": forcing_by_frequency,
        "frequency_dispersion_scan": freq,
        "frequency_dispersion_summary": freq_summary,
        "frequency_dispersion_by_frequency_summary": freq_by_frequency,
        "paired_time_step_convergence": dt_df,
        "paired_time_step_convergence_summary": dt_summary,
        "additional_robustness": add,
        "additional_robustness_summary": add_summary,
    }

    write_compact_results(output, tables)
    make_figures(output, core_summary, factorial_summary, coupling_summary, jitter_summary, forcing_summary, freq_summary, dt_summary, stimulus_locking_summary)
    missing = [name for name in CSV_FILES + FIGURE_FILES if not (output / name).exists()]
    if missing:
        raise RuntimeError(f"Missing expected output files: {missing}")
    return tables


# Creates one multiprocessing pool and executes the complete analysis pipeline.
def run_all(config: Config, output: Path, progress: Callable[[str], None] | None = None) -> dict[str, pd.DataFrame]:
    validate_config(config)
    output = make_output_dir(output)
    crash_log = output / "crash_log.txt"
    progress = safe_progress_callback(progress)
    if crash_log.exists():
        try:
            crash_log.unlink()
        except OSError:
            pass
    try:
        if config.workers > 1:
            with cf.ProcessPoolExecutor(max_workers=config.workers, mp_context=mp.get_context("spawn")) as executor:
                return run_pipeline(config, output, progress, executor)
        return run_pipeline(config, output, progress, None)
    except Exception:
        crash_log.write_text(traceback.format_exc(), encoding="utf-8")
        raise


# =============================================================================
# 16. COMMAND-LINE INTERFACE
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="General frequency-matched phase-forcing simulations for Kuramoto oscillators.")
    parser.add_argument("--no-gui", action="store_true", help="Run from the command line without opening the GUI.")
    parser.add_argument("--self-check", action="store_true", help="Run validation and multiprocessing self-checks, then exit.")
    parser.add_argument("--example-small-run", action="store_true", help="Run a very small end-to-end example for quick reproducibility checks.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory.")
    parser.add_argument("--replicates", type=int, default=Config.n_replicates)
    parser.add_argument("--scan-replicates", type=int, default=Config.scan_replicates)
    parser.add_argument("--robustness-replicates", type=int, default=Config.robustness_replicates)
    parser.add_argument("--n-oscillators", type=int, default=Config.n_oscillators)
    parser.add_argument("--duration", type=float, default=Config.duration_s)
    parser.add_argument("--dt", type=float, default=Config.dt)
    parser.add_argument("--burn-in", type=float, default=Config.burn_in_s)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--individual-forcing-log-sd", type=float, default=Config.individual_forcing_log_sd)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        n_oscillators=args.n_oscillators,
        duration_s=args.duration,
        dt=args.dt,
        burn_in_s=args.burn_in,
        n_replicates=args.replicates,
        scan_replicates=args.scan_replicates,
        robustness_replicates=args.robustness_replicates,
        workers=args.workers,
        seed=args.seed,
        individual_forcing_log_sd=args.individual_forcing_log_sd,
    )


# =============================================================================
# 17. GRAPHICAL USER INTERFACE
# =============================================================================


def launch_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise RuntimeError(
            "GUI requires tkinter. In a headless environment, use --no-gui."
        ) from exc

    root = tk.Tk()
    root.title("Phase-forcing simulation")
    fields = {
        "Output directory": tk.StringVar(value=str(default_output_dir())),
        "Oscillators": tk.StringVar(value=str(Config.n_oscillators)),
        "Duration (s)": tk.StringVar(value=str(Config.duration_s)),
        "Time step dt": tk.StringVar(value=str(Config.dt)),
        "Burn-in (s)": tk.StringVar(value=str(Config.burn_in_s)),
        "Replicates": tk.StringVar(value=str(Config.n_replicates)),
        "Scan replicates": tk.StringVar(value=str(Config.scan_replicates)),
        "Robustness replicates": tk.StringVar(value=str(Config.robustness_replicates)),
        "CPU workers": tk.StringVar(value=str(DEFAULT_WORKERS)),
        "Seed": tk.StringVar(value=str(Config.seed)),
    }
    for row, (label, var) in enumerate(fields.items()):
        ttk.Label(root, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(root, textvariable=var, width=40).grid(row=row, column=1, sticky="ew", padx=6, pady=4)
    def browse():
        directory = filedialog.askdirectory(initialdir=fields["Output directory"].get())
        if directory:
            fields["Output directory"].set(directory)
    ttk.Button(root, text="Browse", command=browse).grid(row=0, column=2, padx=6, pady=4)
    log = tk.Text(root, width=80, height=14)
    log.grid(row=len(fields), column=0, columnspan=3, padx=6, pady=6)
    q: queue.Queue[str] = queue.Queue()
    active_outputs: set[Path] = set()
    run_button: ttk.Button | None = None
    def progress(msg: str) -> None:
        q.put(msg)
    def poll() -> None:
        while not q.empty():
            log.insert("end", q.get() + "\n")
            log.see("end")
        root.after(200, poll)
    def start() -> None:
        try:
            cfg = Config(
                n_oscillators=int(fields["Oscillators"].get()),
                duration_s=float(fields["Duration (s)"].get()),
                dt=float(fields["Time step dt"].get()),
                burn_in_s=float(fields["Burn-in (s)"].get()),
                n_replicates=int(fields["Replicates"].get()),
                scan_replicates=int(fields["Scan replicates"].get()),
                robustness_replicates=int(fields["Robustness replicates"].get()),
                workers=int(fields["CPU workers"].get()),
                seed=int(fields["Seed"].get()),
            )
            validate_config(cfg)
            out = Path(fields["Output directory"].get()).expanduser().resolve()
            if out in active_outputs:
                messagebox.showerror("Output directory in use", "An analysis is already running for this output directory.")
                return
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return
        active_outputs.add(out)
        if run_button is not None:
            run_button.configure(state="disabled")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--no-gui",
            "--output", str(out),
            "--replicates", str(cfg.n_replicates),
            "--scan-replicates", str(cfg.scan_replicates),
            "--robustness-replicates", str(cfg.robustness_replicates),
            "--n-oscillators", str(cfg.n_oscillators),
            "--duration", str(cfg.duration_s),
            "--dt", str(cfg.dt),
            "--burn-in", str(cfg.burn_in_s),
            "--workers", str(cfg.workers),
            "--seed", str(cfg.seed),
            "--individual-forcing-log-sd", str(cfg.individual_forcing_log_sd),
        ]
        def work():
            return_code = -1
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    progress(line.rstrip())
                return_code = process.wait()
                progress(f"Process finished with exit code {return_code}")
                if return_code != 0:
                    root.after(0, lambda: messagebox.showerror("Analysis failed", f"Process exited with code {return_code}"))
            except Exception:
                progress(traceback.format_exc())
                root.after(0, lambda: messagebox.showerror("Analysis failed", "Could not launch or monitor the subprocess."))
            finally:
                active_outputs.discard(out)
                root.after(0, lambda: run_button.configure(state="normal") if run_button is not None else None)
        threading.Thread(target=work, daemon=True).start()
    run_button = ttk.Button(root, text="Run", command=start)
    run_button.grid(row=len(fields)+1, column=0, columnspan=3, pady=8)
    poll()
    root.mainloop()


# =============================================================================
# 18. APPLICATION ENTRY POINT
# =============================================================================


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    if args.self_check:
        run_self_checks(config)
        console_progress("Self-checks passed")
        return
    if args.example_small_run:
        config = replace(config, n_oscillators=8, duration_s=0.04, dt=0.005, burn_in_s=0.01, n_replicates=2, scan_replicates=2, robustness_replicates=2, workers=1)
        out = make_output_dir(args.output)
        run_all(config, out, console_progress)
        console_progress(f"Small example outputs written to {out}")
        return
    if args.no_gui:
        out = make_output_dir(args.output)
        run_all(config, out, console_progress)
        console_progress(f"Outputs written to {out}")
    else:
        launch_gui()


if __name__ == "__main__":
    mp.freeze_support()
    main()
