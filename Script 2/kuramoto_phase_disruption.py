#!/usr/bin/env python3
from __future__ import annotations

# =============================================================================
# 1. IMPORTS
# =============================================================================

import argparse
import csv
import hashlib
import math
import os
import platform
import random
import statistics
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
try:
    import scipy
    from scipy.stats import t as student_t
except Exception as exc:  # pragma: no cover - import-time dependency guard
    raise RuntimeError(
        "SciPy is required for exact t-based confidence intervals "
        "and for submission-ready reproducibility."
    ) from exc
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

# =============================================================================
# 2. MODEL CONSTANTS AND OUTPUT DEFINITIONS
# =============================================================================

SCRIPT_VERSION = "2.0.3-closed-loop-delay-fixed"

FREQUENCIES = (7.83, 10.00, 14.30, 20.80, 27.30, 33.80)
PHASE_OFFSETS_DEG = tuple(range(0, 361, 30))
K_VALUES = (0.35, 0.50, 0.65, 0.80, 0.95, 1.10, 1.25, 1.40, 1.55)
MAINTENANCE_CONDITIONS = (
    "no_driver",
    "matched_driver",
    "near_detuned_driver",
    "far_detuned_driver",
    "jittered_driver",
)
SUPPLEMENT_DELAYS = (0.0, 0.0625, 0.125, 0.25, 0.375, 0.5)
SUPPLEMENT_JITTER = (0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75)
DEFAULT_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))

CSV_EXPORTS = (
    "configuration_parameters.csv",
    "environment_manifest.csv",
    "main_results.csv",
    "main_summary.csv",
    "paired_contrasts.csv",
    "supplementary_results.csv",
    "supplementary_summary.csv",
)
FIGURE_EXPORTS = (
    "figures/Fig4_external_driver_phase_disruption.png",
    "figures/Fig4_external_driver_phase_disruption.svg",
    "figures/Fig5_internal_closed_loop_phase_disruption.png",
    "figures/Fig5_internal_closed_loop_phase_disruption.svg",
    "figures/Fig6_K_dependent_phase_disruption.png",
    "figures/Fig6_K_dependent_phase_disruption.svg",
    "figures/Fig7_driver_maintenance.png",
    "figures/Fig7_driver_maintenance.svg",
    "figures/FigS3_delay_scan.png",
    "figures/FigS3_delay_scan.svg",
    "figures/FigS4_jitter_scan.png",
    "figures/FigS4_jitter_scan.svg",
)

FIGURE_DPI = 300
PANEL_WIDTH_IN = 6.0
PANEL_HEIGHT_IN = 5.0
TITLE_FONTSIZE = 12
PANEL_TITLE_FONTSIZE = 10
AXIS_LABEL_FONTSIZE = 11
TICK_LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 9
LINE_WIDTH = 1.8
CAPSIZE = 3.0
MATCHED_COLOR = "#1f77b4"
CONTROL_COLOR = "#b8b8b8"
ERROR_COLOR = "#222222"


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
    pre_s: float = 12.0
    intervention_s: float = 12.0
    recovery_s: float = 12.0
    maintenance_induction_s: float = 12.0
    maintenance_test_s: float = 12.0
    maintenance_recovery_s: float = 12.0
    natural_frequency_sd_hz: float = 0.75
    phase_diffusion_cycles_per_sqrt_s: float = 0.035
    external_k_cycles_per_s: float = 0.72
    internal_k_cycles_per_s: float = 0.72
    external_driver_gain_cycles_per_s: float = 0.12
    control_gain_cycles_per_s: float = 0.12
    maintenance_k_induction_cycles_per_s: float = 1.10
    maintenance_k_test_cycles_per_s: float = 0.72
    maintenance_driver_gain_cycles_per_s: float = 0.12
    lock_threshold: float = 0.55
    phase_validity_r1_threshold: float = 0.20
    minimum_sustained_lock_s: float = 0.10
    include_supplement: bool = True
    quick: bool = False
    single_frequency: bool = False


@dataclass(frozen=True)
class Epoch:
    name: str
    start_step: int
    end_step: int
    k_cycles_per_s: float
    driver_enabled: bool
    control_enabled: bool


@dataclass(frozen=True)
class DriverSpec:
    enabled: bool
    gain_cycles_per_s: float
    frequency_hz: float
    phase_offset_rad: float = 0.0
    jitter_level: float = 0.0


@dataclass(frozen=True)
class ControlSpec:
    control_type: str
    gain_cycles_per_s: float
    frequency_hz: float | None
    phase_offset_rad: float
    delay_cycles: float = 0.0
    jitter_level: float = 0.0


@dataclass
class TrialBase:
    theta0: list[float]
    frequency_z: list[float]
    dW: list[list[float]]
    unit_frequency_path: list[float]
    unit_phase_path: list[float]


# =============================================================================
# 4. REPRODUCIBILITY AND CONFIGURATION HELPERS
# =============================================================================

def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFF


def n_steps(seconds: float, dt: float) -> int:
    return max(1, int(round(seconds / dt)))


def frequencies(config: Config) -> tuple[float, ...]:
    return (10.0,) if config.single_frequency else FREQUENCIES


def quick_config(config: Config) -> Config:
    return replace(
        config,
        n_oscillators=min(config.n_oscillators, 32),
        n_replicates=min(config.n_replicates, 3),
        pre_s=0.30,
        intervention_s=0.30,
        recovery_s=0.30,
        maintenance_induction_s=0.30,
        maintenance_test_s=0.30,
        maintenance_recovery_s=0.30,
        quick=True,
    )


def validate_config(config: Config) -> None:
    if config.n_oscillators < 4:
        raise ValueError("n_oscillators must be >= 4")
    if config.n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")
    if config.dt <= 0:
        raise ValueError("dt must be > 0")
    for field in (
        "pre_s", "intervention_s", "recovery_s", "maintenance_induction_s",
        "maintenance_test_s", "maintenance_recovery_s", "natural_frequency_sd_hz",
        "phase_diffusion_cycles_per_sqrt_s", "external_k_cycles_per_s",
        "internal_k_cycles_per_s", "external_driver_gain_cycles_per_s",
        "control_gain_cycles_per_s", "maintenance_k_induction_cycles_per_s",
        "maintenance_k_test_cycles_per_s", "maintenance_driver_gain_cycles_per_s",
    ):
        if getattr(config, field) < 0:
            raise ValueError(f"{field} must be nonnegative")
    for field in ("lock_threshold", "phase_validity_r1_threshold"):
        value = getattr(config, field)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be in [0, 1]")
    if config.minimum_sustained_lock_s <= 0:
        raise ValueError("minimum_sustained_lock_s must be > 0")


# =============================================================================
# 5. TRIAL GENERATION AND PHASE-METRIC UTILITIES
# =============================================================================

def make_trial(config: Config, frequency_hz: float, replicate: int, total_steps: int) -> TrialBase:
    rng = random.Random(stable_seed(config.seed, "trial", f"{frequency_hz:.8g}", replicate))
    theta0 = [rng.random() * 2.0 * math.pi for _ in range(config.n_oscillators)]
    frequency_z = [rng.gauss(0.0, 1.0) for _ in range(config.n_oscillators)]
    sqrt_dt = math.sqrt(config.dt)
    dW = [[rng.gauss(0.0, sqrt_dt) for _ in range(config.n_oscillators)] for _ in range(total_steps)]
    unit_frequency_path = [rng.gauss(0.0, 1.0) for _ in range(total_steps)]
    unit_phase_path = [rng.gauss(0.0, 1.0) for _ in range(total_steps)]
    return TrialBase(theta0, frequency_z, dW, unit_frequency_path, unit_phase_path)


def circular_phase_difference(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def closed_loop_reference_phase(current_psi: float, psi_history: Sequence[float], delay_steps: int) -> float:
    """Return Psi(t - tau) for closed-loop control.

    The current population phase is computed before this helper is called, while
    psi_history contains only phases from previous numerical steps. Therefore a
    zero-delay controller must use current_psi directly; otherwise tau=0 would
    silently become a one-step lag. Positive delays are rounded to integer steps
    before this function is called.
    """
    if delay_steps <= 0:
        return current_psi
    if len(psi_history) >= delay_steps:
        return float(psi_history[-delay_steps])
    return float(psi_history[0]) if psi_history else current_psi


def order_parameter(theta: Sequence[float], harmonic: int = 1) -> tuple[float, float]:
    c = sum(math.cos(harmonic * x) for x in theta) / len(theta)
    s = sum(math.sin(harmonic * x) for x in theta) / len(theta)
    return math.hypot(c, s), math.atan2(s, c) / harmonic


def complex_mean_abs_angle(values: Iterable[complex]) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v.real) and math.isfinite(v.imag)]
    if not vals:
        return float("nan"), float("nan")
    z = sum(vals) / len(vals)
    return abs(z), math.atan2(z.imag, z.real)


def longest_true_run(mask: Sequence[bool], dt: float) -> float:
    best = run = 0
    for value in mask:
        if value:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best * dt


def first_sustained_true_time(mask: Sequence[bool], dt: float, minimum_duration_s: float) -> float:
    required = max(1, int(math.ceil(minimum_duration_s / dt)))
    run = 0
    for index, value in enumerate(mask):
        if value:
            run += 1
            if run >= required:
                return (index - required + 1) * dt
        else:
            run = 0
    return float("nan")


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def sd(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0 if len(vals) == 1 else float("nan")


def ci_stats(values: Iterable[float]) -> dict[str, float | int]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return {
            "mean": float("nan"),
            "sd": float("nan"),
            "sem": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "ci95_halfwidth": float("nan"),
            "n": 0,
        }
    m = sum(vals) / len(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    sem = s / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    half = float(student_t.ppf(0.975, len(vals) - 1) * sem) if len(vals) > 1 else 0.0
    return {"mean": m, "sd": s, "sem": sem, "ci95_low": m - half, "ci95_high": m + half, "ci95_halfwidth": half, "n": len(vals)}


# =============================================================================
# 6. EPOCH METRICS AND KURAMOTO SCHEDULE SIMULATION
# =============================================================================

def epoch_metrics(config: Config, name: str, start: int, end: int, trace: dict[str, list]) -> dict[str, float]:
    r1 = trace["R1"][start:end]
    r2 = trace["R2"][start:end]
    r4 = trace["R4"][start:end]
    above = [x >= config.lock_threshold for x in r1]
    below = [not x for x in above]
    lock_strength, lock_lag = complex_mean_abs_angle(trace["driver_lock"][start:end])
    return {
        f"{name}_mean_R1": mean(r1),
        f"{name}_mean_R2": mean(r2),
        f"{name}_mean_R4": mean(r4),
        f"{name}_lock_fraction": sum(above) / len(above) if above else float("nan"),
        f"{name}_longest_lock_run_s": longest_true_run(above, config.dt),
        f"{name}_time_to_sustained_lock_s": first_sustained_true_time(above, config.dt, config.minimum_sustained_lock_s),
        f"{name}_time_to_sustained_drop_below_lock_s": first_sustained_true_time(below, config.dt, config.minimum_sustained_lock_s),
        f"{name}_driver_locking_plv": lock_strength,
        f"{name}_driver_lock_strength": lock_strength,  # backward-compatible alias for earlier outputs
        f"{name}_driver_lock_lag_rad": lock_lag,
    }


def simulate_schedule(
    config: Config,
    frequency_hz: float,
    trial: TrialBase,
    epochs: Sequence[Epoch],
    driver: DriverSpec,
    control: ControlSpec,
    return_trace: bool = False,
) -> dict[str, object]:
    total_steps = epochs[-1].end_step
    theta = list(trial.theta0)
    natural = [frequency_hz + config.natural_frequency_sd_hz * z for z in trial.frequency_z]
    trace: dict[str, list] = {"time_s": [], "R1": [], "R2": [], "R4": [], "psi": [], "driver_lock": [], "driver_phase_error": [], "control_phase_error": []}
    psi_history: list[float] = []
    epoch_index = 0
    for step in range(total_steps):
        while epoch_index + 1 < len(epochs) and step >= epochs[epoch_index].end_step:
            epoch_index += 1
        epoch = epochs[epoch_index]
        t = step * config.dt
        r1, psi = order_parameter(theta, 1)
        r2, _ = order_parameter(theta, 2)
        r4, _ = order_parameter(theta, 4)
        z1 = complex(r1 * math.cos(psi), r1 * math.sin(psi))
        driver_phase = 2.0 * math.pi * driver.frequency_hz * t + driver.phase_offset_rad
        driver_gain = driver.gain_cycles_per_s if (driver.enabled and epoch.driver_enabled) else 0.0
        if driver.jitter_level:
            driver_phase += driver.jitter_level * 0.40 * trial.unit_phase_path[step]
            driver_phase += 2.0 * math.pi * driver.jitter_level * 0.04 * trial.unit_frequency_path[step] * t
        control_gain = control.gain_cycles_per_s if epoch.control_enabled else 0.0
        if control_gain and control.control_type == "open_loop":
            cf = control.frequency_hz if control.frequency_hz is not None else frequency_hz
            control_phase = 2.0 * math.pi * cf * t + control.phase_offset_rad
            if control.jitter_level:
                control_phase += control.jitter_level * 0.40 * trial.unit_phase_path[step]
                control_phase += 2.0 * math.pi * control.jitter_level * 0.04 * trial.unit_frequency_path[step] * t
        elif control_gain and control.control_type == "closed_loop":
            delay_steps = max(0, int(round((control.delay_cycles / max(frequency_hz, 1e-9)) / config.dt)))
            delayed_psi = closed_loop_reference_phase(psi, psi_history, delay_steps)
            control_phase = delayed_psi + control.phase_offset_rad
        else:
            control_phase = float("nan")
        trace["time_s"].append(t)
        trace["R1"].append(r1)
        trace["R2"].append(r2)
        trace["R4"].append(r4)
        trace["psi"].append(psi)
        # Rotating the population order parameter into the driver phase frame gives
        # z1 * exp(-i*driver_phase). The epoch-wise magnitude of its time average
        # is mathematically equivalent to a pooled oscillator-driver PLV over all
        # oscillators and all samples in the analysed epoch for a shared driver.
        trace["driver_lock"].append(z1 * complex(math.cos(-driver_phase), math.sin(-driver_phase)) if driver_gain else complex(float("nan"), float("nan")))
        if r1 >= config.phase_validity_r1_threshold and driver_gain:
            trace["driver_phase_error"].append(circular_phase_difference(psi, driver_phase))
        else:
            trace["driver_phase_error"].append(float("nan"))
        if r1 >= config.phase_validity_r1_threshold and control_gain and math.isfinite(control_phase):
            trace["control_phase_error"].append(circular_phase_difference(psi, control_phase))
        else:
            trace["control_phase_error"].append(float("nan"))
        psi_history.append(psi)
        for i, x in enumerate(theta):
            internal = epoch.k_cycles_per_s * r1 * math.sin(psi - x)
            ext = driver_gain * math.sin(driver_phase - x)
            ctrl = control_gain * math.sin(control_phase - x) if math.isfinite(control_phase) else 0.0
            cycles_per_s = natural[i] + internal + ext + ctrl
            theta_i = x + 2.0 * math.pi * cycles_per_s * config.dt + 2.0 * math.pi * config.phase_diffusion_cycles_per_sqrt_s * trial.dW[step][i]
            theta_i = math.fmod(theta_i, 2.0 * math.pi)
            if theta_i < 0.0:
                theta_i += 2.0 * math.pi
            theta[i] = theta_i
    row: dict[str, object] = {}
    for epoch in epochs:
        row.update(epoch_metrics(config, epoch.name, epoch.start_step, epoch.end_step, trace))
    if return_trace:
        row["trace"] = trace
    return row


def disruption_epochs(config: Config, paradigm: str, k_cycles: float) -> list[Epoch]:
    pre = n_steps(config.pre_s, config.dt)
    intervention = pre + n_steps(config.intervention_s, config.dt)
    total = intervention + n_steps(config.recovery_s, config.dt)
    driver_on = paradigm == "external_driver"
    return [
        Epoch("pre", 0, pre, k_cycles, driver_on, False),
        Epoch("intervention", pre, intervention, k_cycles, driver_on, True),
        Epoch("recovery", intervention, total, k_cycles, driver_on, False),
    ]


def maintenance_epochs(config: Config) -> list[Epoch]:
    induction = n_steps(config.maintenance_induction_s, config.dt)
    test = induction + n_steps(config.maintenance_test_s, config.dt)
    total = test + n_steps(config.maintenance_recovery_s, config.dt)
    return [
        Epoch("induction", 0, induction, config.maintenance_k_induction_cycles_per_s, False, False),
        Epoch("test", induction, test, config.maintenance_k_test_cycles_per_s, True, False),
        Epoch("post_driver_recovery", test, total, config.maintenance_k_test_cycles_per_s, False, False),
    ]


def paired_delta(control: dict[str, object], sham: dict[str, object], metrics: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in metrics:
        a = float(control.get(metric, float("nan")))
        b = float(sham.get(metric, float("nan")))
        out[f"delta_{metric}"] = a - b if math.isfinite(a) and math.isfinite(b) else float("nan")
    return out


# =============================================================================
# 7. ANALYSIS WORKERS
# =============================================================================

def phase_worker(job: tuple[Config, float, int]) -> list[dict[str, object]]:
    config, f0, replicate = job
    rows: list[dict[str, object]] = []
    steps = sum(n_steps(x, config.dt) for x in (config.pre_s, config.intervention_s, config.recovery_s))
    trial = make_trial(config, f0, replicate, steps)
    external_epochs = disruption_epochs(config, "external_driver", config.external_k_cycles_per_s)
    internal_epochs = disruption_epochs(config, "internal_sync", config.internal_k_cycles_per_s)
    driver = DriverSpec(True, config.external_driver_gain_cycles_per_s, f0)
    no_driver = DriverSpec(False, 0.0, f0)
    sham_open = ControlSpec("open_loop", 0.0, f0, 0.0)
    sham_closed = ControlSpec("closed_loop", 0.0, None, 0.0)
    external_sham = simulate_schedule(config, f0, trial, external_epochs, driver, sham_open)
    internal_sham = simulate_schedule(config, f0, trial, internal_epochs, no_driver, sham_closed)
    external_metrics = ("intervention_driver_locking_plv", "intervention_mean_R1", "intervention_mean_R2", "intervention_mean_R4")
    internal_metrics = ("intervention_mean_R1", "intervention_mean_R2", "intervention_mean_R4")
    for phase_deg in PHASE_OFFSETS_DEG:
        phase = math.radians(phase_deg)
        control = ControlSpec("open_loop", config.control_gain_cycles_per_s, f0, phase)
        result = simulate_schedule(config, f0, trial, external_epochs, driver, control)
        row = {"analysis": "external_driver_phase_disruption", "frequency_hz": f0, "replicate": replicate, "phase_offset_deg": phase_deg}
        row.update(paired_delta(result, external_sham, external_metrics))
        rows.append(row)
        control = ControlSpec("closed_loop", config.control_gain_cycles_per_s, None, phase)
        result = simulate_schedule(config, f0, trial, internal_epochs, no_driver, control)
        row = {"analysis": "internal_closed_loop_phase_disruption", "frequency_hz": f0, "replicate": replicate, "phase_offset_deg": phase_deg}
        row.update(paired_delta(result, internal_sham, internal_metrics))
        rows.append(row)
    return rows


def k_worker(job: tuple[Config, float, int]) -> list[dict[str, object]]:
    config, f0, replicate = job
    rows: list[dict[str, object]] = []
    steps = sum(n_steps(x, config.dt) for x in (config.pre_s, config.intervention_s, config.recovery_s))
    trial = make_trial(config, f0, replicate, steps)
    no_driver = DriverSpec(False, 0.0, f0)
    for k in K_VALUES:
        epochs = disruption_epochs(config, "internal_sync", k)
        sham = simulate_schedule(config, f0, trial, epochs, no_driver, ControlSpec("closed_loop", 0.0, None, math.pi))
        control = simulate_schedule(config, f0, trial, epochs, no_driver, ControlSpec("closed_loop", config.control_gain_cycles_per_s, None, math.pi))
        row = {"analysis": "k_dependent_phase_disruption", "frequency_hz": f0, "replicate": replicate, "K_cycles_per_s": k}
        row.update(paired_delta(control, sham, ("intervention_mean_R1", "intervention_mean_R2", "intervention_mean_R4")))
        rows.append(row)
    return rows


def maintenance_condition_driver(config: Config, f0: float, condition: str) -> DriverSpec:
    if condition == "no_driver":
        return DriverSpec(False, 0.0, f0)
    if condition == "matched_driver":
        return DriverSpec(True, config.maintenance_driver_gain_cycles_per_s, f0)
    if condition == "near_detuned_driver":
        return DriverSpec(True, config.maintenance_driver_gain_cycles_per_s, f0 + 2.0)
    if condition == "far_detuned_driver":
        return DriverSpec(True, config.maintenance_driver_gain_cycles_per_s, f0 + 6.0)
    if condition == "jittered_driver":
        return DriverSpec(True, config.maintenance_driver_gain_cycles_per_s, f0, jitter_level=0.50)
    raise ValueError(condition)


def maintenance_worker(job: tuple[Config, float, int]) -> list[dict[str, object]]:
    config, f0, replicate = job
    epochs = maintenance_epochs(config)
    steps = epochs[-1].end_step
    trial = make_trial(config, f0, replicate, steps)
    rows: list[dict[str, object]] = []
    for condition in MAINTENANCE_CONDITIONS:
        if condition in {"near_detuned_driver", "far_detuned_driver"}:
            shifts = (-2.0, 2.0) if condition == "near_detuned_driver" else (-6.0, 6.0)
            partials = []
            for shift in shifts:
                driver = DriverSpec(True, config.maintenance_driver_gain_cycles_per_s, f0 + shift)
                partials.append(simulate_schedule(config, f0, trial, epochs, driver, ControlSpec("open_loop", 0.0, None, 0.0)))
            metrics = partials[0].keys()
            result = {k: mean(float(p.get(k, float("nan"))) for p in partials) for k in metrics if k != "trace"}
        else:
            driver = maintenance_condition_driver(config, f0, condition)
            result = simulate_schedule(config, f0, trial, epochs, driver, ControlSpec("open_loop", 0.0, None, 0.0))
        row = {"condition": condition, "frequency_hz": f0, "replicate": replicate}
        for src, dest in (
            ("test_mean_R1", "test_mean_R1"),
            ("test_lock_fraction", "test_lock_fraction"),
            ("test_longest_lock_run_s", "test_longest_lock_run_s"),
            ("test_driver_locking_plv", "test_driver_locking_plv"),
            ("post_driver_recovery_mean_R1", "post_driver_recovery_mean_R1"),
        ):
            row[dest] = result.get(src, float("nan"))
        rows.append(row)
    return rows


def supplement_worker(job: tuple[Config, float, int]) -> list[dict[str, object]]:
    config, f0, replicate = job
    rows: list[dict[str, object]] = []
    steps = sum(n_steps(x, config.dt) for x in (config.pre_s, config.intervention_s, config.recovery_s))
    trial = make_trial(config, f0, replicate, steps)
    driver = DriverSpec(True, config.external_driver_gain_cycles_per_s, f0)
    no_driver = DriverSpec(False, 0.0, f0)
    ext_epochs = disruption_epochs(config, "external_driver", config.external_k_cycles_per_s)
    int_epochs = disruption_epochs(config, "internal_sync", config.internal_k_cycles_per_s)
    sham_ext = simulate_schedule(config, f0, trial, ext_epochs, driver, ControlSpec("open_loop", 0.0, f0, math.pi))
    sham_int = simulate_schedule(config, f0, trial, int_epochs, no_driver, ControlSpec("closed_loop", 0.0, None, math.pi))
    for delay in SUPPLEMENT_DELAYS:
        ctl = ControlSpec("closed_loop", config.control_gain_cycles_per_s, None, math.pi, delay_cycles=delay)
        result = simulate_schedule(config, f0, trial, int_epochs, no_driver, ctl)
        row = {"analysis": "delay_scan", "frequency_hz": f0, "replicate": replicate, "scan_value": delay}
        row.update(paired_delta(result, sham_int, ("intervention_mean_R1",)))
        rows.append(row)
    for jitter in SUPPLEMENT_JITTER:
        ctl = ControlSpec("open_loop", config.control_gain_cycles_per_s, f0, math.pi, jitter_level=jitter)
        result = simulate_schedule(config, f0, trial, ext_epochs, driver, ctl)
        row = {"analysis": "jitter_scan", "frequency_hz": f0, "replicate": replicate, "scan_value": jitter}
        row.update(paired_delta(result, sham_ext, ("intervention_driver_locking_plv",)))
        rows.append(row)
    return rows


# =============================================================================
# 8. EXECUTION, SUMMARIES, AND CSV EXPORT
# =============================================================================

def run_jobs(config: Config, jobs: list[tuple], worker: Callable[[tuple], list[dict[str, object]]]) -> list[dict[str, object]]:
    if config.workers <= 1 or len(jobs) <= 1:
        rows: list[dict[str, object]] = []
        for job in jobs:
            rows.extend(worker(job))
        return rows
    import concurrent.futures as cf
    rows = []
    with cf.ProcessPoolExecutor(max_workers=config.workers) as ex:
        for part in ex.map(worker, jobs):
            rows.extend(part)
    return rows


def grouped_summary(rows: list[dict[str, object]], keys: Sequence[str], metrics: Sequence[str]) -> list[dict[str, object]]:
    # Frequency templates are fixed model templates, not independent statistical samples.
    # Therefore each metric is first averaged across frequencies within each replicate,
    # and confidence intervals are then computed across replicate-level blocks.
    replicate_groups: dict[tuple, list[dict[str, object]]] = {}
    for row in rows:
        replicate = int(row.get("replicate", 0))
        group_key = tuple(row.get(k) for k in keys)
        replicate_groups.setdefault(group_key + (replicate,), []).append(row)

    grouped_values: dict[tuple, dict[str, list[float]]] = {}
    for combined_key, members in replicate_groups.items():
        group_key = combined_key[:-1]
        bucket = grouped_values.setdefault(group_key, {metric: [] for metric in metrics})
        for metric in metrics:
            bucket[metric].append(mean(float(member.get(metric, float("nan"))) for member in members))

    out: list[dict[str, object]] = []
    for group_key, metric_values in sorted(grouped_values.items(), key=lambda item: item[0]):
        base = {k: v for k, v in zip(keys, group_key)}
        for metric in metrics:
            rec = dict(base)
            rec["metric"] = metric
            rec.update(ci_stats(metric_values[metric]))
            out.append(rec)
    return out


def replicate_pooled_contrasts(phase_rows: list[dict[str, object]], maintenance_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    contrasts: list[dict[str, object]] = []
    for analysis, phase in (("external_driver_phase_disruption", 180), ("internal_closed_loop_phase_disruption", 180)):
        rows = [r for r in phase_rows if r["analysis"] == analysis and int(r["phase_offset_deg"]) == phase]
        for metric in sorted(k for k in rows[0] if k.startswith("delta_")) if rows else []:
            by_rep: dict[int, list[float]] = {}
            for row in rows:
                by_rep.setdefault(int(row["replicate"]), []).append(float(row.get(metric, float("nan"))))
            stats = ci_stats(mean(vals) for vals in by_rep.values())
            contrasts.append({"contrast": f"{analysis}_180__{metric}", **stats})
    for condition in ("matched_driver", "near_detuned_driver", "far_detuned_driver", "jittered_driver"):
        rows = [r for r in maintenance_rows if r["condition"] in {"no_driver", condition}]
        for metric in ("test_mean_R1", "test_lock_fraction", "test_longest_lock_run_s", "post_driver_recovery_mean_R1"):
            by_rep: dict[int, dict[str, list[float]]] = {}
            for row in rows:
                by_rep.setdefault(int(row["replicate"]), {}).setdefault(str(row["condition"]), []).append(float(row.get(metric, float("nan"))))
            deltas = []
            for vals in by_rep.values():
                if condition in vals and "no_driver" in vals:
                    deltas.append(mean(vals[condition]) - mean(vals["no_driver"]))
            contrasts.append({"contrast": f"{condition}_minus_no_driver__{metric}", **ci_stats(deltas)})
    return contrasts


def write_csv(path: Path, rows: Sequence[dict[str, object]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(output: Path, config: Config) -> None:
    write_csv(output / "configuration_parameters.csv", [{"parameter": k, "value": v} for k, v in asdict(config).items()], ["parameter", "value"])
    code_bytes = Path(__file__).read_bytes()
    write_csv(output / "environment_manifest.csv", [
        {"key": "script_version", "value": SCRIPT_VERSION},
        {"key": "script", "value": Path(__file__).name},
        {"key": "python", "value": sys.version.replace("\n", " ")},
        {"key": "matplotlib_version", "value": matplotlib.__version__},
        {"key": "scipy_version", "value": scipy.__version__},
        {"key": "platform", "value": platform.platform()},
        {"key": "cpu_count", "value": os.cpu_count()},
        {"key": "workers", "value": config.workers},
        {"key": "code_sha256", "value": hashlib.sha256(code_bytes).hexdigest()},
    ], ["key", "value"])


# =============================================================================
# 9. FIGURE GENERATION
# =============================================================================

def series_from_summary(summary: list[dict[str, object]], selector: dict[str, object], metric: str, x_key: str) -> list[tuple[float, float, float]]:
    rows = []
    for row in summary:
        if row.get("metric") != metric:
            continue
        if all(row.get(k) == v for k, v in selector.items()):
            rows.append((float(row[x_key]), float(row["mean"]), float(row.get("ci95_halfwidth", 0.0))))
    return sorted(rows)


def plot_lines(path_base: Path, title: str, panels: list[dict[str, object]]) -> None:
    n_panels = len(panels)
    if n_panels == 1:
        fig, axes_obj = plt.subplots(1, 1, figsize=(PANEL_WIDTH_IN, PANEL_HEIGHT_IN), constrained_layout=True)
        axes = [axes_obj]
    else:
        fig, axes_obj = plt.subplots(1, n_panels, figsize=(PANEL_WIDTH_IN * n_panels, PANEL_HEIGHT_IN), constrained_layout=True)
        axes = list(axes_obj)
    if title:
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    for ax, panel in zip(axes, panels):
        if panel.get("title"):
            ax.set_title(str(panel["title"]), loc="left", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
        plotted = False
        for series in panel["series"]:
            points = [(x, y, e) for x, y, e in series["points"] if math.isfinite(y)]
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            es = [p[2] if math.isfinite(p[2]) else 0.0 for p in points]
            ax.errorbar(xs, ys, yerr=es, marker="o", linewidth=LINE_WIDTH, capsize=CAPSIZE, label=series["label"])
            plotted = True
        ax.axhline(0.0, linewidth=0.9, linestyle="--", alpha=0.6)
        ax.set_xlabel(str(panel["xlabel"]), fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(str(panel["ylabel"]), fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.22)
        if plotted:
            ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    path_base.with_suffix(".png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=FIGURE_DPI)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def condition_label(name: str) -> str:
    labels = {
        "no_driver": "no driver",
        "matched_driver": "matched",
        "near_detuned_driver": "near detuned",
        "far_detuned_driver": "far detuned",
        "jittered_driver": "jittered",
    }
    return labels.get(name, name)


def plot_driver_maintenance(path_base: Path, maint_summary: list[dict[str, object]]) -> None:
    panels = [
        (
            "A. Test-epoch synchrony",
            "test_mean_R1",
            r"Mean synchrony, $R_1$",
            list(MAINTENANCE_CONDITIONS),
        ),
        (
            "B. Driver-locking PLV",
            "test_driver_locking_plv",
            "Driver-locking PLV",
            [c for c in MAINTENANCE_CONDITIONS if c != "no_driver"],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(PANEL_WIDTH_IN * 2, PANEL_HEIGHT_IN), constrained_layout=True)
    fig.suptitle("Matched-driver maintenance of elevated synchrony", fontsize=TITLE_FONTSIZE, fontweight="bold")
    for ax, (panel_title, metric, ylabel, conditions) in zip(axes, panels):
        by_condition = {str(r["condition"]): r for r in maint_summary if r.get("metric") == metric}
        x = list(range(len(conditions)))
        y = [float(by_condition[c]["mean"]) for c in conditions]
        e = [float(by_condition[c].get("ci95_halfwidth", 0.0)) for c in conditions]
        colors = [MATCHED_COLOR if c == "matched_driver" else CONTROL_COLOR for c in conditions]
        ax.bar(x, y, yerr=e, capsize=CAPSIZE, color=colors, edgecolor="none", error_kw={"ecolor": ERROR_COLOR, "elinewidth": 1.0})
        ax.set_title(panel_title, loc="left", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([condition_label(c) for c in conditions], rotation=35, ha="right")
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25)
    path_base.with_suffix(".png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=FIGURE_DPI)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def plot_all(output: Path, phase_summary: list[dict[str, object]], k_summary: list[dict[str, object]], maint_summary: list[dict[str, object]], supplement_summary: list[dict[str, object]] | None) -> None:
    fig = output / "figures"
    plot_lines(fig / "Fig4_external_driver_phase_disruption", "External-driver phase disruption", [
        {"title": "A. Pooled driver-locking PLV", "ylabel": r"$\Delta$ pooled driver-locking PLV", "xlabel": "Control phase offset (deg)", "series": [{"label": "Pooled driver-locking PLV", "points": series_from_summary(phase_summary, {"analysis": "external_driver_phase_disruption"}, "delta_intervention_driver_locking_plv", "phase_offset_deg")}]},
        {"title": "B. Order-parameter changes", "ylabel": r"$\Delta$ order parameter", "xlabel": "Control phase offset (deg)", "series": [
            {"label": r"$\Delta R_1$", "points": series_from_summary(phase_summary, {"analysis": "external_driver_phase_disruption"}, "delta_intervention_mean_R1", "phase_offset_deg")},
            {"label": r"$\Delta R_2$", "points": series_from_summary(phase_summary, {"analysis": "external_driver_phase_disruption"}, "delta_intervention_mean_R2", "phase_offset_deg")},
            {"label": r"$\Delta R_4$", "points": series_from_summary(phase_summary, {"analysis": "external_driver_phase_disruption"}, "delta_intervention_mean_R4", "phase_offset_deg")},
        ]},
    ])
    plot_lines(fig / "Fig5_internal_closed_loop_phase_disruption", "Internal closed-loop phase disruption", [
        {"title": r"A. Mean synchrony, $R_1$", "ylabel": r"$\Delta R_1$", "xlabel": "Closed-loop phase offset (deg)", "series": [{"label": r"$\Delta R_1$", "points": series_from_summary(phase_summary, {"analysis": "internal_closed_loop_phase_disruption"}, "delta_intervention_mean_R1", "phase_offset_deg")}]},
        {"title": "B. Order-parameter changes", "ylabel": r"$\Delta$ order parameter", "xlabel": "Closed-loop phase offset (deg)", "series": [
            {"label": r"$\Delta R_1$", "points": series_from_summary(phase_summary, {"analysis": "internal_closed_loop_phase_disruption"}, "delta_intervention_mean_R1", "phase_offset_deg")},
            {"label": r"$\Delta R_2$", "points": series_from_summary(phase_summary, {"analysis": "internal_closed_loop_phase_disruption"}, "delta_intervention_mean_R2", "phase_offset_deg")},
            {"label": r"$\Delta R_4$", "points": series_from_summary(phase_summary, {"analysis": "internal_closed_loop_phase_disruption"}, "delta_intervention_mean_R4", "phase_offset_deg")},
        ]},
    ])
    plot_lines(fig / "Fig6_K_dependent_phase_disruption", "State dependence of anti-phase disruption", [
        {"title": r"Anti-phase control across coupling regimes", "ylabel": r"$\Delta R_1$", "xlabel": "Coupling strength, K (cycles/s)", "series": [{"label": "Anti-phase control", "points": series_from_summary(k_summary, {"analysis": "k_dependent_phase_disruption"}, "delta_intervention_mean_R1", "K_cycles_per_s")}]},
    ])
    plot_driver_maintenance(fig / "Fig7_driver_maintenance", maint_summary)
    if supplement_summary is not None:
        plot_lines(fig / "FigS3_delay_scan", "Control-delay dependence", [
            {"title": r"Timing-dependent reversal of anti-phase control", "ylabel": r"$\Delta R_1$", "xlabel": "Control delay (cycles)", "series": [{"label": "Control delay", "points": series_from_summary(supplement_summary, {"analysis": "delay_scan"}, "delta_intervention_mean_R1", "scan_value")}]},
        ])
        plot_lines(fig / "FigS4_jitter_scan", "Anti-phase control jitter dependence", [
            {"title": r"Coherence-dependent loss of disruption", "ylabel": r"$\Delta$ pooled driver-locking PLV", "xlabel": "Control phase jitter", "series": [{"label": "Control phase jitter", "points": series_from_summary(supplement_summary, {"analysis": "jitter_scan"}, "delta_intervention_driver_locking_plv", "scan_value")}]},
        ])


# =============================================================================
# 10. PIPELINE, SELF-CHECKS, CLI, AND GUI
# =============================================================================

def run_pipeline(config: Config, output: Path) -> dict[str, list[dict[str, object]]]:
    validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output}", flush=True)
    write_manifest(output, config)
    jobs = [(config, f0, rep) for f0 in frequencies(config) for rep in range(config.n_replicates)]
    print("Running phase-disruption scans", flush=True)
    phase_rows = sorted(run_jobs(config, jobs, phase_worker), key=lambda r: (r["analysis"], r["phase_offset_deg"], r["frequency_hz"], r["replicate"]))
    print("Running K-dependent susceptibility scan", flush=True)
    k_rows = sorted(run_jobs(config, jobs, k_worker), key=lambda r: (r["K_cycles_per_s"], r["frequency_hz"], r["replicate"]))
    print("Running driver-maintenance analysis", flush=True)
    maintenance_rows = sorted(run_jobs(config, jobs, maintenance_worker), key=lambda r: (r["condition"], r["frequency_hz"], r["replicate"]))
    phase_summary = grouped_summary(phase_rows, ["analysis", "phase_offset_deg"], ["delta_intervention_driver_locking_plv", "delta_intervention_mean_R1", "delta_intervention_mean_R2", "delta_intervention_mean_R4"])
    k_summary = grouped_summary(k_rows, ["analysis", "K_cycles_per_s"], ["delta_intervention_mean_R1", "delta_intervention_mean_R2", "delta_intervention_mean_R4"])
    maintenance_summary = grouped_summary(maintenance_rows, ["condition"], ["test_mean_R1", "test_lock_fraction", "test_longest_lock_run_s", "test_driver_locking_plv", "post_driver_recovery_mean_R1"])
    contrasts = replicate_pooled_contrasts(phase_rows, maintenance_rows)
    main_results = []
    main_results.extend(phase_rows)
    main_results.extend(k_rows)
    for row in maintenance_rows:
        merged = {"analysis": "driver_maintenance", **row}
        main_results.append(merged)
    main_summary = []
    main_summary.extend(phase_summary)
    main_summary.extend(k_summary)
    for row in maintenance_summary:
        merged = {"analysis": "driver_maintenance", **row}
        main_summary.append(merged)
    write_csv(output / "main_results.csv", main_results)
    write_csv(output / "main_summary.csv", main_summary)
    write_csv(output / "paired_contrasts.csv", contrasts)
    supplement_summary = None
    supplement_rows: list[dict[str, object]] = []
    if config.include_supplement:
        print("Running supplementary delay/jitter analyses", flush=True)
        supplement_rows = sorted(run_jobs(config, jobs, supplement_worker), key=lambda r: (r["analysis"], r["scan_value"], r["frequency_hz"], r["replicate"]))
        supplement_summary = grouped_summary(supplement_rows, ["analysis", "scan_value"], ["delta_intervention_mean_R1", "delta_intervention_driver_locking_plv"])
        write_csv(output / "supplementary_results.csv", supplement_rows)
        write_csv(output / "supplementary_summary.csv", supplement_summary)
    print("Rendering figures", flush=True)
    plot_all(output, phase_summary, k_summary, maintenance_summary, supplement_summary)
    print("Done", flush=True)
    return {"phase": phase_rows, "k": k_rows, "maintenance": maintenance_rows, "phase_summary": phase_summary, "k_summary": k_summary, "maintenance_summary": maintenance_summary, "contrasts": contrasts, "supplement": supplement_rows}


def required_outputs(config: Config) -> tuple[str, ...]:
    if config.include_supplement:
        return CSV_EXPORTS + FIGURE_EXPORTS
    csv_exports = tuple(name for name in CSV_EXPORTS if name not in {"supplementary_results.csv", "supplementary_summary.csv", "trace_examples.csv"})
    figure_exports = tuple(name for name in FIGURE_EXPORTS if not name.startswith("figures/FigS"))
    return csv_exports + figure_exports


def assert_required_outputs(output: Path, config: Config) -> None:
    missing = [name for name in required_outputs(config) if not (output / name).exists() or (output / name).stat().st_size == 0]
    if missing:
        raise AssertionError(f"Missing required outputs: {missing}")


def run_self_checks() -> None:
    cfg = quick_config(Config(n_oscillators=12, n_replicates=1, workers=1, single_frequency=True))
    validate_config(cfg)
    f0 = 10.0
    steps = sum(n_steps(x, cfg.dt) for x in (cfg.pre_s, cfg.intervention_s, cfg.recovery_s))
    trial = make_trial(cfg, f0, 0, steps)
    epochs = disruption_epochs(cfg, "internal_sync", cfg.internal_k_cycles_per_s)
    no_driver = DriverSpec(False, 0.0, f0)
    sham = simulate_schedule(cfg, f0, trial, epochs, no_driver, ControlSpec("closed_loop", 0.0, None, math.pi))
    zero = simulate_schedule(cfg, f0, trial, epochs, no_driver, ControlSpec("closed_loop", 0.0, None, math.pi))
    assert abs(float(sham["intervention_mean_R1"]) - float(zero["intervention_mean_R1"])) < 1e-15
    net = cfg.external_driver_gain_cycles_per_s + cfg.external_driver_gain_cycles_per_s * complex(math.cos(math.pi), math.sin(math.pi))
    assert abs(net) < 1e-12
    phases = [0.0] * 10 + [math.pi] * 10
    r1, _ = order_parameter(phases, 1)
    r2, _ = order_parameter(phases, 2)
    assert r1 < 1e-12 and abs(r2 - 1.0) < 1e-12
    assert abs(circular_phase_difference(math.radians(179), math.radians(-179))) < math.radians(3)

    # Zero-delay closed-loop control must use the current population phase, not
    # the previous sample. Positive delays use previous samples from history.
    assert closed_loop_reference_phase(1.234, [0.10, 0.20, 0.30], 0) == 1.234
    assert closed_loop_reference_phase(1.234, [0.10, 0.20, 0.30], 1) == 0.30
    assert closed_loop_reference_phase(1.234, [0.10, 0.20, 0.30], 2) == 0.20

    # Check that the population-frame driver-locking PLV is algebraically
    # equivalent to pooled oscillator-driver phase concentration for a shared driver.
    demo_phases = [0.0, math.pi / 2.0, math.pi]
    demo_driver_phase = math.pi / 4.0
    demo_r1, demo_psi = order_parameter(demo_phases, 1)
    via_order_parameter = complex(demo_r1 * math.cos(demo_psi - demo_driver_phase), demo_r1 * math.sin(demo_psi - demo_driver_phase))
    explicit_pooled = sum(complex(math.cos(x - demo_driver_phase), math.sin(x - demo_driver_phase)) for x in demo_phases) / len(demo_phases)
    assert abs(via_order_parameter - explicit_pooled) < 1e-12
    start = time.time()
    out = Path("_self_check_quick_outputs")
    if out.exists():
        for root, dirs, files in os.walk(out, topdown=False):
            for name in files:
                Path(root, name).unlink()
            for name in dirs:
                Path(root, name).rmdir()
        out.rmdir()
    run_pipeline(cfg, out)
    assert_required_outputs(out, cfg)
    assert time.time() - start < 60.0



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-core Kuramoto phase-disruption model with pooled driver-locking PLV metrics")
    parser.add_argument("--no-gui", action="store_true", help="Run from the command line without opening the GUI.")
    parser.add_argument("--output", type=Path, default=Path("kuramoto_phase_disruption_outputs"))
    parser.add_argument("--n-oscillators", type=int, default=Config.n_oscillators)
    parser.add_argument("--replicates", type=int, default=Config.n_replicates)
    parser.add_argument("--workers", type=int, default=Config.workers)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--dt", type=float, default=Config.dt)
    parser.add_argument("--pre-s", type=float, default=Config.pre_s)
    parser.add_argument("--intervention-s", type=float, default=Config.intervention_s)
    parser.add_argument("--recovery-s", type=float, default=Config.recovery_s)
    parser.add_argument("--maintenance-induction-s", type=float, default=Config.maintenance_induction_s)
    parser.add_argument("--maintenance-test-s", type=float, default=Config.maintenance_test_s)
    parser.add_argument("--maintenance-recovery-s", type=float, default=Config.maintenance_recovery_s)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--single-frequency", action="store_true")
    parser.add_argument("--no-supplement", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        n_oscillators=args.n_oscillators,
        n_replicates=args.replicates,
        workers=args.workers,
        seed=args.seed,
        dt=args.dt,
        pre_s=args.pre_s,
        intervention_s=args.intervention_s,
        recovery_s=args.recovery_s,
        maintenance_induction_s=args.maintenance_induction_s,
        maintenance_test_s=args.maintenance_test_s,
        maintenance_recovery_s=args.maintenance_recovery_s,
        single_frequency=args.single_frequency,
        include_supplement=not args.no_supplement,
    )


def default_output_dir() -> Path:
    return (Path(__file__).resolve().parent / "kuramoto_phase_disruption_outputs").resolve()


def launch_gui() -> None:
    import queue
    import subprocess
    import threading
    import traceback
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Phase-disruption simulation")
    root.columnconfigure(1, weight=1)
    root.rowconfigure(15, weight=1)

    output_var = tk.StringVar(value=str(default_output_dir()))
    status_var = tk.StringVar(value="Ready")
    messages: queue.Queue[str] = queue.Queue()

    def add_entry(row: int, label: str, value: object) -> tk.StringVar:
        var = tk.StringVar(value=str(value))
        ttk.Label(root, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(root, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        return var

    ttk.Label(root, text="Output folder").grid(row=0, column=0, sticky="w", padx=8, pady=8)
    ttk.Entry(root, textvariable=output_var).grid(row=0, column=1, sticky="ew", padx=8, pady=8)

    def browse() -> None:
        directory = filedialog.askdirectory(initialdir=output_var.get())
        if directory:
            output_var.set(directory)

    ttk.Button(root, text="Browse", command=browse).grid(row=0, column=2, padx=8, pady=8)

    vars_map = {
        "n_oscillators": add_entry(1, "Oscillators", Config.n_oscillators),
        "n_replicates": add_entry(2, "Replicates", Config.n_replicates),
        "workers": add_entry(3, "CPU workers", Config.workers),
        "seed": add_entry(4, "Seed", Config.seed),
        "dt": add_entry(5, "Time step dt", Config.dt),
        "pre_s": add_entry(6, "Pre epoch (s)", Config.pre_s),
        "intervention_s": add_entry(7, "Intervention epoch (s)", Config.intervention_s),
        "recovery_s": add_entry(8, "Recovery epoch (s)", Config.recovery_s),
        "maintenance_induction_s": add_entry(9, "Maintenance induction (s)", Config.maintenance_induction_s),
        "maintenance_test_s": add_entry(10, "Maintenance test (s)", Config.maintenance_test_s),
        "maintenance_recovery_s": add_entry(11, "Maintenance recovery (s)", Config.maintenance_recovery_s),
    }

    progress_bar = ttk.Progressbar(root, mode="indeterminate")
    progress_bar.grid(row=12, column=0, columnspan=3, sticky="ew", padx=8, pady=6)
    ttk.Label(root, textvariable=status_var).grid(row=13, column=0, columnspan=3, sticky="w", padx=8, pady=2)

    log = tk.Text(root, width=110, height=18)
    log.grid(row=15, column=0, columnspan=3, sticky="nsew", padx=8, pady=6)

    buttons = ttk.Frame(root)
    buttons.grid(row=16, column=0, columnspan=3, pady=8)
    run_button = ttk.Button(buttons, text="RUN")
    run_button.grid(row=0, column=0, padx=6)

    def append_log(text: str) -> None:
        log.insert("end", text + "\n")
        log.see("end")

    def poll() -> None:
        while not messages.empty():
            append_log(messages.get())
        root.after(150, poll)

    def set_running(running: bool) -> None:
        run_button.configure(state="disabled" if running else "normal")
        if running:
            progress_bar.start(10)
        else:
            progress_bar.stop()

    def cfg_from_form() -> Config:
        return Config(
            n_oscillators=int(vars_map["n_oscillators"].get()),
            n_replicates=int(vars_map["n_replicates"].get()),
            workers=int(vars_map["workers"].get()),
            seed=int(vars_map["seed"].get()),
            dt=float(vars_map["dt"].get()),
            pre_s=float(vars_map["pre_s"].get()),
            intervention_s=float(vars_map["intervention_s"].get()),
            recovery_s=float(vars_map["recovery_s"].get()),
            maintenance_induction_s=float(vars_map["maintenance_induction_s"].get()),
            maintenance_test_s=float(vars_map["maintenance_test_s"].get()),
            maintenance_recovery_s=float(vars_map["maintenance_recovery_s"].get()),
            include_supplement=True,
        )

    def open_output_folder(path: Path) -> None:
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception:
            pass

    def start() -> None:
        try:
            cfg = cfg_from_form()
            validate_config(cfg)
            out = Path(output_var.get()).expanduser().resolve()
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--no-gui",
            "--output", str(out),
            "--n-oscillators", str(cfg.n_oscillators),
            "--replicates", str(cfg.n_replicates),
            "--workers", str(cfg.workers),
            "--seed", str(cfg.seed),
            "--dt", str(cfg.dt),
            "--pre-s", str(cfg.pre_s),
            "--intervention-s", str(cfg.intervention_s),
            "--recovery-s", str(cfg.recovery_s),
            "--maintenance-induction-s", str(cfg.maintenance_induction_s),
            "--maintenance-test-s", str(cfg.maintenance_test_s),
            "--maintenance-recovery-s", str(cfg.maintenance_recovery_s),
        ]

        log.delete("1.0", "end")
        append_log("Starting run")
        append_log("Output folder: " + str(out))
        status_var.set("Running")
        set_running(True)

        def worker() -> None:
            return_code = -1
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert process.stdout is not None
                for line in process.stdout:
                    messages.put(line.rstrip())
                return_code = process.wait()
                messages.put(f"Process finished with exit code {return_code}")
                root.after(0, lambda: status_var.set("Done" if return_code == 0 else "Error"))
                if return_code == 0:
                    root.after(0, lambda: open_output_folder(out))
                else:
                    root.after(0, lambda: messagebox.showerror("Analysis failed", f"Process exited with code {return_code}"))
            except Exception:
                messages.put(traceback.format_exc())
                root.after(0, lambda: status_var.set("Error"))
                root.after(0, lambda: messagebox.showerror("Analysis failed", "Could not launch or monitor the subprocess."))
            finally:
                root.after(0, lambda: set_running(False))

        threading.Thread(target=worker, daemon=True).start()

    run_button.configure(command=start)
    poll()
    root.mainloop()

def main() -> None:
    args = parse_args()
    if args.self_check:
        run_self_checks()
        print("Self-checks passed", flush=True)
        return
    if not args.no_gui:
        launch_gui()
        return
    cfg = config_from_args(args)
    if args.quick:
        cfg = quick_config(cfg)
    out = args.output.expanduser().resolve()
    run_pipeline(cfg, out)
    assert_required_outputs(out, cfg)
    print(f"Outputs written to {out}", flush=True)


if __name__ == "__main__":
    main()
