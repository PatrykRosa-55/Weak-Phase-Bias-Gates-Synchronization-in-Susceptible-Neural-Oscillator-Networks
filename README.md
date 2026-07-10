# Phase-oscillator simulations for weak phase-bias gating

This repository contains the Python simulation code used for the computational analyses reported in the manuscript **"Weak Phase-Bias Gates Synchronization in Susceptible Neural Oscillator Networks"**.

The analyses use minimal stochastic, phase-only Kuramoto-type oscillator models to test whether weak effective phase-bias inputs can:

1. increase population synchrony when they are frequency-matched and phase-stable;
2. disrupt or maintain lock-like states depending on phase relation and timing;
3. interact with transient susceptibility windows to increase synchrony or driver-locking beyond either component alone.

The models are phenomenological and phase-only. They do **not** model membrane voltage, synaptic conductances, action potentials, tissue field propagation, field dosimetry, stimulation safety, physical field amplitudes, or biological transduction mechanisms. All results should therefore be interpreted at the level of model phase dynamics and collective synchronization.

## Repository structure

The repository contains three main simulation scripts:

```text
kuramoto_phase_forcing.py
kuramoto_phase_disruption.py
psmh_phase_ignition.py
```

These scripts correspond to the three supplementary methods modules:

```text
Supplementary Methods 1 — stochastic phase-forcing model
Supplementary Methods 2 — phase disruption and maintenance simulations
Supplementary Methods 3 — phase-ignition and state-gated synchrony simulations
```

## Requirements

Recommended environment:

```text
Python >= 3.10
numpy
pandas
scipy
matplotlib
```

A minimal environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy pandas scipy matplotlib
```

The scripts include optional graphical user interfaces based on `tkinter`. In headless, server, or reviewer environments, run scripts from the command line with `--no-gui`. On minimal Linux installations, `tkinter` may require a separate system package such as `python3-tk`.

## CPU workers and computational load

All scripts support the `--workers` argument. The default worker count is automatically capped at a small number, typically 1–4 CPU workers depending on the available CPU count. This is intentional: full default runs can be computationally expensive, and using too many workers may overload a laptop or shared machine without improving reproducibility.

Recommended settings:

```bash
--workers 1   # safest option for laptops, debugging, and reviewer smoke tests
--workers 2   # reasonable default for ordinary desktops
--workers 4   # upper practical default used by the scripts
```

Avoid running several full scripts at the same time with multiple workers each. If multiple analyses are launched in parallel, use `--workers 1` for each process.

For a first check, run the self-checks and quick/small runs before launching full analyses.

## Running the full simulations

Each script can be run independently. Output folders are specified with `--output`.

### Supplementary Methods 1: phase-forcing simulations

```bash
python kuramoto_phase_forcing.py
```

This script tests whether weak frequency-matched external phase forcing increases population synchrony relative to a no-stimulus baseline, and whether the effect is reduced by detuning, jitter, irregularity, frequency dispersion, or changes in the coupling regime.

Expected main CSV outputs include:

```text
configuration_parameters.csv
environment_manifest.csv
core_metrics.csv
main_results.csv
main_summary.csv
paired_contrasts.csv
by_frequency_summary.csv
supplementary_results.csv
supplementary_summary.csv
stimulus_locking_summary.csv
stimulus_locking_by_frequency_summary.csv
```

Expected figures:

```text
Fig. 1
Fig. 2
Fig. 3
Supplementary Fig. S1
Supplementary Fig. S2
```

### Supplementary Methods 2: phase disruption and maintenance simulations

```bash
python kuramoto_phase_disruption.py 
```

This script tests whether phase-shifted control can disrupt driver-locking or internally generated synchrony, whether anti-phase effects depend on coupling strength, and whether a matched driver can maintain a previously induced synchronized state.

Expected main CSV outputs include:

```text
configuration_parameters.csv
environment_manifest.csv
main_results.csv
main_summary.csv
paired_contrasts.csv
supplementary_results.csv
supplementary_summary.csv
```

Expected figures:

```text
Fig. 4
Fig. 5
Fig. 6
Fig. 7
Supplementary Fig. S3
Supplementary Fig. S4
```

Fig. 7 displays the primary maintenance panels. The full maintenance metrics, including threshold-based and post-driver recovery metrics, are available in the CSV outputs.

### Supplementary Methods 3: phase-ignition simulations

```bash
python psmh_phase_ignition.py 
```

This script tests whether a short transient increase in effective internal coupling, interpreted operationally as a temporary susceptibility window, can interact with a weak matched phase driver to increase synchrony or driver-locking beyond either component alone.

Expected main CSV outputs include:

```text
configuration_parameters.csv
environment_manifest.csv
main_results.csv
main_summary.csv
paired_contrasts.csv
supplementary_results.csv
supplementary_summary.csv
trace_examples.csv
```

Expected figures:

```text
Fig. 8
Fig. 9
Fig. 10
Supplementary Fig. S5
```

## Self-checks and smoke tests

All three scripts include a `--self-check` flag for internal validation:

```bash
python kuramoto_phase_forcing.py --self-check
python kuramoto_phase_disruption.py --self-check
python psmh_phase_ignition.py --self-check
```

Small end-to-end smoke-test runs are also available. The first script uses `--example-small-run`; the second and third scripts use `--quick`.

```bash
python kuramoto_phase_forcing.py --no-gui --workers 1 --example-small-run --output results/sm1_small_test
python kuramoto_phase_disruption.py --no-gui --workers 1 --quick --output results/sm2_small_test
python psmh_phase_ignition.py --no-gui --workers 1 --quick --output results/sm3_small_test
```

To inspect the available command-line options for any script, run:

```bash
python script_name.py --help
```

## Output organization

Each output directory contains CSV files with raw simulation results, aggregated summaries, planned contrasts, configuration parameters, and environment information. Figures are exported in PNG and SVG formats.

The most important output types are:

```text
configuration_parameters.csv   # parameters used for the run
environment_manifest.csv       # Python/library/platform information and code hash
main_results.csv               # raw or block-level primary results
main_summary.csv               # replicate-level aggregated summaries
paired_contrasts.csv           # planned paired or descriptive contrasts
supplementary_results.csv      # supplementary scan results, where applicable
supplementary_summary.csv      # supplementary scan summaries, where applicable
trace_examples.csv             # representative traces for Supplementary Methods 3
```

The exact column set differs between modules because each module tests a different dynamical question.

## Reproducibility

All simulations use deterministic seed generation based on semantic identifiers such as frequency template, replicate number, analysis block, parameter values, and base seed. This makes outputs reproducible and independent of parallel task ordering.

The default base seed is:

```text
seed = 1234
```

Where paired contrasts are used, compared conditions share the same initial phases, natural-frequency draws, stochastic increments, and relevant jitter trajectories. This common-random-numbers design reduces Monte Carlo variance and ensures that paired differences reflect the manipulated phase input, control input, or transient term rather than independent resampling noise.

For exact reruns, use a fresh output directory or remove old outputs before rerunning a script.

## Statistical summaries

The six analyzed frequencies are treated as fixed model templates, not as independent statistical replicates. For summary statistics, results are first averaged across frequency templates within each stochastic replicate. Means, standard deviations, standard errors, and two-sided 95% confidence intervals are then computed across replicate-level values.

The confidence intervals quantify variability across stochastic realizations of the model. They should not be interpreted as population-level inference over biological neurons, animals, or humans.

Formal hypothesis tests with reported p-values are not used. Interpretation is based on effect direction, effect magnitude, confidence intervals, and planned paired or descriptive contrasts.

## Figure mapping

The scripts generate the figures used in the manuscript and supplementary information:

```text
Fig. 1 — phase-forcing condition comparison and coupling × forcing scan
Fig. 2 — coupling-strength scan
Fig. 3 — jitter and forcing-strength scans
Fig. 4 — disruption of driver-locking by phase-shifted control
Fig. 5 — internal closed-loop phase disruption
Fig. 6 — coupling-dependent susceptibility to anti-phase control
Fig. 7 — maintenance of induced synchrony by a matched driver
Fig. 8 — phase-ignition condition comparison
Fig. 9 — phase-ignition across coupling and transient-amplitude regimes
Fig. 10 — representative phase-ignition traces
Supplementary Fig. S1 — numerical and frequency-dispersion robustness
Supplementary Fig. S2 — stimulus-locking PLV across forcing conditions
Supplementary Fig. S3 — delay dependence of closed-loop anti-phase control
Supplementary Fig. S4 — jitter dependence of driver-locking disruption
Supplementary Fig. S5 — driver-locking PLV across phase-ignition regimes
```

## Interpretation limits

These simulations test model-level sufficiency of phase-dynamical mechanisms. They do not demonstrate that environmental electromagnetic fields entrain biological neural tissue, do not estimate physical field dose, and do not provide a biophysical transduction mechanism.

Terms such as `phase-forcing`, `phase-biasing`, `driver-locking`, `lock-like synchrony`, `phase-ignition`, and `state-gated ignition` refer to behavior of the implemented phase-oscillator models.

## Citation

If using or reusing this code, cite the associated manuscript.
