# Constraint-Relaxation Trade-offs as a By-Product of Evolutionary Black-Box Optimization

Experiment code for the paper

> G. Misitano and J. Branke, "Constraint-Relaxation Trade-offs as a By-Product of Evolutionary
> Black-Box Optimization" (submitted to the European Journal of Operational Research).

The repository is a [Snakemake](https://snakemake.readthedocs.io) pipeline that runs every
experiment of the paper and produces its tables and figures, from the problem definitions to the
LaTeX tables. The selection operator studied in the paper,
`SingleObjectiveConstrainedRankingSelector`, is part of
[DESDEO](https://github.com/industrial-optimization-group/DESDEO) from version 2.9.0.

The run data, reference fronts and derived results are archived on Zenodo,
[doi:10.5281/zenodo.23799274](https://doi.org/10.5281/zenodo.23799274). They are described in
[DATA_README.md](DATA_README.md), which is also the README of the record.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). The exact versions of all packages are pinned
in `uv.lock`:

```
uv sync
```

## Running the pipeline

Everything goes to `results/` (set by `results_dir` in `experiment_config.yaml`). Run Snakemake
through uv, so that the scripts use the project's environment:

```
uv run snakemake --cores 8
```

This runs all 450 configurations (10 problems x 5 modes x 3 threshold levels x 3 population
sizes, 100 runs each) and the NSGA-III reference fronts. It takes a long time, and the run data
need about 16 GB. Rerunning a configuration reproduces its data file in the archive exactly.

Shorter ways to the results, with the archived data from Zenodo. `scripts/fetch_data.py`
downloads the parts of the record (resuming interrupted downloads), checks them, and unpacks them
into `results/` with their original modification times, so Snakemake does not recompute them:

- **Everything (about 17 GB):** `uv run python scripts/fetch_data.py all`. Afterwards
  `uv run snakemake --cores 8` has nothing left to do; delete any derived file to recompute it
  from the run data.
- **From the derived data alone (about 50 MB):** `uv run python scripts/fetch_data.py derived`,
  then `uv run snakemake --cores 8 --config from_summaries=true` rebuilds the tables and figures
  without the run data and fronts.
- **Some problems:** `uv run python scripts/fetch_data.py raw --problem g9 g24` (and `fronts`).

`--from-dir DIR` uses files downloaded by hand, and `--token` an access token of a Zenodo share
link while the record is not yet public.
- **Quick end-to-end check:** `uv run snakemake --configfile test_experiment_config.yaml --cores 8`
  runs the whole pipeline with 20 runs and 200 generations into `results/test_output/`.

A single configuration can be requested by its output file, e.g.

```
uv run snakemake --cores 1 results/data/branin_hypervolume_gen200_runs100_cthigh_psize12.parquet
```

## Configuration

`experiment_config.yaml` holds every setting: problems and their thresholds (`manual_thresholds`),
modes, population sizes, generations, runs, the base seed (run r uses seed 28012026 + r, so all
modes start from the same initial population), and the NSGA-II and NSGA-III operator settings.
The operator settings are DESDEO's presets written out; the scripts stop with an error if the
installed DESDEO's presets differ from them.

## Scripts

| script | rule | what it does |
|---|---|---|
| `experiment_1.py` | `generate_data` | the 100 runs of one configuration |
| `generate_front.py` | `generate_pareto_front` | NSGA-III reference front of one problem (constraints as objectives) |
| `compute_thresholds.py` | `compute_constraint_thresholds` | threshold files from the config |
| `compute_reference_optima.py` | `compute_reference_optima` | best-known optima of every relaxation (the references of the gap metrics) |
| `pooled_front.py` | `pooled_front` | pooled shadow-front of one problem |
| `summary_statistics.py` | `summary_statistics` | best-so-far objectives and hypervolume per generation and run |
| `scenario_metrics.py` | `scenario_metrics` | gap and hypervolume metrics of the relaxation scenarios |
| `mode_comparison_table.py` | `mode_comparison_table` | mode-comparison tables with Wilcoxon signed-rank tests, Holm-corrected |
| `plot_summary_figures.py` | `plot_summary_figures` | per-problem convergence figures |
| `generate_supplement_tex.py` | `generate_supplement` | figure sections of the supplementary material |
| `plot_hv_schematic.py`, `plot_mode_illustration.py` | `plot_hv_illustration`, `plot_mode_illustration` | the illustrations of the paper |
| `plot_threshold_picker.py` | (by hand) | interactive constraint plots used to pick the thresholds |
| `utils.py` | | problem builders and shared helpers |
| `fetch_data.py` | (by hand) | download and unpack the archived data from Zenodo |

## Citation

If you use this code, please cite the paper (see [CITATION.cff](CITATION.cff)).

## License

The code is licensed under the MIT License ([LICENSE](LICENSE)); the archived data under CC BY 4.0.
