# Data: Constraint-Relaxation Trade-offs as a By-Product of Evolutionary Black-Box Optimization

Run data, reference fronts and derived results for the paper

> G. Misitano and J. Branke, "Constraint-Relaxation Trade-offs as a By-Product of Evolutionary
> Black-Box Optimization" (submitted to the European Journal of Operational Research).

The code that produced everything here, and that regenerates the derived results from the run
data, is the experiment repository `shadow-price-experiments` (Snakemake pipeline, Python,
https://github.com/gialmisi/shadow-price-experiments). This archive is its `results/` directory,
without logs, published on Zenodo as https://doi.org/10.5281/zenodo.23799274.

License: CC BY 4.0.

## Contents

```
README.md                 this file
SHA256SUMS                checksums of every file in the archive
experiment_config.yaml    the experiment settings used (copy from the repository)
data/                     raw run data: 450 files, one per configuration (about 16 GB)
fronts/                   NSGA-III reference fronts and pooled fronts (about 0.8 GB)
thresholds/               relaxation thresholds per problem and level
reference/                best-known relaxed optima (the references of the gap metrics)
summary/                  per-run and per-generation metrics of every configuration
tables/                   the mode-comparison tables of the paper (means and statistical tests)
figures/                  the figures of the paper and of the supplement
timing/                   wall times behind the computational-cost figures
```

The record stores this tree in parts: `data_{problem}.tar` (the run data of one problem, 1-3.3 GB
each), `fronts.tar` and `derived.tar.gz` (everything else under `results/`, about 50 MB), next to
`README.md`, `SHA256SUMS` and `experiment_config.yaml`. Extract the parts you need in one
directory, e.g. `for f in *.tar *.tar.gz; do tar -xf "$f"; done`. The derived results alone suffice
to rebuild the tables and figures (see the end of this file).

Check the extracted files with `sha256sum -c SHA256SUMS` (add `--ignore-missing` if you extracted
only some parts).

## The experiment

- **Problems (10):** branin, mystery, cantilevered_beam, pressure_vessel, townsend, g2, g6, g8, g9,
  g24 (one to four constraints, two to twenty variables).
- **Selection modes (5):** baseline, relaxed, joint, ranking, hypervolume.
- **Threshold levels (3):** low, med, high (`thresholds/{problem}.yaml`).
- **Population sizes (3):** N = 12, 36, 60.
- **Runs:** 100 per configuration (problem x mode x level x N = 450 configurations), 200
  generations each, NSGA-II scheme with DESDEO's standard operators. Run r of every configuration
  uses the seed 28012026 + r, so all modes start from the same initial population.
- **Reference fronts:** NSGA-III, 10 000 generations, population size 200, constraints as
  additional objectives.

## File formats

All tables are Apache Parquet files; settings and small results are YAML.

### `data/{problem}_{mode}_gen200_runs100_ct{level}_psize{N}.parquet`

Every solution evaluated in the 100 runs of one configuration: (200 + 1) x N rows per run (the
initial population and 200 generations of offspring). Columns:

| column | meaning |
|---|---|
| `x_1`, ..., `x_n` | decision variables |
| `f_1` | objective value as defined by the problem |
| `f_1_min` | objective value in minimization form (used in all analyses) |
| `c_1`, ..., `c_k` | constraint values; `c_j <= 0` is feasible |
| `generation` | generation in which the solution was evaluated (0 = initial population) |
| `run` | run index 0-99 |
| `seed` | random seed of the run |

The Parquet metadata holds `notes` (problem, mode, level, population size, thresholds),
`ea_options` (the complete algorithm settings as JSON) and `created_at`.

### `fronts/`

- `{problem}_gen10000_psize200.parquet`: the NSGA-III reference front (decision variables,
  objective and constraints, their minimization forms, generation).
- `{problem}_pooled.parquet`: the pooled shadow-front, the non-dominated set in (objective,
  relaxation used per constraint) of the NSGA-III front, every run solution and the reference
  solutions, inside the box of the highest thresholds. Column `source` says where each point comes
  from (`nsga3`, `runs`, `reference`); for run points, `mode`, `ct_level`, `psize` and `run`.
- `{problem}_pooled_summary.yaml`: per level, the objective and hypervolume reached by the NSGA-III
  front alone and by the pooled front.

### `reference/{problem}.yaml`

The best-known optima per level, for the joint relaxation and for each constraint relaxed alone:
`f` (the reference), `source` (front, runs or solver), and the best value of each source
(`f_front`, `f_runs`, `f_solver`). `method` records the solver settings, `strict` the re-solved
strict optimum.

### `summary/{problem}_{mode}_gen200_runs100_ct{level}_psize{N}_*`

| file | content |
|---|---|
| `_per_run.parquet` | final values per run: best-so-far feasible objective, best-so-far threshold-feasible objective, hypervolume |
| `_summary.parquet` | the same per generation: mean, standard deviation, number of runs, 95% Student's t confidence interval |
| `_scenario_per_run.parquet` | final scenario metrics per run: `joint_gap`, `single_gap`, `single_hv`, per-constraint gaps `gap_c_j` and hypervolumes `hv_c_j` |
| `_scenario_summary.parquet` | the scenario metrics per generation, aggregated as above |
| `_summary_meta.yaml`, `_scenario_meta.yaml` | the boxes, reference points, tolerances and contributing constraints used |

### `tables/`

`mode_comparison_{metric}.parquet` (final-generation means per problem, N, level and mode) and
`mode_comparison_{metric}_stats.parquet` (the 10 pairwise two-sided Wilcoxon signed-rank tests per
configuration, Holm-corrected at α = 0.05, with the adjusted p-value in `p_holm`), for the metrics `best_so_far`, `single_gap`, `single_hv`,
`joint_gap` and `hv`, with the LaTeX tables of the paper.

### `timing/runs_2026-11-04.json`

Wall time in seconds per run, keyed `problem|mode|N|run`: 5 runs per problem, mode and population
size at the medium level, one process.

## Software versions

The runs were made with a development version of DESDEO 2.8.0 that contains the selection
operator of the paper (commit 3b8dbc86), and pymoo 0.6.2. The operator was released unchanged in
DESDEO 2.9.0, which the experiment repository now pins (`desdeo==2.9.0`, exact versions of all
packages in `uv.lock`). Rerunning a configuration with it reproduces its data file exactly
(checked on sample configurations, among them two with DESDEO 2.9.0: Branin, hypervolume mode,
and the pressure vessel, ranking mode, both at the high level with N = 12).

The run files were produced in stages:

- baseline and relaxed: before 2026-11-03;
- joint, ranking and hypervolume: rerun on 2026-11-03 after changes to the joint, ranking and
  hypervolume modes;
- pressure_vessel, relaxed and ranking: rerun on 2026-11-06 after a fix for constraints with a zero
  threshold (pressure_vessel's c_4).

The baseline and relaxed files record the ranking mode's niching setting as `kth` in their
`ea_options`, because they predate the switch of that setting to `knearest`. These two modes do
not use the niching, and rerunning them with the current settings gives identical data.

## Regenerating the derived results

With the experiment repository, put `data/` and `fronts/` under its `results/` directory and run
the Snakemake pipeline; it recomputes `reference/`, `summary/`, `tables/` and `figures/` from them.
Without the run data and fronts, with only `derived.tar.gz` extracted under `results/`,
`uv run snakemake --cores 8 --config from_summaries=true` rebuilds the tables and figures from the
thresholds, references and summaries.
