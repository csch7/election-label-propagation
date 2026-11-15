## Election Label Propagation

Repository for the UNC Chapel Hill STOR-565 final project.

Compare how label propagation and kNN perform when predicting election results in counties by either demographic similarity or locale.

**Task.** Predict which party won each county in the 2024 US presidential election (Democratic = 1, Republican = 0), given the results of some other counties, using two ways of deciding which counties are "close":

- **border graph** (locale): two counties are linked if they share a border;
- **demographic graph**: two counties are linked if their age and race/ethnicity make-up is similar.

A third, **combined graph** blends the two. Two models are compared on each graph:

- **kNN**: a county takes the majority (similarity-weighted) vote of the k most similar counties whose result is known.
- **label propagation**: known results are spread along the graph's edges, so a county is influenced by counties several hops away, not only its direct neighbours.

## Running it

```bash
python main.py                                   # full comparison, 10 seeds, block split
python main.py --seeds 3                         # quicker
python main.py --splits stratified block         # also run the stratified split
python main.py --train-frac 0.1 0.4 max          # vary how many county results are known
python main.py --previous-year                   # also run the previous-year seeding experiment
```

```bash
python make_figures.py             # the 3 figures -> results/figures/
python make_figures.py --only charts
python make_figures.py --dots     # draw the map without downloading county boundaries
```

Every setting (split sizes, parameter grids, years, defaults for the command line) is in `config/config.yml`; edit it there rather than in the code. Command-line options override the experiment settings for one run.

`pip install -r requirements.txt`. Only numpy, pandas and PyYAML are needed to run the comparison; matplotlib is needed for the figures. Output goes to `results/runs/<date>_<experiment>[_<variant>].csv` (a rerun on the same day overwrites that day's file) and a summary table is printed. `python -m src.graphs.adjacency` and `python -m src.graphs.demographics` rebuild the saved graphs; `python -m src.evaluation.splits` regenerates the example split files.

## Layout

```
data/raw/            source files (counties, borders, census, votes)
data/preprocessed/   saved graphs (.npz), demographic features, example splits
results/runs/        one CSV per experiment run
config/config.yml    all tunable settings (see Parameters)
src/config.py        loads the config
src/graphs/          border and demographic graph construction, save/load
src/models/          knn.py, label_propagation.py
src/evaluation/      labels, splits, cross-validated tuning, metrics, results paths
src/experiments/     previous-year seeding experiment
src/visualization/   charts and county maps
make_figures.py      makes the figures from the saved runs
results/figures/     generated figures
main.py              runs the full comparison
```

## Design decisions

**Data.** Counties are identified by 5-digit GEOID. Labels come from the 2024 presidential results: the Democratic candidate's votes against the Republican's, using the `TOTAL` row where a county reports by voting mode and summing all modes otherwise (ties go to Republican). Features come from the Census county population estimates (most recent year in the file): the share of the population in each of the 18 age groups and in 7 race/ethnicity groups, 25 features in all. A county is *eligible* if it has both features and a label (3,103 counties; about 14% voted Democratic). Puerto Rico, territories, and a few counties missing from either source are left out of the split, but stay in the graphs as unlabelled nodes.

**Border graph.** Built from `Bordering Counties.csv`, which lists each border once, so it is made symmetric. 3,235 counties, 9,297 edges, 21 counties with no neighbours (islands).

**Demographic graph.** Similarity between counties is the cosine similarity of their standardised features. The default graph links each county to its 5 most similar counties (made symmetric, 3,144 counties, 12,353 edges), with the similarity as the edge weight. We chose this over a similarity threshold because a threshold leaves many counties with no edges (at 0.95, about 77% of counties had none) while others get hundreds. A threshold or a hybrid of both (threshold plus a top-k floor) can still be built with `build_demographic_graph(method=...)`, but the comparison uses the k-nearest-neighbours graph.

**Combined graph.** A weighted blend, `lam * border + (1 - lam) * demographic`, so `lam = 0` is the demographic graph and `lam = 1` the border graph. It covers the 3,144 counties in both graphs (the other 91 have no census data and are never in the split). The two sources are on different scales, so before blending each graph's edges are rescaled to an average weighted degree of 1 (so `lam` is the share of a county's total edge weight that comes from borders), and for kNN the cosine similarity and the geographic proximity are each standardised. `lam` is tuned by cross-validation along with the other parameters, from `blend_lambdas` in the config. Counties without borders (islands) still get edges from the demographic part, so no county is isolated. The previous-year experiment does not include the combined graph.

**kNN does not use the graph's edges.** kNN runs on the similarity between *every* pair of counties: cosine similarity of features for the demographic graph, and geographic proximity of county centres for the border graph (the nearest counties, whether or not they border). An earlier version restricted kNN to a county's direct graph neighbours. With only a few counties labelled, most test counties then had no labelled neighbour (at 5% known, about 70% had none), so kNN fell back to the overall average and looked much worse than label propagation for reasons unrelated to the method. That version was removed. Label propagation, by contrast, uses the graph's edges and spreads labels across many hops.

**Train/test split.** 20% of eligible counties are held out as the test set; the other 80% is the *training pool*. kNN and label propagation have no training phase, so the counties whose labels are known are the training counties. **The reported results use the block split** (`report_split` and `split_strategies` in the config); the stratified split is still available with `--splits stratified`. The two ways of choosing the test counties:

- `stratified`: sorted by state, then winner, then the first principal component of the demographics, and one random county is taken from each block of five, so the test set mirrors each state's party mix and the demographic range.
- `block` (reported): contiguous test regions of about 10 counties are grown from random seeds by breadth-first search over the border graph, within each state. Test counties are then far from labelled ones, a harder test.

Both models are run on the *same* test counties. The graphs do not have to cover the same counties.

**Training size.** `--train-frac` is the share of *all* eligible counties whose labels are known (at most 0.8). Smaller values subsample the training pool, state-stratified. `max` uses the whole pool.

**Tuning by cross-validation.** There is no separate validation set. The known counties are split into 5 folds with the same method as the test split (stratified or block). For each parameter setting, every fold is predicted from the other four, and the best parameters and decision threshold are chosen on the pooled out-of-fold predictions. The final model is then fitted with all known counties and scored once on the test set. (It has 5/5 of the known labels rather than the 4/5 used inside each fold, the usual cross-validation convention.) The test set is never used for tuning.

**Decision threshold.** Only about 14% of counties are Democratic, so a 0.5 cutoff rarely predicts Democratic. The threshold is tuned along with the other parameters, to maximise balanced accuracy.

**Metrics.** Democratic is the positive class. We report balanced accuracy (the average recall over the two classes; the main number, since guessing Republican everywhere scores about 0.86 accuracy but only 0.5 balanced accuracy), accuracy, precision, recall, F1 and AUC. AUC is computed from the predicted probabilities and does not depend on the threshold; the others use the tuned threshold. Each experiment is repeated over several random seeds (default 10). Each seed redraws the test split, the known counties and the folds, and results are reported as mean and standard deviation over seeds. An "always Republican" baseline row is included.

**Previous-year experiment** (`--previous-year`, optional and not part of the reported results). Instead of knowing the 2024 result for the known counties, we only know their 2020 result. Those 2020 labels seed the models, which predict 2024 for the test counties. For reference, "persistence" (each test county's own 2020 result) scores about 0.98 balanced accuracy, far above any graph model, so the graph methods are only useful when a county's own history is unavailable.

## Parameters

All of these live in `config/config.yml` (with comments). Command-line options (`main.py`) override the experiment settings for a single run.

**Command line (`main.py`)**

| option | meaning | default |
|---|---|---|
| `--seeds N` | number of random repeats (each redraws the split, known counties and folds) | `seeds` |
| `--splits` | `stratified` and/or `block` test-split type | `split_strategies` (block) |
| `--train-frac F ...` | share of all counties whose labels are known; `max` = every non-test county (80%) | `train_fractions` (with `--previous-year`: `previous_year_train_fractions`) |
| `--previous-year` | also run the previous-year seeding experiment | off |

**Tuned automatically** (the values tried are the `*_grid` entries in the config)

| parameter | model | meaning | config |
|---|---|---|---|
| `k` | kNN | number of most similar known counties that vote | `knn_k_grid` |
| `weighted` | kNN | votes counted by similarity (true) or equally (false) | `knn_weighted_grid` |
| `alpha` | label propagation | how far labels spread, between 0 and 1. Low values keep a county close to its own initial label, high values let neighbours' influence travel further | `lp_alpha_grid` |
| `lam` | both, combined graph only | share of the border graph in the blend (0 = demographic only, 1 = border only) | `blend_lambdas` |
| `threshold` | both | probability above which a county is called Democratic | `threshold_grid` (start, stop, step) |

**Fixed settings** (all in `config.yml`)

| setting | meaning | default |
|---|---|---|
| `report_split` | split type shown in the figures | block |
| `test_fraction` | share of counties held out for testing | 0.2 |
| `cv_folds` | cross-validation folds used for tuning | 5 |
| `block_size` | target size of each test block in the block split | 10 counties |
| `seeds` | random repeats | 10 |
| `train_fractions`, `previous_year_train_fractions` | share of all counties whose labels are known, for a default run and for the previous-year experiment | `[max]`, `[0.05, 0.1, 0.2, 0.4, max]` |
| `demographic_graph_k` | neighbours per county in the demographic graph | 5 |
| `demographic_similarity` | similarity cutoff; only used by the `threshold` and `hybrid` graph methods, not the comparison | 0.8 |
| `geographic_scale_km` | proximity of two counties = 1 / (1 + km / scale); only changes kNN vote weights, not which counties are nearest | 100 |
| `nearest_neighbors` | default k if `KNNClassifier` is created without one | 5 |
| `lp_alpha`, `lp_max_iter`, `lp_tol` | label propagation defaults when no `alpha` is given, and its stopping rule | 0.9, 1000, 1e-6 |
| `census_year` | Census `YEAR` code for the features (null = latest in the file) | null |
| `target_year`, `previous_year` | election to predict; election used for the previous-year experiment | 2024, 2020 |
| `previous_year_knn_k_grid`, `previous_year_lp_alpha_grid` | smaller grids for the previous-year experiment | `[3, 10, 50]`, `[0.5, 0.8, 0.95]` |

## Results

Block split. Test balanced accuracy, mean over 10 seeds, by the share of all counties whose labels are known (`results/runs/2026-11-15_model_comparison.csv` has every metric and the per-seed numbers; standard deviations are about 0.01 to 0.05).

| graph | model | 5% | 10% | 20% | 40% | 80% (max) |
|---|---|---|---|---|---|---|
| border | kNN | 0.651 | 0.695 | 0.731 | 0.725 | 0.751 |
| border | label propagation | 0.635 | 0.687 | 0.715 | 0.730 | 0.728 |
| demographic | kNN | 0.768 | 0.775 | 0.801 | 0.817 | 0.842 |
| demographic | label propagation | 0.741 | 0.783 | 0.816 | 0.827 | 0.839 |
| combined | kNN | 0.769 | 0.788 | 0.822 | 0.831 | 0.854 |
| combined | label propagation | 0.746 | 0.775 | 0.810 | 0.834 | 0.855 |

- The demographic graph beats the border graph by about 10 points at every training size.
- The combined graph is ahead of the demographic graph by roughly 1 to 2 points, which is within the seed-to-seed spread, so it is suggestive rather than conclusive. Cross-validation chose the smallest border share we tried (`lam = 0.25`) in most runs, so borders add little on top of demographics. A smaller `lam` was not tried.
- kNN and label propagation are close everywhere, within the seed-to-seed spread, so we do not claim a winner. AUC tells the same story. With few known counties (5%), kNN is a little ahead on every graph.
- More known counties help both models, with most of the gain by 20% to 40%.
- For reference, always guessing Republican scores 0.5 balanced accuracy (0.86 accuracy).

## Figures

`python make_figures.py` reads the latest `results/runs/*_model_comparison.csv` and writes to `results/figures/`, for the split set by `report_split` in the config:

| file | shows |
|---|---|
| `learning_curves.png` | balanced accuracy and AUC against the share of counties known, for every graph and model |
| `metrics_heatmap.png` | all six metrics for every graph and model at the full training size |
| `map_errors.png` | right and wrong calls on the test counties for every model and graph (seed 0) |

The map refits the runs for one seed (default 0, `--seed` to change); every refit is checked against the saved metrics. County boundaries come from the US Atlas, downloaded once and cached in `results/`; if the download is unavailable (or with `--dots`), each county is drawn as a dot at its centre.

## Notes and limitations

- The block split is the harder test: its test counties are in regions with few labelled neighbours, which removes the advantage a random split gives the border graph (neighbouring counties tend to vote alike). We report only this split; the stratified split gives higher scores, especially for the border graph.
- Selecting the threshold and parameters on a validation set of only about 70 Democratic counties is noisy; the cross-validation and the repeats over seeds are there to reduce that. Differences of one or two points between models are within the seed-to-seed spread.
- Both graphs and the split use only features and borders, not votes, except that the stratified split and the folds use the labels to balance parties.
